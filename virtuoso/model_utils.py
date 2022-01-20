from numpy import diff
import torch
import math
from .utils import find_boundaries_batch, get_softmax_by_boundary, cal_length_from_padded_beat_numbers
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def sum_with_boundary(x_split, attention_split, num_head):
    weighted_mul = torch.bmm(attention_split.transpose(1,2), x_split)
    restore_size = int(weighted_mul.size(0) / num_head)
    attention = torch.cat(weighted_mul.split(split_size=restore_size, dim=0), dim=2)
    sum_attention = torch.sum(attention, dim=1)
    return sum_attention



def make_higher_node(lower_out, attention_weights, lower_indices, higher_indices, lower_is_note=False):
    # higher_nodes = []

    similarity = attention_weights.get_attention(lower_out)
    if lower_is_note:
        boundaries = find_boundaries_batch(higher_indices)
    else:
        higher_boundaries = find_boundaries_batch(higher_indices)
        zero_shifted_lower_indices = lower_indices - lower_indices[:,0:1]
        num_zero_padded_element_by_batch = ((lower_out!=0).sum(-1)==0).sum(1)
        len_lower_out = (lower_out.shape[1] - num_zero_padded_element_by_batch).tolist()
        boundaries = [zero_shifted_lower_indices[i, higher_boundaries[i][:-1]].tolist() + [len_lower_out[i]] for i in range(len(lower_out))]


        # higher_boundaries = [0] + (torch.where(higher_indices[1:] - higher_indices[:-1] == 1)[0] + 1).cpu().tolist() + [len(higher_indices)]
        # boundaries = [int(lower_indices[x]-lower_indices[0]) for x in higher_boundaries[:-1]] + [lower_out.shape[-2]]
    
    softmax_similarity = torch.nn.utils.rnn.pad_sequence(
      [torch.cat(get_softmax_by_boundary(similarity[batch_idx], boundaries[batch_idx]))
        for batch_idx in range(len(lower_out))], 
      batch_first=True
    )
    # softmax_similarity = torch.cat([torch.softmax(similarity[:,boundaries[i-1]:boundaries[i],:], dim=1) for i in range(1, len(boundaries))], dim=1)
    if hasattr(attention_weights, 'head_size'):
        x_split = torch.stack(lower_out.split(split_size=attention_weights.head_size, dim=2), dim=2)
        weighted_x = x_split * softmax_similarity.unsqueeze(-1).repeat(1,1,1, x_split.shape[-1])
        weighted_x = weighted_x.view(x_split.shape[0], x_split.shape[1], lower_out.shape[-1])
        higher_nodes = torch.nn.utils.rnn.pad_sequence([
          torch.cat([torch.sum(weighted_x[i:i+1,boundaries[i][j-1]:boundaries[i][j],: ], dim=1) for j in range(1, len(boundaries[i]))], dim=0) \
          for i in range(len(lower_out))
        ], batch_first=True
        )
    else:
        weighted_sum = softmax_similarity * lower_out
        higher_nodes = torch.cat([torch.sum(weighted_sum[:,boundaries[i-1]:boundaries[i],:], dim=1) 
                                for i in range(1, len(boundaries))]).unsqueeze(0)
    return higher_nodes


def span_beat_to_note_num(beat_out, beat_number):
  '''
  beat_out (torch.Tensor): N x T_beat x C
  beat_number (torch.Tensor): N x T_note x C
  '''
  zero_shifted_beat_number = beat_number - beat_number[:,0:1]
  len_note = cal_length_from_padded_beat_numbers(beat_number)

  batch_indices = torch.cat([torch.ones(length)*i for i, length in enumerate(len_note)]).long()
  note_indices = torch.cat([torch.arange(length) for length in len_note])
  beat_indices = torch.cat([zero_shifted_beat_number[i,:length] for i, length in enumerate(len_note)]).long()

  span_mat = torch.zeros(beat_number.shape[0], beat_number.shape[1], beat_out.shape[1]).to(beat_out.device)
  span_mat[batch_indices, note_indices, beat_indices] = 1
  spanned_beat = torch.bmm(span_mat, beat_out)
  return spanned_beat

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def sum_with_attention(hidden, attention_net):
    attention = attention_net(hidden)
    attention = torch.nn.functional.softmax(attention, dim=0)
    upper_node = hidden * attention
    upper_node_sum = torch.sum(upper_node, dim=0)

    return upper_node_sum

def combine_splitted_graph_output(temp_output, orig_input, margin=100):
    '''
    Input
        temp_output: Temporary output of GGNN  [ number of edge batch X notes per edge X vector dimension]
        edges: Batch of splitted graph [ (Number of Edge type X number of edge batch) X notes per edge X vector dimension]
        orig_input: original input before GGNN update [ 1 x num notes x C]
    Output:
        updated_input
    '''
    updated_input = torch.zeros_like(orig_input)
    temp_output = temp_output.view(orig_input.shape[0], -1, temp_output.shape[1], temp_output.shape[2])
    updated_input[:, 0:temp_output.shape[2] - margin,:] = temp_output[:, 0, :-margin, :]
    cur_idx = temp_output.shape[2] - margin
    end_idx = cur_idx + temp_output.shape[2] - margin * 2
    for i in range(1, temp_output.shape[1]-1):
        updated_input[:, cur_idx:end_idx, :] = temp_output[:, i, margin:-margin, :]
        cur_idx = end_idx
        end_idx = cur_idx + temp_output.shape[2] - margin * 2
    updated_input[:, cur_idx:, :] = temp_output[:, -1, -(orig_input.shape[1]-cur_idx):, :]
    return updated_input


def combine_splitted_graph_output_with_several_edges(temp_output, orig_input, num_edge_type, margin=100):
    '''
    Input:
        temp_output: Temporary output of GGNN  [ N (num_batch) x S(num_slice) x E (num_edge_type) X L (num_notes_per_slice) X C]
        edges: Batch of splitted graph [ (Number of Edge type X number of edge batch) X notes per edge X vector dimension]
        orig_input: original input before GGNN update [ N x T(num notes) x C]
        margin (int): HAS TO BE SAME WITH GRAPH SPLIT SETTING
    Output:
        combined_output (torch.Tensor): N x E x T x C
    '''
    n_batch = temp_output.shape[0]
    n_edge = temp_output.shape[2]
    l_slice = temp_output.shape[3]
    n_notes = orig_input.shape[1]
    n_features = orig_input.shape[-1]
    batch_is_padded = temp_output.sum([2,3,4]) == 0
    valid_slice_length = temp_output.shape[1] - batch_is_padded.sum(1)
    valid_note_length = orig_input.shape[1] - (orig_input.sum(-1) == 0).sum(1)
    valid_note_length.clamp_min_(l_slice) # TODO: ad-hoc solution to prevent error
    combined_output = torch.zeros(n_batch, n_edge, n_notes, n_features).to(orig_input.device)

    # Handle 0th slice
    combined_output[:, :, 0:l_slice-margin] = temp_output[:, 0,:, :-margin]
    cur_idx = l_slice - margin
    end_idx = cur_idx + l_slice - margin * 2
    for i in range(1, temp_output.shape[1]-1):
      combined_output[~batch_is_padded[:,i], :, cur_idx:end_idx, :] = temp_output[~batch_is_padded[:,i], i, :, margin:-margin, :]
      cur_idx = end_idx
      end_idx = cur_idx + l_slice - margin * 2
    for b_id in range(n_batch):
      combined_output[b_id, : , valid_note_length[b_id] -(l_slice-margin):valid_note_length[b_id] ] = temp_output[b_id, valid_slice_length[b_id]-1, :, margin:]
    return combined_output
    # updated_input = torch.zeros_like(orig_input.repeat(num_edge_type, 1, 1))
    # temp_output = temp_output.view(updated_input.shape[0], -1, temp_output.shape[1], temp_output.shape[2])
    # updated_input[:, 0:temp_output.shape[2] - margin,:] = temp_output[:, 0, :-margin, :]
    # cur_idx = temp_output.shape[2] - margin
    # end_idx = cur_idx + temp_output.shape[2] - margin * 2
    # for i in range(1, temp_output.shape[1]-1):
    #     updated_input[:, cur_idx:end_idx, :] = temp_output[:, i, margin:-margin, :]
    #     cur_idx = end_idx
    #     end_idx = cur_idx + temp_output.shape[2] - margin * 2
    # updated_input[:, cur_idx:, :] = temp_output[:, -1, -(orig_input.shape[1]-cur_idx):, :]
    # return updated_input

def split_note_input_to_graph_batch(orig_input, batch_edges, overlap=200):
  '''
  orig_input (torch.Tensor): N x T x C
  batch_edges (torch.Tensor): N x S(NumSlice) x EdgeType x L(LenSlice) x L
  '''
  num_graph_slice = batch_edges.shape[1]
  if num_graph_slice == 1:
    return orig_input.unsqueeze(1)
  num_notes_per_slice = batch_edges.shape[-1]
  input_split = torch.zeros((orig_input.shape[0], 
                            num_graph_slice, 
                            num_notes_per_slice, 
                            orig_input.shape[2])).to(orig_input.device) # N x S X L x C
  batch_is_padded = batch_edges.sum([2,3,4]) == 0
  valid_batch_length = batch_edges.shape[1] - batch_is_padded.sum(1)

  for i in range(num_graph_slice-1):
    input_split[:, i] = orig_input[:, overlap*i:overlap*i+num_notes_per_slice, :]
  last_slice = get_last_k_notes_from_padded_batch(orig_input, num_notes_per_slice)
  input_split[torch.arange(input_split.shape[0]), valid_batch_length-1] = last_slice
  input_split[batch_is_padded] = 0 # mask out the padded one
  return input_split
  # input_split = torch.zeros((num_batch, num_notes_per_batch, orig_input.shape[2])).to(orig_input.device)
  # for i in range(num_batch-1):
  #     input_split[i] = orig_input[0, overlap*i:overlap*i+num_notes_per_batch, :]
  # input_split[-1] = orig_input[0,-num_notes_per_batch:, :]
  # return input_split

def get_last_k_notes_from_padded_batch(batch_note, k):
  '''
  batch_note(torch.Tensor): zero-padded N x T x C
  '''
  len_batch = batch_note.shape[1] - (batch_note.sum(-1)==0).sum(1)
  len_batch.clamp_min_(k) # TODO: temporal code to prevent error
  output = torch.stack(
    [batch_note[i,len_batch[i]-k:len_batch[i]] for i in range(len(batch_note))]
  , dim=0)
  return output


def masking_half(y):
    num_notes = y.shape[1]
    y = y[:,:num_notes//2,:]
    return y

def mask_batched_graph(edges, num_edge_type, end_idx, margin=100):
    edges_reshaped = edges.view(-1, num_edge_type, edges.shape[1], edges.shape[2])
    end_batch_idx = math.ceil((end_idx - (edges.shape[1] - margin)) / (edges.shape[1] - margin * 2))

    edges_reshaped

    return

def encode_with_net(score_input, mean_net, var_net):
    mu = mean_net(score_input)
    var = var_net(score_input)

    z = reparameterize(mu, var)
    return z, mu, var

def get_beat_corresp_out(note_out, beat_numbers, batch_ids, current_note_idx):
  '''
  note_out (torch.Tensor): N x T x C. Note-level output
  beat_numbers (torch.LongTensor): N x T. Beat (or measure) ids for each note
  batch_ids (torch.LongTensor): n. 
  current_note_idx (int): Currently decoding note index among T

  out (torch.Tensor): zero-padded output of corresponding notes from the previous beat 
  '''

  # find note indices of previous beat
  note_indices = find_note_indices_of_given_beat(beat_numbers, batch_ids, current_note_idx)
  
  out = torch.nn.utils.rnn.pad_sequence(
    [note_out[idx, note_indices[i]] for i, idx in enumerate(batch_ids)]
  , True)

  return out 

def find_note_indices_of_given_beat(beat_numbers, batch_ids, current_note_idx):
  '''
  beat_numbers (torch.LongTensor): N x T. Beat (or measure) ids for each note
  current_note_idx (int): Currently decoding note index among T

  out: List of tensor indices
  '''
  
  # select by batch_ids
  selected_beat_numbers = beat_numbers[batch_ids]

  # get current beat ids
  current_beat = selected_beat_numbers[:, current_note_idx]

  # get prev_beat
  prev_beat = current_beat - 1

  # find indices of note indices that belong to prev_beat
  note_indices = [torch.where(selected_beat_numbers[i,:current_note_idx]==prev_beat[i])[0] for i in range(len(batch_ids))]
  return note_indices


def run_hierarchy_lstm_with_pack(sequence, lstm):
  '''
  sequence (torch.Tensor): zero-padded sequece of N x T x C
  lstm (torch.LSTM): LSTM layer
  '''
  batch_note_length = sequence.shape[1] - (sequence.sum(-1)==0).sum(dim=1)
  packed_sequence = pack_padded_sequence(sequence, batch_note_length.cpu(), True, False )
  hidden_out, _ = lstm(packed_sequence)
  hidden_out, _ = pad_packed_sequence(hidden_out, True)

  return hidden_out