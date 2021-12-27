from numpy import diff
import torch
import math
from .utils import find_boundaries_batch, get_softmax_by_boundary, cal_length_from_padded_beat_numbers

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
        len_lower_out = (lower_out.shape[1] - (lower_out.sum(-1)==0).sum(1)).tolist()
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
    Input:
        temp_output: Temporary output of GGNN  [ number of edge batch X notes per edge X vector dimension]
        edges: Batch of splitted graph [ (Number of Edge type X number of edge batch) X notes per edge X vector dimension]
        orig_input: original input before GGNN update [ 1 x num notes x vector dimension]
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
        temp_output: Temporary output of GGNN  [ (number of edge type x number of edge batch)  X notes per edge X vector dimension]
        edges: Batch of splitted graph [ (Number of Edge type X number of edge batch) X notes per edge X vector dimension]
        orig_input: original input before GGNN update [ 1 x num notes x vector dimension]
    Output:
        updated_input
    '''
    updated_input = torch.zeros_like(orig_input.repeat(num_edge_type, 1, 1))
    temp_output = temp_output.view(updated_input.shape[0], -1, temp_output.shape[1], temp_output.shape[2])
    updated_input[:, 0:temp_output.shape[2] - margin,:] = temp_output[:, 0, :-margin, :]
    cur_idx = temp_output.shape[2] - margin
    end_idx = cur_idx + temp_output.shape[2] - margin * 2
    for i in range(1, temp_output.shape[1]-1):
        updated_input[:, cur_idx:end_idx, :] = temp_output[:, i, margin:-margin, :]
        cur_idx = end_idx
        end_idx = cur_idx + temp_output.shape[2] - margin * 2
    updated_input[:, cur_idx:, :] = temp_output[:, -1, -(orig_input.shape[1]-cur_idx):, :]
    return updated_input

def split_note_input_to_graph_batch(orig_input, num_graph_batch, num_notes_per_batch, overlap=200):
    input_split = torch.zeros((orig_input.shape[0], num_graph_batch, num_notes_per_batch, orig_input.shape[2])).to(orig_input.device)
    for i in range(num_graph_batch-1):
        input_split[:, i] = orig_input[:, overlap*i:overlap*i+num_notes_per_batch, :]
    input_split[:, -1] = orig_input[:,-num_notes_per_batch:, :]
    return input_split.view(-1, num_notes_per_batch, orig_input.shape[-1])
    # input_split = torch.zeros((num_batch, num_notes_per_batch, orig_input.shape[2])).to(orig_input.device)
    # for i in range(num_batch-1):
    #     input_split[i] = orig_input[0, overlap*i:overlap*i+num_notes_per_batch, :]
    # input_split[-1] = orig_input[0,-num_notes_per_batch:, :]
    # return input_split

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