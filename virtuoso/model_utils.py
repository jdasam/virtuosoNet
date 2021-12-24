from numpy import diff
import torch
import math

def sum_with_boundary(x_split, attention_split, num_head):
    weighted_mul = torch.bmm(attention_split.transpose(1,2), x_split)
    restore_size = int(weighted_mul.size(0) / num_head)
    attention = torch.cat(weighted_mul.split(split_size=restore_size, dim=0), dim=2)
    sum_attention = torch.sum(attention, dim=1)
    return sum_attention

def get_softmax_by_boundary(similarity, boundaries):
  '''
  similarity = similarity of a single sequence of data (T x C)
  boundaries = list of a boundary index
  '''
  return  [torch.softmax(similarity[boundaries[i-1]:boundaries[i],: ], dim=0)  \
              for i in range(1, len(boundaries))
                if boundaries[i-1] < boundaries[i] # sometimes, boundaries can start like [0, 0, ...]
          ]

def find_boundaries(diff_boundary, higher_indices, i):
  out = [0] + (diff_boundary[diff_boundary[:,0]==i][:,1]+1 ).tolist() + [torch.max(torch.nonzero(higher_indices[i])).item()+1]
  if out[1] == 0:
    out.pop(0)
  return out

def make_higher_node(lower_out, attention_weights, lower_indices, higher_indices, lower_is_note=False):
    # higher_nodes = []

    similarity = attention_weights.get_attention(lower_out)
    diff_boundary = torch.nonzero(higher_indices[:,1:] - higher_indices[:,:-1] == 1).cpu()
    if lower_is_note:
        boundaries = [find_boundaries(diff_boundary, higher_indices, i) for i in range(len(lower_out))]
        # boundaries = [0] + (torch.nonzero(higher_indices[:,1:] - higher_indices[:,:-1] == 1) + 1).cpu().tolist() + [len(higher_indices[0])]
        # boundaries = [0] + (torch.where(higher_indices[0,1:] - higher_indices[0,:-1] == 1)[0] + 1).cpu().tolist() + [len(higher_indices[0])]
    else:
        higher_boundaries = [find_boundaries(diff_boundary, higher_indices, i) for i in range(len(lower_out))]
        zero_shifted_lower_indices = lower_indices - lower_indices[:,0:1]
        boundaries = [zero_shifted_lower_indices[i, higher_boundaries[i]-1] for i in range(len(lower_out))]


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
        weighted_x = x_split * softmax_similarity.unsqueeze(-1).repeat(1,1,1,64)
        weighted_x = x_split.view(x_split.shape[0], x_split.shape[1], lower_out.shape[-1])
        higher_nodes = torch.nn.utils.rnn.pad_sequence([
          torch.cat([torch.sum(weighted_x[i:i+1,boundaries[i][j-1]:boundaries[i][j],: ], dim=1) for j in range(1, len(boundaries[i]))], dim=0) \
          for i in range(len(lower_out))
        ], batch_first=True
        )
        # higher_nodes = torch.cat([sum_with_boundary(x_split[:,boundaries[i-1]:boundaries[i],:], 
        #                     softmax_similarity[:,boundaries[i-1]:boundaries[i],:], attention_weights.num_head)
        #                     for i in range(1, len(boundaries))]).unsqueeze(0)
        # higher_nodes = torch.stack([sum_with_boundary(x_split[:,boundaries[i-1]:boundaries[i],:], 
        #                     softmax_similarity[:,boundaries[i-1]:boundaries[i],:], attention_weights.num_head)
        #                     for i in range(1, len(boundaries))]).permute(1,0,2)
    else:
        weighted_sum = softmax_similarity * lower_out
        higher_nodes = torch.cat([torch.sum(weighted_sum[:,boundaries[i-1]:boundaries[i],:], dim=1) 
                                for i in range(1, len(boundaries))]).unsqueeze(0)
    # for low_index in range(num_lower_nodes):
    #     if lower_is_note:
    #         current_note_index = low_index
    #     else:
    #         absolute_low_index = start_lower_index + low_index
    #         current_note_index = lower_indices.index(absolute_low_index)
    #     if higher_indices[current_note_index] > prev_higher_index:
    #         # new beat start
    #         lower_node_end = low_index
    #         corresp_lower_out = lower_out[:, lower_node_start:lower_node_end, :]
    #         higher = attention_weights(corresp_lower_out)
    #         higher_nodes.append(higher)

    #         lower_node_start = low_index
    #         prev_higher_index = higher_indices[current_note_index]

    # corresp_lower_out = lower_out[:, lower_node_start:, :]
    # higher = attention_weights(corresp_lower_out)
    # higher_nodes.append(higher)

    # higher_nodes = torch.stack(higher_nodes).view(1, -1, lower_hidden_size)
    return higher_nodes

def span_beat_to_note_num(beat_out, beat_number):
    start_beat = beat_number[0]
    num_beat = beat_out.shape[1]
    num_notes = beat_number.shape[0]
    span_mat = torch.zeros(beat_out.shape[0], num_notes, num_beat)
    beat_indices = torch.Tensor(list(enumerate(beat_number - start_beat))).to(torch.long)
    span_mat[:, beat_indices[:,0], beat_indices[:,1]] = 1
    # for i in range(num_notes):
    #     beat_index = beat_number[i] - start_beat
    #     if beat_index >= num_beat:
    #         beat_index = num_beat-1
    #     span_mat[0,i,beat_index] = 1
    span_mat = span_mat.to(beat_out.device)

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
