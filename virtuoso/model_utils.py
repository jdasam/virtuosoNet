import torch


def sum_with_boundary(x_split, attention_split, num_head):
    weighted_mul = torch.bmm(attention_split.transpose(1,2), x_split)
    restore_size = int(weighted_mul.size(0) / num_head)
    attention = torch.cat(weighted_mul.split(split_size=restore_size, dim=0), dim=2)
    sum_attention = torch.sum(attention, dim=1)
    return sum_attention

def make_higher_node(lower_out, attention_weights, lower_indices, higher_indices, lower_is_note=False):
    higher_nodes = []

    similarity = attention_weights.get_attention(lower_out)
    if lower_is_note:
        boundaries = [0] + (torch.where(higher_indices[1:] - higher_indices[:-1] == 1)[0] + 1).cpu().tolist() + [len(higher_indices)]
        
    else:
        higher_boundaries = [0] + (torch.where(higher_indices[1:] - higher_indices[:-1] == 1)[0] + 1).cpu().tolist() + [len(higher_indices)]
        boundaries = [int(lower_indices[x]-lower_indices[0]) for x in higher_boundaries[:-1]] + [lower_out.shape[-2]]
    softmax_similarity = torch.cat([torch.softmax(similarity[:,boundaries[i-1]:boundaries[i],:].permute(0,1,2), dim=1) for i in range(1, len(boundaries))], dim=1)
    if hasattr(attention_weights, 'head_size'):
        x_split = torch.cat(lower_out.split(split_size=attention_weights.head_size, dim=2), dim=0)
        # higher_nodes = torch.cat([sum_with_boundary(x_split[:,boundaries[i-1]:boundaries[i],:], 
        #                     softmax_similarity[:,boundaries[i-1]:boundaries[i],:], attention_weights.num_head)
        #                     for i in range(1, len(boundaries))]).unsqueeze(0)
        higher_nodes = torch.stack([sum_with_boundary(x_split[:,boundaries[i-1]:boundaries[i],:], 
                            softmax_similarity[:,boundaries[i-1]:boundaries[i],:], attention_weights.num_head)
                            for i in range(1, len(boundaries))]).permute(1,0,2)
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
    updated_input[0, 0:temp_output.shape[1] - margin,:] = temp_output[0, :-margin, :]
    cur_idx = temp_output.shape[1] - margin
    end_idx = cur_idx + temp_output.shape[1] - margin * 2
    for i in range(1, temp_output.shape[0]-1):
        updated_input[0, cur_idx:end_idx, :] = temp_output[i, margin:-margin, :]
        cur_idx = end_idx
        end_idx = cur_idx + temp_output.shape[1] - margin * 2
    updated_input[0, cur_idx:, :] = temp_output[-1, -(orig_input.shape[1]-cur_idx):, :]
    return updated_input

def split_note_input_to_graph_batch(orig_input, num_batch, num_notes_per_batch, overlap=200):
    input_split = torch.zeros((num_batch, num_notes_per_batch, orig_input.shape[2])).to(orig_input.device)
    for i in range(num_batch-1):
        input_split[i] = orig_input[0, overlap*i:overlap*i+num_notes_per_batch, :]
    input_split[-1] = orig_input[0,-num_notes_per_batch:, :]
    return input_split

def masking_half(y):
    num_notes = y.shape[1]
    y = y[:,:num_notes//2,:]
    return y

def encode_with_net(score_input, mean_net, var_net):
    mu = mean_net(score_input)
    var = var_net(score_input)

    z = reparameterize(mu, var)
    return z, mu, var
