import numpy as np
import torch as th



def edges_to_matrix(edges, num_notes):
    if not MODEL.is_graph:
        return None
    num_keywords = len(GRAPH_KEYS)
    matrix = np.zeros((N_EDGE_TYPE, num_notes, num_notes))

    for edg in edges:
        if edg[2] not in GRAPH_KEYS:
            continue
        edge_type = GRAPH_KEYS.index(edg[2])
        matrix[edge_type, edg[0], edg[1]] = 1
        if edge_type != 0:
            matrix[edge_type+num_keywords, edg[1], edg[0]] = 1
        else:
            matrix[edge_type, edg[1], edg[0]] = 1

    matrix[num_keywords, :, :] = np.identity(num_notes)
    matrix = th.Tensor(matrix)
    return matrix


def edges_to_matrix_short(edges, slice_index):
    if not MODEL.is_graph:
        return None
    num_keywords = len(GRAPH_KEYS)
    num_notes = slice_index[1] - slice_index[0]
    matrix = np.zeros((N_EDGE_TYPE, num_notes, num_notes))
    start_edge_index = xml_matching.binary_index_for_edge(
        edges, slice_index[0])
    end_edge_index = xml_matching.binary_index_for_edge(
        edges, slice_index[1] + 1)
    for i in range(start_edge_index, end_edge_index):
        edg = edges[i]
        if edg[2] not in GRAPH_KEYS:
            continue
        if edg[1] >= slice_index[1]:
            continue
        edge_type = GRAPH_KEYS.index(edg[2])
        matrix[edge_type, edg[0]-slice_index[0], edg[1]-slice_index[0]] = 1
        if edge_type != 0:
            matrix[edge_type+num_keywords, edg[1] -
                   slice_index[0], edg[0]-slice_index[0]] = 1
        else:
            matrix[edge_type, edg[1]-slice_index[0], edg[0]-slice_index[0]] = 1
    matrix[num_keywords, :, :] = np.identity(num_notes)
    matrix = th.Tensor(matrix)

    return matrix


def edges_to_sparse_tensor(edges):
    num_keywords = len(GRAPH_KEYS)
    edge_list = []
    edge_type_list = []

    for edg in edges:
        edge_type = GRAPH_KEYS.index(edg[2])
        edge_list.append(edg[0:2])
        edge_list.append([edg[1], edg[0]])
        edge_type_list.append(edge_type)
        if edge_type != 0:
            edge_type_list.append(edge_type+num_keywords)
        else:
            edge_type_list.append(edge_type)

        edge_list = th.LongTensor(edge_list)
    edge_type_list = th.FloatTensor(edge_type_list)

    matrix = th.sparse.FloatTensor(edge_list.t(), edge_type_list)

    return matrix
