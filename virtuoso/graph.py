import numpy as np
import torch as th

GRAPH_KEYS=[]
N_EDGE_TYPE=0


def edges_to_matrix(edges, num_notes, graph_keys):
    if len(graph_keys)==0:
        return None
    num_keywords = len(graph_keys)
    graph_dict = {key: i for i, key in enumerate(graph_keys) }
    if 'rest_as_forward' in graph_dict:
        graph_dict['rest_as_forward'] = graph_dict['forward']
        num_keywords -= 1
    matrix = np.zeros((num_keywords * 2, num_notes, num_notes))
    edg_indices = [(graph_dict[edg[2]], edg[0], edg[1])  
                for edg in edges
                if edg[2] in graph_dict]
    reverse_indices = [ (edg[0]+num_keywords, edg[2], edg[1]) if edg[0] != 0 else 
        (edg[0], edg[2], edg[1]) for edg in edg_indices ]
    edg_indices = np.asarray(edg_indices + reverse_indices)
    matrix[edg_indices[:,0], edg_indices[:,1], edg_indices[:,2]] = 1
    # for edg in edges:
    #     if edg[2] not in graph_keys:
    #         continue
    #     edge_type = graph_keys.index(edg[2])
    #     matrix[edge_type, edg[0], edg[1]] = 1
    #     if edge_type != 0:
    #         matrix[edge_type+num_keywords, edg[1], edg[0]] = 1
    #     else:
    #         matrix[edge_type, edg[1], edg[0]] = 1

    matrix[num_keywords, :, :] = np.identity(num_notes)
    matrix = th.Tensor(matrix)
    return matrix


def edges_to_matrix_short_old(edges, slice_index, graph_keys):
    num_keywords = len(graph_keys)
    num_notes = slice_index[1] - slice_index[0]
    matrix = np.zeros((num_keywords * 2, num_notes, num_notes))
    start_edge_index = binary_index_for_edge(edges, slice_index[0])
    end_edge_index = binary_index_for_edge(edges, slice_index[1] + 1)
    for i in range(start_edge_index, end_edge_index):
        edg = edges[i]
        if edg[2] not in graph_keys:
            continue
        if edg[1] >= slice_index[1]:
            continue
        edge_type = graph_keys.index(edg[2])
        matrix[edge_type, edg[0]-slice_index[0], edg[1]-slice_index[0]] = 1
        if edge_type != 0:
            matrix[edge_type+num_keywords, edg[1] -
                   slice_index[0], edg[0]-slice_index[0]] = 1
        else:
            matrix[edge_type, edg[1]-slice_index[0], edg[0]-slice_index[0]] = 1
    matrix[num_keywords, :, :] = np.identity(num_notes)
    matrix = th.Tensor(matrix)

    return matrix

def edges_to_matrix_short(edges, slice_index, graph_keys):
    if len(graph_keys)==0:
        return None
    num_keywords = len(graph_keys)
    graph_dict = {key: i for i, key in enumerate(graph_keys) }
    if 'rest_as_forward' in graph_dict:
        graph_dict['rest_as_forward'] = graph_dict['forward']
        num_keywords -= 1
    num_notes = slice_index[1] - slice_index[0]
    matrix = np.zeros((num_keywords * 2, num_notes, num_notes))
    start_edge_index = binary_index_for_edge(edges, slice_index[0])
    end_edge_index = binary_index_for_edge(edges, slice_index[1] + 1)


    edg_indices = [(graph_dict[edg[2]], edg[0]-slice_index[0], edg[1]-slice_index[0])  
                    for edg in edges[start_edge_index:end_edge_index]
                    if edg[2] in graph_dict and edg[1] < slice_index[1] ]

    reverse_indices = [ (edg[0]+num_keywords, edg[2], edg[1]) if edg[0] != 0 else 
        (edg[0], edg[2], edg[1]) for edg in edg_indices ]
    edg_indices = np.asarray(edg_indices + reverse_indices)
    matrix[edg_indices[:,0], edg_indices[:,1], edg_indices[:,2]] = 1
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

def binary_index_for_edge(alist, item):
    first = 0
    last = len(alist) - 1
    midpoint = 0

    if (item < alist[first][0]):
        return 0

    while first < last:
        midpoint = (first + last) // 2
        currentElement = alist[midpoint][0]

        if currentElement < item:
            if alist[midpoint + 1][0] > item:
                return midpoint
            else:
                first = midpoint + 1
            if first == last and alist[last][0] > item:
                return midpoint
        elif currentElement > item:
            last = midpoint - 1
        else:
            if midpoint + 1 == len(alist):
                return midpoint
            while midpoint >= 1 and alist[midpoint - 1][0] == item:
                midpoint -= 1
                if midpoint == 0:
                    return midpoint
            return midpoint
    return last