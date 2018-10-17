"""
Load and process datasets used by graphsage (http://snap.stanford.edu/graphsage/)

These are datasets with node labels and attributes

code adapted from https://github.com/williamleif/GraphSAGE/blob/master/graphsage/utils.py
"""

import numpy as np
import random
import json
import sys
import os
import sklearn

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"


def graphsage_load_data(prefix, normalize=True):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None

    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)

    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    # broken_count = 0
    # for node in G.nodes():
    #     if not 'val' in G.node[node] or not 'test' in G.node[node]:
    #         G.remove_node(node)
    #         broken_count += 1
    # print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    # print("Loaded data.. now preprocessing..")
    # for edge in G.edges():
    #     if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
    #             G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
    #         G[edge[0]][edge[1]]['train_removed'] = True
    #     else:
    #         G[edge[0]][edge[1]]['train_removed'] = False

    # if normalize and not feats is None:
    #     from sklearn.preprocessing import StandardScaler
    #     train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
    #     train_feats = feats[train_ids]
    #     scaler = StandardScaler()
    #     scaler.fit(train_feats)
    #     feats = scaler.transform(feats)

    data = {'graph': G, 'feats': feats, 'id_map': id_map, 'class_map': class_map}
    return data


def _process_graphsage_data(graphsage_data):
    """
    Convert data loaded by default graphsage loader to a format expected by an ERM algorithm
    (Note this is just convenience... one could directly write a tensorflow dataset to use the graphsage data as-is)
    """

    # convert data everything is in arrays where the array index corresponds to vertex label
    # this works under the presumption of a single connected component with contiguous vertex labels
    graph = graphsage_data['graph']
    features = graphsage_data['feats']
    class_map = graphsage_data['class_map']
    id_map = graphsage_data['id_map']

    # graph data

    # restrict to single cc. Not logically necessary, but samplers may rely on this
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc)

    node_ids = graph.nodes()
    graph = nx.convert_node_labels_to_integers(graph)
    id_map2 = dict(zip(node_ids, graph.nodes()))

    edge_list = np.array(graph.edges())

    # feature data
    keep_feats = np.zeros([graph.number_of_nodes(), features.shape[1]], np.float32)
    for k,v in id_map2.items():
        # TODO: not actually 100% sure how id_map was coded
        original_node_index = id_map[k]
        keep_feats[v] = features[original_node_index]

    # classes
    # TODO: assuming class is given as an integer... doesn't work for protein
    # num_classes = len(class_map[labelled_nodes[0]])
    classes = np.zeros(graph.number_of_nodes(), dtype=np.int32)
    for k,v in id_map2.items():
        classes[v] = class_map[k]

    node_ids = np.array(node_ids)
    return {'edge_list': edge_list, 'features': features, 'classes': classes, 'node_ids': node_ids}


def _preprocess_packed_adjacency_list(data):
    # precompute the stuff required for the packed adjacency list representation
    # saves some compute / peak memory when loading this dataset later

    from ..graph_ops.representations import create_packed_adjacency_from_redundant_edge_list

    # Load the current edge list, and go to canonical form
    # without self edges
    edge_list = data['edge_list']
    edge_list = edge_list[edge_list[:, 0] != edge_list[:, 1], :]
    edge_list.sort(axis=-1)
    edge_list = np.unique(edge_list, axis=0)

    # Compute redundant edge list
    red_edge_list = np.concatenate((edge_list, np.flip(edge_list, axis=1)))
    packed_adj_list = create_packed_adjacency_from_redundant_edge_list(red_edge_list )

    return {
        **data,
        'neighbours': packed_adj_list.neighbours,
        'lengths': packed_adj_list.lengths,
    }



def main():
    graphsage_data = graphsage_load_data("../data/reddit/reddit")
    # graphsage_data = graphsage_load_data("../data/ppi/ppi")
    data = _process_graphsage_data(graphsage_data)
    data = _preprocess_packed_adjacency_list(data)
    np.savez_compressed('reddit.npz', **data)

if __name__ == '__main__':
    main()