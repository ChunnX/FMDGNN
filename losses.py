import torch
import torch.nn.functional as F
import networkx as nx
import time

from utils import *
from config import *


def reconstruction_loss(pred, target):
    """Compute the reconstruction loss"""
    return F.l1_loss(pred, target)


def kl_divergence(pred, target):
    """Compute the KL-divergence-based loss"""
    kl_loss = torch.abs(F.kl_div(F.softmax(target, dim=0), F.softmax(pred, dim=0), None, None, 'sum'))
    kl_loss = (1/350) * kl_loss
    return kl_loss


def topological_measures(feature):
    """Compute the topological measures (BC, EC, PC)"""
    # ROI is the number of brain regions (i.e.,35 in our case)
    ROI = N_ROI # from config file
    BC = np.empty((0, ROI), int)
    EC = np.empty((0, ROI), int)
    PC = np.empty((0, ROI), int)

    topology = []
    start_time = time.time()
    for i in range(feature.shape[0]):
        adjacency_matrix = antiVectorize(feature[i]).detach().cpu().numpy()

        # create a graph from similarity matrix
        graph = nx.from_numpy_matrix(adjacency_matrix)

        # betweeness centrality
        bc = nx.betweenness_centrality(graph, weight="weight")
        betweenness_centrality = np.array([bc[g] for g in graph])
        # eigenvector centrality
        ec = nx.eigenvector_centrality(graph, max_iter=1000, weight="weight")
        eigenvector_centrality = np.array([ec[g] for g in graph])
        # pagerank centrality
        pc = nx.pagerank(graph, max_iter=1000, weight="weight")
        pagerank_centrality = np.array([pc[g] for g in graph])

        # create a matrix of all subjects centralities
        BC = np.vstack((BC, betweenness_centrality))
        EC = np.vstack((EC, eigenvector_centrality))
        PC = np.vstack((PC, pagerank_centrality))

    topology.append(BC)  # 1
    topology.append(EC)  # 2
    topology.append(PC)  # 3

    end_time = time.time()
    # print(start_time - end_time)
    return topology


