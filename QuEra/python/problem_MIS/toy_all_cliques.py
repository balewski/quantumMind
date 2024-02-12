#!/usr/bin/env python3

__author__ = "ChatGPT"
__email__ = "janstar1122@gmail.com"

'''
The Bron-Kerbosch algorithm is primarily used to find cliques in an undirected graph, rather than maximal independent sets. 

In its original form, the Bron-Kerbosch algorithm is used to find all cliques (complete subgraphs) in an undirected graph. A clique is a subset of vertices where every pair of vertices is connected by an edge. The algorithm explores all possible cliques in the graph.

Every maximal independent set (MIS) corresponds to a complement of a clique in the complement graph. 

'''


import networkx as nx

def bron_kerbosch(graph, clique=None, candidates=None, excluded=None):
    if candidates is None:
        candidates = set(graph.nodes)
    if excluded is None:
        excluded = set()
    if clique is None:
        clique = set()
    
    if not candidates and not excluded:
        yield clique
        return

    for node in list(candidates):
        new_candidates = candidates.intersection(graph.neighbors(node))
        new_excluded = excluded.intersection(graph.neighbors(node))
        yield from bron_kerbosch(graph, clique | {node}, new_candidates, new_excluded)
        candidates.remove(node)
        excluded.add(node)

# Create a sample graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)])

# Find all cliques in a sets
all_cliques= list(bron_kerbosch(G))
print("all_cliques sets:", all_cliques)
