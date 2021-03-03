### This file  1) constructs the graph representation of a given MDP
# 2) Finds strongly connected components
# 3) Finds maximal end components
# 4) Given a target set, finds the states with zero reaching probability
# 5) Given a target set, finds the states whose reachability probability is non-zero
# 6) Checks if the graph includes non-proper policies
import numpy as np
import sys
class Graph(object):

    def __init__(self, MDP=None):
        print('try commit')
        self.graph = MDP
        self.V=self.vertices()
        self.succ=self.successors()
        self.E=self.edges()
        self.Time=0

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.graph[0].keys())

    def successors(self):
     # MDP.successors[i] gives the 'set' of successor states for the state i
        succ = {i : set() for i in self.V }
        for pair in self.graph[1]:
            succ[pair[0]].update(self.graph[1][pair][1])
        return succ

    def edges(self):
     # MDP.successors[i] gives the 'set' of successor states for the state i
        edges = []
        for vertex in self.succ.keys():
            for succ in self.succ[vertex]:
                edges.append((vertex,succ))
        return edges


    def Floyd_Warshall(self):
        dist = {}
        for vertex in self.V:
            for vertex_2 in self.V:
                dist[(vertex,vertex_2)] = float('inf')
        for edge in self.E:
            if edge[0] == edge[1]:
                dist[edge] = 0
            else:
                dist[edge] = 1
        for k in self.V:
            for i in self.V:
                if (i,k) in self.E:
                    for j in self.V:
                        if dist[(i,j)] > dist[(i,k)] + dist[(k,j)]:
                            dist[(i,j)] = dist[(i,k)] + dist[(k,j)]
        return dist

    def minDistance(self, dist, sptSet):

            # Initilaize minimum distance for next node
            min = sys.maxsize

            # Search not nearest vertex not in the
            # shortest path tree
            for v in self.V:
                if dist[v] < min and sptSet[v] == False:
                    min = dist[v]
                    min_index = v

            return min_index

    # Funtion that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def Dijkstra(self, src):
        dist={}
        sptSet = {}
        for vertex in self.V:
            dist[vertex] = sys.maxsize
            sptSet[vertex] = False

        dist[src] = 0

        for vertex in self.V:
        	print("stuffknskjlksjdflksjdf")

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)

            # Put the minimum distance vertex in the
            # shotest path tree
            sptSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shotest path tree
            for v in self.V:
                if (u,v) in self.E and \
                     sptSet[v] == False and \
                     dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1

        return dist
