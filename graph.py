import numpy as np
import sys

class Graph(object):

    def __init__(self, MDP=None):
        self.graph = MDP
        self.V=self.vertices()
        self.succ=self.successors()
        self.E=self.edges()
        self.Time=0

    def vertices(self):
    # returns the vertices of a graph
        return list(self.graph[0].keys())

    def successors(self):
    # returns a dictionary object in which the keys are the vertices and the
    # values are the set of neighboring vertices
        succ = {i : set() for i in self.V }
        for pair in self.graph[1]:
            succ[pair[0]].update(self.graph[1][pair][1])
        return succ

    def edges(self):
    # returns a list object in which the elements are the edge pairs (u,v)
        edges = []
        for vertex in self.succ.keys():
            for succ in self.succ[vertex]:
                edges.append((vertex,succ))
        return edges


    def Floyd_Warshall(self):
    # an implementation of the Floyd_Warshall algorithm.
    # the algorithm computes the minimum distances between any two vertices
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


    def verify_path(self, init, target):
    # an implementation of the breadth-first-search algorithm
    # to verify whether there is a path between the pair (init,target)

        visited = [] # List to keep track of visited nodes.
        queue = []     #Initialize a queue
        visited.append(init)
        queue.append(init)

        while queue:
            s = queue.pop(0)

            for neighbour in self.successors()[s]:
                if neighbour not in visited:
                    visited.append(neighbour)
                    queue.append(neighbour)
                if target in visited:
                    return True
        return False

    def minDistance(self, dist, sptSet):

            # Initialize minimum distance for next node
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
