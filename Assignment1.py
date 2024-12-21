from collections import defaultdict, deque
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain, combinations
from scipy.cluster.hierarchy import dendrogram, linkage

class Graph:
    def __init__(self, v_set = None):
        self.adj_list = defaultdict(list)
        self.edges = 0
        self.v_set = v_set
        self.edge_list = []

    def add_edge(self, u, v):

        if self.v_set is not None:
            u = self.v_set[u]
            v = self.v_set[v]

        self.adj_list[u] = self.adj_list.get(u, [])
        self.adj_list[v] = self.adj_list.get(v, [])

        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)
            self.edge_list.append(tuple(sorted([u,v])))
            self.edges += 1

    def get_adj_list(self):
        return self.adj_list

    def remove_edge(self, u, v):
        if v in self.adj_list[u]:
            self.adj_list[u].remove(v)
        if u in self.adj_list[v]:
            self.adj_list[v].remove(u)

    def dfs(self, node, visited, community, min_value):
        stack = [(node, min_value)]  # Stack to keep track of nodes to visit, along with current min_value

        while stack:
            current_node, current_min = stack.pop()
            if current_node not in visited:
                visited.add(current_node)
                community.add(current_node)
                min_value = min(min_value, current_min)  # Update min_value if necessary
                # print(min_value)

                # Push all neighbors onto the stack
                for neighbor in self.adj_list[current_node]:
                    if neighbor not in visited:
                        stack.append((neighbor, min(min_value, neighbor)))

        return min_value

    def identify_communities(self):
        visited = set()
        communities = []
        node_to_community = np.zeros(len(self.adj_list), dtype=int)
        for node in self.adj_list:
            if node not in visited:
                community = set()
                community_id = self.dfs(node, visited, community, node)
                # print(community_id, 'com_id')
                communities.append(community)
                for n in community:
                    node_to_community[n] = community_id

        return node_to_community

def bfs_sssp(graph, start, max_n):
    distances = {}
    path = []
    predecessors = {}
    sigma = {}

    # initialization
    for i in range(max_n):
        distances[i] = float('inf')
        predecessors[i] = []
        sigma[i] = 0

    distances[start] = 0
    sigma[start] = 1.0

    queue = deque([start])

    while queue:
        node = queue.popleft()
        path.append(node)

        for neighbor in graph[node]:

            if distances[neighbor] == float('inf'): #path discovery
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

            if (distances[neighbor] == distances[node] + 1): # path counting
                sigma[neighbor] += sigma[node]
                predecessors[neighbor].append(node)

    return path, sigma, predecessors

edges = {}

def accumulation(betweenness, path, predecessors, sigma):
    delta = {i : 0 for i in path}
    while path:
        w = path.pop()
        for v in predecessors[w]:
            c = (delta[w] + 1)*sigma[v] / sigma[w]
            key = tuple(sorted((v,w)))
            betweenness[key] = betweenness.get(key,0) + c
            delta[v] += c
    return betweenness


def calculate_edge_betweenness(graph_adj_list, max_n, edge_list):
    edge_betweeness = dict.fromkeys(edge_list, 0.0)
    for i in range(max_n):
        path, sigma, predecessors = bfs_sssp(graph_adj_list, i, max_n)
        accumulation(edge_betweeness, path, predecessors, sigma)
    return  edge_betweeness

def Girvan_Newman_one_level(graph, max_n):
    prev = np.array([0]*max_n)
    c = []
    n = graph.edges
    while n > 0:
        n = n-1
        b = calculate_edge_betweenness(graph.get_adj_list(), max_n, graph.edge_list)
        m_b = max(b, key=b.get)
        graph.remove_edge(m_b[0], m_b[1])
        comms = graph.identify_communities()
        # print(comms)
        if not np.array_equal(comms, prev):
            c.append(comms)
            prev = comms
            break
    c = np.array(c).T
    return c

def Girvan_Newman(graph, max_n):
    prev = np.array([0]*max_n)
    c = []
    n = graph.edges
    level = 0
    edges_removed = 0
    while n > 0:
        n = n-1
        edges_removed += 1
        print(edges_removed)
        b = calculate_edge_betweenness(graph.get_adj_list(), max_n, graph.edge_list)
        m_b = max(b, key=b.get)
        graph.remove_edge(m_b[0], m_b[1])
        comms = graph.identify_communities()
        if not np.array_equal(comms, prev):
            level += 1
            print("Level =", level)
            c.append(comms)
            prev = comms
        if ( level == 3 ): break
    c = np.array(c).T
    return c

def import_wiki_vote_data(path):
    df = pd.read_csv(path, delimiter='\t', comment='#', header=None, names=['FromNodeId', 'ToNodeId'])
    df_values = df.values

    v_set = set()
    for i,j in df_values:
        v_set.add(i)
        v_set.add(j)

    nodes = len(v_set)
    v_set_dict = {}
    v_set_dict_rev = {}

    for i,j in enumerate(v_set):
        v_set_dict[j] = i
        v_set_dict_rev[i] = j

    dataset_graph = Graph(v_set_dict)

    for i,j in df_values:
        dataset_graph.add_edge(i,j)
    
    return dataset_graph, nodes

def import_lastfm_asia_data(path):
    df = pd.read_csv(path)
    df_values = df.values

    v_set = set()
    for i,j in df_values:
        v_set.add(i)
        v_set.add(j)

    nodes = len(v_set)
    v_set_dict = {}
    v_set_dict_rev = {}

    for i,j in enumerate(v_set):
        v_set_dict[j] = i
        v_set_dict_rev[i] = j

    dataset_graph = Graph(v_set_dict)

    for i,j in df_values:
        dataset_graph.add_edge(i,j)
    
    return dataset_graph, nodes

class Louvain:
    def __init__(self, G):
        self.G = G  # Graph as adjacency list
        self.node_to_community = {node: node for node in G}  # Initially each node is its own community
        self.community_to_nodes = {node: {node} for node in G}  # Communities with nodes
        self.total_weight = sum(sum(self.G[node].values()) for node in G)  # Sum of all edge weights in the graph

    def run(self, flag = False):
        while True:
            moved = self.phase_one()
            if not moved:
                break
            self.phase_two()
            if flag: break
        return self.node_to_community

    def phase_one(self):
        moved = False
        for node in self.G:
            current_community = self.node_to_community[node]
            best_community = current_community
            best_increase = 0

            self.remove_node_from_community(node, current_community)
            communities = self.get_neighbor_communities(node)

            for community in communities:
                increase = self.calculate_modularity_gain(node, community)
                if increase > best_increase:
                    best_community = community
                    best_increase = increase
            self.community_to_nodes[current_community] = self.community_to_nodes.get(current_community, set())
            self.community_to_nodes[current_community].add(node)
            self.add_node_to_community(node, best_community)
            if best_community != current_community:
                moved = True

        return moved

    def phase_two(self):
        new_graph = {}
        new_node_to_community = {}
        new_community_to_nodes = {}

        for community, nodes in self.community_to_nodes.items():
            new_node = tuple(sorted(self._flatten(nodes)))
            new_graph[new_node] = {}

            for node in nodes:
                for neighbor, weight in self.G[node].items():
                    neighbor_community = self.node_to_community[neighbor]
                    neighbor_node = tuple(sorted(self._flatten(self.community_to_nodes[neighbor_community])))

                    if neighbor_node == new_node:
                        continue

                    if neighbor_node not in new_graph[new_node]:
                        new_graph[new_node][neighbor_node] = 0
                    new_graph[new_node][neighbor_node] += weight

            new_node_to_community[new_node] = new_node
            new_community_to_nodes[new_node] = {new_node}  # Convert the tuple back to a set for community management

        self.G = new_graph
        self.node_to_community = new_node_to_community
        self.community_to_nodes = new_community_to_nodes
        self.total_weight = sum(sum(self.G[i].values()) for i in self.G) / 2

    def _flatten(self, nodes):
        flat_list = []
        for node in nodes:
            if isinstance(node, tuple):
                flat_list.extend(node)
            else:
                flat_list.append(node)
        return flat_list

    def remove_node_from_community(self, node, community):
        self.community_to_nodes[community].remove(node)
        if not self.community_to_nodes[community]:
            del self.community_to_nodes[community]

    def add_node_to_community(self, node, community):
        current_community = self.node_to_community[node]
        
        if current_community == community:
            self.community_to_nodes[community].add(node)
            return
        nodes_to_move = list(self.community_to_nodes[current_community])

        for n in nodes_to_move:
            self.community_to_nodes[current_community].remove(n)
            self.community_to_nodes[community].add(n)
            self.node_to_community[n] = community

        del self.community_to_nodes[current_community]

    def get_neighbor_communities(self, node):
        neighbor_communities = set()
        for neighbor in self.G[node]:
            neighbor_communities.add(self.node_to_community[neighbor])
        return neighbor_communities

    def calculate_modularity_gain(self, node, community):

        k_i_in = sum(self.G[node].get(neighbor, 0) for neighbor in self.community_to_nodes[community])
        sigma_tot = sum(sum(self.G[n].values()) for n in self.community_to_nodes[community])
        a = [self.G[node].get(neighbor, 0) for neighbor in self.community_to_nodes[community]]
        k_i = sum(self.G[node].values())
        delta_Q = (k_i_in / (2 * self.total_weight)) - ((sigma_tot * k_i) / (2 * (self.total_weight**2)))
        
        return delta_Q

def convert_to_nx_output(c):
    x = c.T
    nodes, levels = c.shape
    res = []
    for l in range(levels):
        comm_tup = []
        visited = [False]*nodes
        v = 0
        while v < nodes:
            comm = set()
            comm_id = -1
            for i in range(nodes):
                if visited[i]: continue
                if comm_id == -1:
                    comm.add(i)
                    comm_id = x[l][i]
                    visited[i] = True
                    v += 1
                elif x[l][i] == comm_id:
                    comm.add(i)
                    visited[i] = True
                    v += 1
            comm_tup.append(comm)
        res.append(tuple(comm_tup))
    return res

def visualise_dendogram(communities):
    communities = convert_to_nx_output(communities)
    print(communities)
    community_structures = list(communities)
    num_levels = len(community_structures)
    dendro_data = []

    for level in range(num_levels - 1):
        current_size = len(community_structures[level])
        next_size = len(community_structures[level + 1])
        dendro_data.append([current_size, next_size, 1])  # Example data; adjust as needed

    dendro_data = np.array(dendro_data)
    linked = linkage(dendro_data, 'single')

    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', distance_sort='descending')
    plt.title('Dendrogram for Girvan-Newman Algorithm')
    plt.xlabel('Number of Communities')
    plt.ylabel('Distance')
    plt.show()

def convert_to_integer_participation_list(mapping):
    max_value = max(max(key) if isinstance(key, tuple) else key for key in mapping)
    participation_list = [0] * (max_value + 1)
    
    for key in mapping:
        if isinstance(key, tuple):
            min_value = min(key)
            for element in key:
                participation_list[element] = min_value
        else:
            participation_list[key] = key
    
    return participation_list

def preprocess(adjacency_list):
    adjacency_dict = {}
    
    for node, neighbors in adjacency_list.items():
        if node not in adjacency_dict:
            adjacency_dict[node] = {}
        for neighbor in neighbors:
            if neighbor not in adjacency_dict:
                adjacency_dict[neighbor] = {}
            # Add or update the edge weight
            adjacency_dict[node][neighbor] = 1
            adjacency_dict[neighbor][node] = 1  # Assuming undirected graph
    
    return adjacency_dict

def louvain_one_iter(graph):
    G = preprocess(graph.get_adj_list())
    louvain = Louvain(G)
    communities = louvain.run(flag = True)
    communities = convert_to_integer_participation_list(communities)
    return communities

if __name__ == "__main__":

    nodes_connectivity_list_wiki, max_n = import_wiki_vote_data("Wiki-Vote.txt")
    graph_partition_wiki  = Girvan_Newman_one_level(nodes_connectivity_list_wiki, max_n)
    community_mat_wiki = Girvan_Newman(nodes_connectivity_list_wiki, max_n)
    visualise_dendogram(community_mat_wiki)

    graph_partition_louvain_wiki = louvain_one_iter(nodes_connectivity_list_wiki)


    nodes_connectivity_list_lastfm, max_n = import_lastfm_asia_data("lasftm_asia/lastfm_asia_edges.csv")
    graph_partition_lastfm = Girvan_Newman_one_level(nodes_connectivity_list_lastfm, max_n)
    community_mat_lastfm = Girvan_Newman(nodes_connectivity_list_lastfm, max_n)
    visualise_dendogram(community_mat_lastfm)
    graph_partition_louvain_lastfm = louvain_one_iter(nodes_connectivity_list_lastfm)
    print(graph_partition_louvain_lastfm)
    
