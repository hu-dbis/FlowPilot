import pygraphviz as pgv
import networkx as nx
import graphviz
from collections import defaultdict, deque
import re
from tqdm import tqdm

class my_graph:
    dot_file_path = ''
    graphviz_source = None
    graphviz_object = None
    nx_graph = None
    nodes = [] # list of pairs of nodeid and label
    edges = [] # list of triplets of node1, node2, and edge label
    adjacency_list = {}
    nf_ops_list = [x.upper() for x in list(
        {'branch', 'channel', 'collect', 'combine', 'emit', 'flatten', 'join', 'merge', 'output', 'scatter',
         'split', 'zip', 'map', 'filter', 'group', 'set', 'setval', 'mix', 'buffer', 'collate', 'collectFile',
         'concat', 'count', 'cross', 'distinct', 'emit', 'expand', 'filter', 'flatten', 'fold', 'group', 'head',
         'join', 'map', 'max', 'min', 'mix', 'output', 'pair', 'pick', 'reduce', 'reverse', 'sample', 'set',
         'setval', 'size', 'skip', 'sort', 'split', 'tail', 'take', 'toFile', 'toPath', 'toSet', 'toTuple',
         'unique', 'unzip', 'zip', 'countfasta', 'countFastq', 'countJson', 'countLines', 'cross', 'distinct',
         'dump', 'filter', 'first', 'flatmap', 'flatten', 'grouptuple', 'ifEmpty', 'join', 'last', 'merge', 'map',
         'max', 'min', 'mix', 'multiMap', 'randomSample', 'reduce', 'set', 'splitCsv', 'splitFasta', 'splitFastq',
         'splitJson', 'splitText', 'subscribe', 'sum', 'take', 'tap', 'toInteger', 'toList', 'toSortedList',
         'transpose', 'unique', 'until', 'view', ''})]

    def __init__(self, dot_file_path):
        if dot_file_path != '':
            self.dot_file_path = dot_file_path
            self.graphviz_source = graphviz.Source.from_file(dot_file_path)
            self.graphviz_object = pgv.AGraph(dot_file_path)
            self.nx_graph = nx.nx_agraph.from_agraph(self.graphviz_object)
            self.nodes = list(self.get_nodes_with_label().items())
            self.edges = [(nodes[0], nodes[1], e_label) for nodes, e_label in list(self.get_edges_with_label().items())]
            self.adjacency_list = {node[0]: [] for node in self.nodes}
            for v1, v2, edge_label in self.edges:
                self.adjacency_list[v1].append((v2, edge_label))

    def get_nodes(self):
        return self.nx_graph.nodes(data=True)

    def get_nodes_with_label(self, xlabel=True):
        """
        :param xlabel: whether or not consider xlabels if label doesn't exist
        :return: a dictionary of nodes, where the value is their labels
        """
        nodes = self.nx_graph.nodes(data=True)
        dict = {}
        for node in nodes:
            label = ''
            if 'label' in node[1]:
                label = node[1]['label']
            elif xlabel and 'xlabel' in node[1]:
                label = node[1]['xlabel']
            dict[node[0]] = label.upper()
        return dict

    def get_edges_with_label(self):
        """
        :return: a dictionary of edges (pair of nodes), where the value is their labels
        """
        edges = self.nx_graph.edges(data=True)
        dict = {}
        for edge in edges:
            label = ''
            if 'label' in edge[2]:
                label = edge[2]['label']
            dict[(edge[0], edge[1])] = label.lower()
        return dict

    def get_edges(self):
        return self.nx_graph.edges(data=True)

    def get_networkx_object(self):
        return self.nx_graph

    def rename_nx_nodes(self, mapping):
        self.nx_graph = nx.relabel_nodes(self.nx_graph, mapping)
        self.nodes = list(self.get_nodes_with_label().items())
        self.edges = [(nodes[0], nodes[1], e_label) for nodes, e_label in list(self.get_edges_with_label().items())]
        self.adjacency_list = {node[0]: [] for node in self.nodes}
        for v1, v2, edge_label in self.edges:
            self.adjacency_list[v1].append((v2, edge_label))

    def print_graph(self):
        print("Nodes:", self.nx_graph.nodes(data=True))
        print("Edges:", self.nx_graph.edges(data=True))

    def render_graph(self):
        self.graphviz_source.render(view=True)

    def print_node_attributes(self, attr='label'):
        print(f"Nodes and their attributes: {attr}")
        for node in self.graphviz_object.nodes():
            label = node.attr.get(attr, f"No {attr}")
            print(f"Node {node} has {attr}: {label}")

    def print_edge_attributes(self, attr='weight'):
        print(f"Edges and their attributes: {attr}")
        for edge in self.graphviz_object.edges():
            label = edge.attr.get(attr, f"No {attr}")
            print(f"Edge {edge} has {attr}: {label}")

    def suc(self, node):
        successors = []
        for edge in self.get_edges():
            if edge[0] == node:
                successors += [edge[1]]
        return successors

    def path_ngrams(self, n=3):

        def compute_paths(v, depth):
            if depth == 0:
                return [[v]]
            paths = []
            for w in self.suc(v):
                tmp = compute_paths(w, depth-1)
                for u in tmp:
                    paths.append([v] + u)
            return paths

        n_grams = []
        for v in [x[0] for x in self.get_nodes()]:
            computed_paths = compute_paths(v, n - 1)
            if len(computed_paths) > 0:
                n_grams = n_grams + computed_paths

        return n_grams

    def get_all_paths_with_edges(self, min_length=1, max_length=-1, include_graph_hierarchy=False):
        """
        Find all paths in the graph.
        :return: List of all possible paths, where each path is a list [V1, E1, V2, ...]
        """
        all_paths = []

        def dfs(current_node, path, current_length):
            # Append the current node to the path
            current_node_id = current_node[0]
            current_node_label = current_node[1]

            if include_graph_hierarchy:
                path.append(current_node_label)
            else:
                path.append(current_node_label.split(':')[-1])
            current_length += 1

            # Check if the path length exceeds the allowed length
            if max_length != -1 and current_length > max_length:
                path.pop()  # Backtrack: remove node
                return

            # If there are no outgoing edges, add the path to results if it matches the required length
            if current_node_id not in self.adjacency_list or not self.adjacency_list[current_node_id]:
                if (max_length == -1 or current_length <= max_length) and current_length >= min_length:
                    all_paths.append(path[:])  # Add a copy of the current path
            else:
                # Explore all neighbors
                for neighbor, edge_label in self.adjacency_list[current_node_id]:
                    # if (max_length == -1 or current_length < max_length) and current_length > min_length:
                    #     all_paths.append(path[:])  # Add a copy of the current path
                    path.append(edge_label)  # Add edge to path
                    dfs((neighbor, self.get_nodes_with_label()[neighbor]), path, current_length + 1)  # Recursive call
                    path.pop()  # Backtrack: remove edge

            path.pop()  # Backtrack: remove node

        # Start DFS from every vertex
        for node in self.nodes:
            dfs(node, [], 0)

        final_all_paths = []
        for path in all_paths:
            final_all_paths.append([x for x in path if x.strip() != '' and x[-1] not in self.nf_ops_list])
        return final_all_paths


    def get_all_paths_with_edges_for_given_nodes(self, min_length=1, max_length=-1, node_list=[]):
        """
        Find all paths in the graph.
        :return: List of all possible paths, where each path is a list [V1, E1, V2, ...]
        """
        all_paths = []

        def dfs(current_node, path, current_length):
            # Append the current node to the path
            current_node_id = current_node[0]
            current_node_label = current_node[1]

            path.append(current_node_label.split(':')[-1])
            current_length += 1

            # Check if the path length exceeds the allowed length
            if max_length != -1 and current_length > max_length:
                path.pop()  # Backtrack: remove node
                return

            # If there are no outgoing edges, add the path to results if it matches the required length
            if current_node_id not in self.adjacency_list or not self.adjacency_list[current_node_id]:
                if (max_length == -1 or current_length <= max_length) and current_length >= min_length:
                    all_paths.append(path[:])  # Add a copy of the current path
            else:
                # Explore all neighbors
                for neighbor, edge_label in self.adjacency_list[current_node_id]:
                    if (max_length == -1 or current_length < max_length) and current_length > min_length:
                        all_paths.append(path[:])  # Add a copy of the current path
                    path.append(edge_label)  # Add edge to path
                    dfs((neighbor, self.get_nodes_with_label()[neighbor]), path, current_length + 1)  # Recursive call
                    path.pop()  # Backtrack: remove edge

            path.pop()  # Backtrack: remove node

        # Start DFS from every vertex
        for node in self.nodes:
            if node[0] in node_list:
                dfs(node, [], 0)

        final_all_paths = []
        for path in all_paths:
            final_all_paths.append([x for x in path if x.strip() != ''])
        return final_all_paths

    def is_connected(self, subgraph_nodes, edges):
        """Check if a subset of nodes is connected in the graph using BFS."""
        if not subgraph_nodes:
            return False

        # Build adjacency list for the subset
        adjacency_list = defaultdict(list)
        for u, v in edges:
            if u in subgraph_nodes and v in subgraph_nodes:
                adjacency_list[u].append(v)
                adjacency_list[v].append(u)

        # Perform BFS/DFS to check connectivity
        visited = set()
        queue = deque([next(iter(subgraph_nodes))])  # Start with any node from the subset

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(neighbor for neighbor in adjacency_list[node] if neighbor not in visited)

        return len(visited) == len(subgraph_nodes)

    def generate_connected_subgraphs(self):
        """Get all connected subgraphs by a recursive procedure"""

        con_comp = [c for c in sorted(nx.connected_components(self.nx_graph), key=len, reverse=True)]

        def recursive_local_expand(node_set, possible, excluded, results, max_size):
            """
            Recursive function to add an extra node to the subgraph being formed
            """
            results.append(node_set)
            if len(node_set) == max_size:
                return
            for j in possible - excluded:
                new_node_set = node_set | {j}
                excluded = excluded | {j}
                new_possible = (possible | set(self.nx_graph.neighbors(j))) - excluded
                recursive_local_expand(new_node_set, new_possible, excluded, results, max_size)

        results = []
        for cc in con_comp:
            max_size = len(cc)

            excluded = set()
            for i in self.nx_graph:
                excluded.add(i)
                recursive_local_expand({i}, set(self.nx_graph.neighbors(i)) - excluded, excluded, results, max_size)

        results.sort(key=len)

        node_to_label = self.get_nodes_with_label()
        all_edges = [(edge[0], edge[1]) for edge in self.get_edges()]
        subgraphs_labeled_nodes_and_edges = []
        for subgraph in results:
            current_subgraph_with_label = []
            current_subgraph_edges = []
            for nodeid in subgraph:
                current_subgraph_with_label.append((nodeid, node_to_label[nodeid]))
            for edge in all_edges:
                if edge[0] in subgraph and edge[1] in subgraph:
                    current_subgraph_edges.append(edge)
            subgraphs_labeled_nodes_and_edges.append((current_subgraph_with_label, current_subgraph_edges))
        return subgraphs_labeled_nodes_and_edges

    def get_max_node_id(self):
        nodes = [x[0] for x in self.get_nodes()]
        return max([int(node_id.replace('v', '')) for node_id in nodes])

    def convert_edges_to_labeled_intermediate_nodes(self):
        # the function yet needs to be tested
        current_node_id = self.get_max_node_id() + 1
        nodes = list(self.get_nodes_with_label().items())
        edges = self.get_edges_with_label().items()

        updated_edge_list = []
        for edge in edges:
            if edge[1] == '':
                updated_edge_list.append(edge[0])
                continue
            nodes.append((f'v{current_node_id}', edge[1]))
            updated_edge_list.append((edge[0][0], f'v{current_node_id}'))
            updated_edge_list.append((f'v{current_node_id}', edge[0][0]))
            current_node_id += 1

        G = nx.Graph()
        G.add_nodes_from([(node[0], {'label':node[1]}) for node in nodes])
        G.add_edges_from(updated_edge_list)
        self.nx_graph = G
        return nodes, updated_edge_list

    def return_subgraphs(self, html_graph_representation_path):
        '''
        This function returns the subgraphs already in the original graph
        :return:
        '''
        f = open(html_graph_representation_path)
        graph_text = f.read()
        subgraphs = {}
        edges = []

        lines = graph_text.splitlines()
        stack = []
        current_subgraph = None
        random_counter = 1

        for line in lines:
            line = line.strip()
            if line.startswith("subgraph"):
                # Start of a new subgraph
                subgraph_name = re.match(r'subgraph\s+"?(.*?)"?\s*$', line).group(1).strip()
                if subgraph_name == '':
                    subgraph_name = f'sg_{random_counter}'
                    random_counter += 1
                subgraph = {"name": subgraph_name, "nodes": set(), "edges": [], "subgraphs": []}
                if current_subgraph is not None:
                    subgraphs[current_subgraph]["subgraphs"].append(subgraph_name)
                subgraphs[subgraph_name] = subgraph
                stack.append(current_subgraph)
                current_subgraph = subgraph_name
            elif line == "end":
                # End of the current subgraph
                current_subgraph = stack.pop()
            elif " --> " in line:
                # Edge definition
                source, target = re.findall(r'v\d+', line)
                edges.append((source, target))
                if current_subgraph is not None:
                    subgraphs[current_subgraph]["edges"].append((source, target))
            elif re.match(r'v\d+(\[|\()', line):
                # Node definition
                node = re.match(r'(v\d+)(\[|\()', line).group(1)
                if current_subgraph is not None:
                    subgraphs[current_subgraph]["nodes"].add(node)

        for subgraph in subgraphs:
            nodes = subgraphs[subgraph]['nodes']
            subgraphs[subgraph]['edges'] = [edge for edge in edges if (edge[0] in nodes or edge[1] in nodes)]
        return subgraphs

    def find_roots(self):
        all_nodes = set(self.adjacency_list.keys())
        child_nodes = {child for children in self.adjacency_list.values() for child in children}
        roots = all_nodes - child_nodes
        return roots