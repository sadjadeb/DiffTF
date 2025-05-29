import networkx as nx
import  dal.load_dblp_data as dblp


class DBLPGraph:
    def __init__(self):
        self.g = nx.Graph()
        self.authorID = None
        self.edgeHash = None
        self.nameID = None

    def load_authorID(self, path='authordota.txt'):
        ids, names = dblp.load_authors(path)
        self.authorID = {name: int(id_) for id_, name in zip(ids, names)}

    def load_nameID(self, path='authordota.txt'):
        ids, names = dblp.load_authors(path)
        self.nameID = {int(id_): name for id_, name in zip(ids, names)}

    def load_edgeHash(self, path=None):
        if path is None:
            print("No edgeHash file provided. Skipping edgeHash loading.")
            self.edgeHash = {}
            return

        self.edgeHash = {}
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                splitted = line.split('\t')
                nodeID = int(splitted[0])
                neighbors = []
                for neighbor in splitted[1:]:
                    if neighbor.strip():
                        neighborInfo = neighbor.split('#')
                        neighbors.append((int(neighborInfo[0]), float(neighborInfo[1])))
                self.edgeHash[nodeID] = neighbors



    def build_graph_from_team_data(self, team_data):
        self.g = nx.Graph()
        for team in team_data:
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    src, trg = team[i], team[j]
                    if self.g.has_edge(src, trg):
                        self.g[src][trg]['weight'] += 1
                    else:
                        self.g.add_edge(src, trg, weight=1)


    def load_files(self, path_authorID=None, path_edgeHash=None):
        if path_authorID is None:
            self.load_authorID()
        else:
            self.load_authorID(path=path_authorID)

        if path_authorID is None:
            self.load_nameID()
        else:
            self.load_nameID(path=path_authorID)

        if path_edgeHash is None:
            self.load_edgeHash()
        else:
            self.load_edgeHash(path=path_edgeHash)

    def build_graph(self, edgeHash=None, path='dataset/dota_graph.gpickle'):
        self.g = nx.Graph()
        if edgeHash is None:
            edgeHash = self.edgeHash
        nodes = edgeHash.keys()
        nodes = [int(i) for i in nodes]
        self.g.add_nodes_from(nodes)
        for node in self.g.nodes:
            edges = edgeHash.get(str(node))
            for edge in edges:
                if edge[0] in self.g.nodes:
                    self.g.add_edge(node, edge[0], weight=int(edge[1]))
                else:
                    # node is not in the keys list but in connections list!
                    print('Mismatch: A node is not in the keys list but in connections list!')
        nx.write_gpickle(self.g, path)

    def set_graph(self, g):
        self.g = g

    def get_graph(self):
        return self.g

    def read_graph(self, path='dataset/dota_graph.gpickle'):
        self.g = nx.read_gpickle(path=path)
        return self.g

    def write_graph(self, g=None, path='dataset/dota_graph.gpickle'):
        if g is None:
            g = self.g
        nx.write_gpickle(g, path)

    def shortest_path_id(self, src, trg, g=None, maxDist=10):
        if g is None:
            g = self.g
        try:
            return nx.shortest_path_length(g, src, trg)
        except nx.NetworkXNoPath:
            return maxDist

    def shortest_path_name(self, src, trg, g=None, maxDist=10):
        if self.nameID is None:
            self.load_nameID()
        if g is None:
            g = self.g
        # find id from name
        if src in self.nameID and trg in self.nameID:
            src = self.nameID.get(src)
            trg = self.nameID.get(trg)

            if src == trg:
                return 0
            try:
                return nx.shortest_path_length(g, src, trg)
            except nx.NetworkXNoPath:
                return maxDist

        else:
            #if src not in self.nameID:
            #    print('*** SRC: {} is not in the list! ***'.format(src))
            #if trg not in self.nameID:
            #    print('*** TRG: {} is not in the list! ***'.format(trg))
            return maxDist
