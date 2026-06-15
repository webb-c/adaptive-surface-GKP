import networkx as nx

from abc import ABC, abstractmethod
from graph import *
from utils import *


class Decoder(ABC):
    def __init__(self, d: int, n_round: int):
        self.d = d
        self.n_round = n_round

    @abstractmethod
    def make_syndrome_graph(self) -> None:
        pass
    
    @abstractmethod
    def make_decoding_graph(self) -> None:
        pass
    
    @abstractmethod
    def decode(self, syndrome) -> list:
        pass

class SurfaceDecoder(Decoder):
    def __init__(self, d: int, n_round: int):
        super().__init__(d, n_round)
        self.n_qubit_2d = self.d ** 2
        self.n_ancilla_2d = (self.d ** 2 - 1) // 2      
        self.n_virtual_2d = self.d + 1             
        self.n_vertex_2d = self.n_ancilla_2d + self.n_virtual_2d
        self.n_space_edge_2d = self.d ** 2 + self.d  
        self.n_space_edge_3d = (self.n_round + 1) * self.n_space_edge_2d  
        
        self.make_syndrome_graph()
        
        
    def make_syndrome_graph(self):
        self.z_synd_graph = None
        self.x_synd_graph = None
        
        pass


    def make_decoding_graph(self):
        pass


    def decode(self, syndrome):
        pass
    
    
class SurfaceGKPDecoder(SurfaceDecoder):
    def __init__(self, d: int, n_round: int, scale: int = 1000):
        super().__init__(d, n_round)
        self.scale = scale


    def make_syndrome_graph(self):
        self.z_synd_graph = ZSyndromeGraph(self.d, self.n_round)
        self.x_synd_graph = XSyndromeGraph(self.d, self.n_round)        


    def make_decoding_graph(self, n_z_check, n_x_check):
        z_nx_graph = self.z_synd_graph.get_nx_graph()
        z_check_synd_list = [node.id_3d for node in self.z_synd_graph.nodes if node.syndrome != 0]
        
        x_decoding_edges = np.zeros((int(n_z_check * (n_z_check - 1) / 2), 2)).astype(np.uint)
        x_decoding_weights = np.zeros((int(n_z_check * (n_z_check - 1) / 2), 1))
        counter = 0
        for i in range(n_z_check - 1):
            for j in range(i + 1, n_z_check):
                u = z_check_synd_list[i]
                v = z_check_synd_list[j]
                w, _ = nx.single_source_dijkstra(z_nx_graph, source=u, target=v, weight='weight')
                x_decoding_edges[counter, :2] = [u, v]
                x_decoding_weights[counter] = w
                counter += 1
        
        self.x_decoding_graph = nx.Graph()
        for (u, v), w in zip(x_decoding_edges, x_decoding_weights):
            self.x_decoding_graph.add_edge(int(u.item()), int(v.item()), weight = -w)     
        
        x_nx_graph = self.x_synd_graph.get_nx_graph()
        x_check_synd_list = [node.id_3d for node in self.x_synd_graph.nodes if node.syndrome != 0]
        
        z_decoding_edges = np.zeros((int(n_x_check * (n_x_check - 1) / 2), 2)).astype(np.uint)
        z_decoding_weights = np.zeros((int(n_x_check * (n_x_check - 1) / 2), 1))
        counter = 0
        for i in range(n_x_check - 1):
            for j in range(i + 1, n_x_check):
                u = x_check_synd_list[i]
                v = x_check_synd_list[j]
                w, _ = nx.single_source_dijkstra(x_nx_graph, source=u, target=v, weight='weight')
                z_decoding_edges[counter, :2] = [u, v]
                z_decoding_weights[counter] = w
                counter += 1
        
        self.z_decoding_graph = nx.Graph()
        for (u, v), w in zip(z_decoding_edges, z_decoding_weights):
            self.z_decoding_graph.add_edge(int(u.item()), int(v.item()), weight = -w)     
    
    
    def update_syndrome(self, Z_synd, X_synd, t):
        for idx, syndrome in enumerate(Z_synd):
            u = self.z_synd_graph._get_node_id(idx, t)
            self.z_synd_graph.update_node(u, syndrome)
        
        for idx, syndrome in enumerate(X_synd):
            u = self.x_synd_graph._get_node_id(idx, t)
            self.x_synd_graph.update_node(u, syndrome)
    
    
    def update_weight(self, Z_space_weight, X_space_weight, Z_time_weight, X_time_weight, t):
        Z_space_weight = np.round(self.scale * Z_space_weight)
        X_space_weight = np.round(self.scale * X_space_weight)
        Z_time_weight = np.round(self.scale * Z_time_weight)
        X_time_weight = np.round(self.scale * X_time_weight)
        
        for idx, weight in enumerate(Z_space_weight):
            if weight < 0:
                continue
            u_2d, v_2d = self.z_synd_graph.data_to_edge[idx]
            u = self.z_synd_graph._get_node_id(u_2d, t)
            v = self.z_synd_graph._get_node_id(v_2d, t)
            self.z_synd_graph.update_weight(u, v, weight)
            
        for idx, weight in enumerate(X_space_weight):
            if weight < 0:
                continue
            u_2d, v_2d = self.x_synd_graph.data_to_edge[idx]
            u = self.x_synd_graph._get_node_id(u_2d, t)
            v = self.x_synd_graph._get_node_id(v_2d, t)
            self.x_synd_graph.update_weight(u, v, weight)
        
        for idx, weight in enumerate(Z_time_weight):
            if weight < 0:
                continue
            u = self.z_synd_graph._get_node_id(idx, t)
            v = self.z_synd_graph._get_node_id(idx, t+1)
            self.z_synd_graph.update_weight(u, v, weight)
        
        for idx, weight in enumerate(X_time_weight):
            if weight < 0:
                continue
            u = self.x_synd_graph._get_node_id(idx, t)
            v = self.x_synd_graph._get_node_id(idx, t+1)
            self.x_synd_graph.update_weight(u, v, weight)
            

    
    def decode(self, verbose=False):
        def trans_3d_to_2d_idx(idx):
            if idx >= self.n_qubit_2d:
                return idx % self.n_qubit_2d
            return idx

        x_correction = np.zeros((self.n_qubit_2d))
        z_nx_graph = self.z_synd_graph.get_nx_graph()  
        if verbose:
            print("# of nodes in Z syndrome graph: ", len(self.x_decoding_graph))
            print_edges_for_nx(self.x_decoding_graph)
        
        X_matching = list(nx.algorithms.max_weight_matching(self.x_decoding_graph, maxcardinality=True, weight="weight"))  
        if not len(X_matching) == 0:
            for matching in X_matching:
                u, v = matching
                path = nx.shortest_path(z_nx_graph, source=u, target=v, weight='weight')
                for i in range(len(path) - 1):
                    pu, pv = path[i], path[i+1]
                    pu_qubits = self.z_synd_graph.get_all_data_qubit_id_3d(pu)                    
                    pv_qubits = self.z_synd_graph.get_all_data_qubit_id_3d(pv)
                    inter_qubit = np.intersect1d(pu_qubits, pv_qubits)
                    if len(inter_qubit) != 0:
                        inter_qubit_2d = trans_3d_to_2d_idx(inter_qubit[0]) # always 1?
                        temp = np.zeros((self.n_qubit_2d))
                        temp[inter_qubit_2d] = 1
                        x_correction = (x_correction + temp) % 2

        z_correction = np.zeros((self.n_qubit_2d))
        x_nx_graph = self.x_synd_graph.get_nx_graph()
        if verbose:
            print("# of nodes in X syndrome graph: ", len(self.z_decoding_graph))
            print_edges_for_nx(self.z_decoding_graph)
        
        Z_matching = list(nx.algorithms.matching.max_weight_matching(self.z_decoding_graph, maxcardinality=True, weight="weight"))
        if not len(Z_matching) == 0:
            for matching in Z_matching:
                u, v = matching
                path = nx.shortest_path(x_nx_graph, source=u, target=v, weight='weight')
                for i in range(len(path) - 1):
                    pu, pv = path[i], path[i+1]
                    pu_qubits = self.x_synd_graph.get_all_data_qubit_id_3d(pu)                    
                    pv_qubits = self.x_synd_graph.get_all_data_qubit_id_3d(pv)
                    inter_qubit = np.intersect1d(pu_qubits, pv_qubits)
                    
                    if len(inter_qubit) != 0:
                        inter_qubit_2d = trans_3d_to_2d_idx(inter_qubit[0]) 
                        temp = np.zeros((self.n_qubit_2d))
                        temp[inter_qubit_2d] = 1
                        z_correction = (z_correction + temp) % 2
            
        return x_correction, z_correction