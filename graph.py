import numpy as np
import networkx as nx

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, TypeAlias

NeighborMeta: TypeAlias = Tuple[int, float] 


@dataclass
class SyndromeNode:
    id_3d: int
    id_2d: int  
    t: int      
    syndrome: int = 0
    neighbors: Dict[int, NeighborMeta] = field(default_factory=dict)

class SyndromeGraph:
    def __init__(self, distance: int, n_round: int):
        self.d = distance
        self.n_round = n_round
        self.nodes: List[SyndromeNode] = []
        
        self._init_parameters()
        self._make_mapping()
        self._build_nodes()
        self._build_edges()

    
    def _init_parameters(self):
        self.n_data_qubit = self.d ** 2
        self.n_syndrome_2d = (self.d ** 2 - 1) // 2
        self.n_virtual_2d = self.d + 1  
        self.n_node_2d = self.n_syndrome_2d + self.n_virtual_2d
        
        self.n_space_edge_2d = self.d ** 2 + self.d
        self.n_time_edge_2d = self.n_node_2d
        self.n_edge_3d = self.n_space_edge_2d * (self.n_round + 1) + self.n_time_edge_2d * self.n_round

        self.neighbor_table = [[] for _ in range(self.n_node_2d)]
        self.data_to_edge = [[] for _ in range(self.n_data_qubit)]
        self.edge_to_data = {}
        self.nx_graph = None
    
    
    def _get_node_id(self, id_2d: int, t: int) -> int:
        return t * self.n_node_2d + id_2d


    def _get_data_qubit_id(self, u: int, v: int) -> int:
        key = (u, v) if u <= v else (v, u)
        return self.edge_to_data.get(key, -1)


    def _is_virtual(self, id_2d):
        if id_2d >= self.n_syndrome_2d:
            return True
        return False


    def _make_mapping(self):
        pass


    def _build_nodes(self):
        for t in range(self.n_round + 1):
            for id_2d in range(self.n_syndrome_2d):
                id_3d = self._get_node_id(id_2d, t)
                self.nodes.append(SyndromeNode(id_3d, id_2d, t))
            for r in range(self.n_virtual_2d):
                id_2d = self.n_syndrome_2d + r
                id_3d = self._get_node_id(id_2d, t)
                self.nodes.append(SyndromeNode(id_3d, id_2d, t))


    def _build_edges(self):
        pass
    

    def update_weight(self, source_id: int, dest_id: int, weight: float) -> None:
        eid, _ = self.nodes[source_id].neighbors[dest_id]
        self.nodes[source_id].neighbors[dest_id] = (eid, weight)
        self.nodes[dest_id].neighbors[source_id] = (eid, weight)


    def update_node(self, node_id: int, syndrome) -> None:
        self.nodes[node_id].syndrome = syndrome


    def make_nx_graph(self):
        self.nx_graph = nx.Graph()
        for node in self.nodes:
            for neigh, info in node.neighbors.items():
                self.nx_graph.add_edge(node.id_3d, neigh, weight = info[1])
    
    
    def get_lattice_graph(self):
        lattice_graph = nx.Graph()
        for i in range(self.n_syndrome_2d):
            for v in self.neighbor_table[i]:
                lattice_graph.add_edge(i, v, weight = 1.0)

        return lattice_graph
    
    def get_nx_graph(self):
        if self.nx_graph is None:
            self.make_nx_graph()
        return self.nx_graph
    
    
    def get_all_data_qubit_id_3d(self, u:int) -> List:
        qubits = []
        for neigh, info in self.nodes[u].neighbors.items():
            t = self.nodes[u].t
            data_idx = info[0]
            if data_idx != -1:
                qubits.append(data_idx + (t * self.n_data_qubit))
        
        return qubits
    
    
class ZSyndromeGraph(SyndromeGraph):
    def __init__(self, distance: int, n_round: int):
        super().__init__(distance, n_round)
    
    
    def _make_mapping(self) -> None:
        def _add(u: int, v: int):
            self.neighbor_table[u].append(v)
            self.neighbor_table[v].append(u)
        
        virtual_counter, syndrome_counter = 0, 0
        row_idx = 1

        for i in range(self.n_data_qubit):
            if i < self.d:
                u = self.n_syndrome_2d + virtual_counter
                v = syndrome_counter
                _add(u, v)
                if ((i + 1) % 2 == 0 and (i + 1) < self.d) or i == self.d - 1:
                    virtual_counter += 1
                if ((i + 1) % 2 == 1):
                    syndrome_counter += 1
            elif i > self.n_data_qubit - self.d - 1: 
                if i == self.n_data_qubit - self.d: 
                    syndrome_counter -= ((self.d - 1) // 2) + 1
                u = self.n_syndrome_2d + virtual_counter
                v = syndrome_counter
                _add(u, v)
                if ((i + 1) % 2 == 0):
                    syndrome_counter += 1
                else:
                    virtual_counter += 1
            else:
                u = syndrome_counter
                if row_idx % 2 == 0:            
                    if (i + 1) % 2 == 0:        
                        v = syndrome_counter - ((self.d + 1) // 2) - 1
                    else:                       
                        v = syndrome_counter - ((self.d + 1) // 2)
                        syndrome_counter += 1
                    if (i + 1) % self.d == 0:   
                        row_idx += 1
                else:
                    if (i + 1) % 2 == 0:        
                        v = syndrome_counter - ((self.d + 1) // 2)
                    else:                      
                        v = syndrome_counter - ((self.d + 1) // 2) + 1
                        syndrome_counter += 1
                    if (i + 1) % (2 * self.d) == 0:
                        syndrome_counter += 1
                        row_idx += 1
                _add(u, v)
            self.data_to_edge[i] = [u, v]
        virtual_counter = self.n_syndrome_2d
        _add(virtual_counter, virtual_counter + (self.n_virtual_2d // 2)) 
        for i in range(self.d):
            if (i + 1) == (self.n_virtual_2d // 2):
                virtual_counter += 1
                continue
            v = virtual_counter
            u = virtual_counter + 1
            _add(v, u)
            virtual_counter += 1
        
        self.neighbor_table = [sorted(set(lst)) for lst in self.neighbor_table]
        for i, (u, v) in enumerate(self.data_to_edge):
            key = (u, v) if u <= v else (v, u)
            self.edge_to_data[key] = i     
        
        
    def _build_edges(self):
        for node in self.nodes:
            id_3d, id_2d, t = node.id_3d, node.id_2d, node.t
            neighbor_list = self.neighbor_table[id_2d]
            for n_id_2d in neighbor_list:
                weight = 1.0
                if self._is_virtual(n_id_2d) and self._is_virtual(id_2d):
                    weight = 0.0
                n_id_3d = self._get_node_id(n_id_2d, t)
                node.neighbors[n_id_3d] = (self._get_data_qubit_id(id_2d, n_id_2d), weight) 
            for dt in [-1, 1]:
                nt = t + dt
                if nt < 0 or nt > self.n_round:
                    continue
                weight = 1.0
                if self._is_virtual(id_2d):
                    weight = 0.0
                n_id_3d = self._get_node_id(id_2d, nt)
                node.neighbors[n_id_3d] = (-1, weight)
    
    
class XSyndromeGraph(SyndromeGraph):
    def __init__(self, distance: int, n_round: int):
        super().__init__(distance, n_round)
    
    
    def _make_mapping(self):
        def _add(u: int, v: int):
            self.neighbor_table[u].append(v)
            self.neighbor_table[v].append(u)
        
        virtual_counter, syndrome_counter = 0, 0
        col_idx = 1
        
        for i in range(self.n_data_qubit):
            if i < self.d:
                u = self.n_syndrome_2d + virtual_counter
                v = syndrome_counter
                _add(u, v)
                if ((i + 1) % 2 == 1) or i == self.d - 1:
                    virtual_counter += 1
                if ((i + 1) % 2 == 0) or i == self.d - 1:
                    syndrome_counter += 1
            elif i > self.n_data_qubit - self.d - 1: 
                if i == self.n_data_qubit - self.d: 
                    syndrome_counter -= ((self.d + 1) // 2) 
                u = self.n_syndrome_2d + virtual_counter
                v = syndrome_counter
                _add(u, v)
                if ((i + 1) % 2 == 1):
                    syndrome_counter += 1
                else:
                    virtual_counter += 1
            else:
                u = syndrome_counter
                if col_idx % 2 == 0:            
                    if (i + 1) % 2 == 0:        
                        v = syndrome_counter - ((self.d + 1) // 2) + 1
                        syndrome_counter += 1
                    else:                      
                        v = syndrome_counter - ((self.d + 1) // 2)
                    if (i + 1) % self.d == 0:  
                        syndrome_counter += 1
                        col_idx += 1
                else:
                    if (i + 1) % 2 == 0:        
                        v = syndrome_counter - ((self.d + 1) // 2)
                        syndrome_counter += 1
                    else:                      
                        v = syndrome_counter - ((self.d + 1) // 2) - 1
                    if (i + 1) % (2 * self.d) == 0:
                        col_idx += 1
                _add(u, v)
                
            row, col = divmod(i, self.d)
            j = col*self.d + row
            self.data_to_edge[j] = [u, v]

        virtual_counter = self.n_syndrome_2d
        _add(virtual_counter, virtual_counter + (self.n_virtual_2d // 2))
        for i in range(self.d):
            if (i + 1) == (self.n_virtual_2d // 2):
                virtual_counter += 1
                continue
            v = virtual_counter
            u = virtual_counter + 1
            _add(v, u)
            virtual_counter += 1
        
        self.neighbor_table = [sorted(set(lst)) for lst in self.neighbor_table]
        for i, (u, v) in enumerate(self.data_to_edge):
            key = (u, v) if u <= v else (v, u)
            self.edge_to_data[key] = i     

    
    def _build_edges(self):
        for node in self.nodes:
            id_3d, id_2d, t = node.id_3d, node.id_2d, node.t
            neighbor_list = self.neighbor_table[id_2d]
            for n_id_2d in neighbor_list:
                weight = 1.0
                if self._is_virtual(n_id_2d) and self._is_virtual(id_2d):
                    weight = 0.0
                n_id_3d = self._get_node_id(n_id_2d, t)
                node.neighbors[n_id_3d] = (self._get_data_qubit_id(id_2d, n_id_2d), weight) 
            for dt in [-1, 1]:
                nt = t + dt
                if nt < 0 or nt > self.n_round:
                    continue
                weight = 1.0
                if self._is_virtual(id_2d):
                    weight = 0.0
                n_id_3d = self._get_node_id(id_2d, nt)
                node.neighbors[n_id_3d] = (-1, weight)