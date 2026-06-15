import numpy as np

from abc import ABC, abstractmethod
from noise import SurfaceGKPNoiseModel
from decoder import *
from utils import *


class Simulator(ABC):
    def __init__(self, d: int, n_round: int, n_iter: int):
        self.d = d
        self.n_round = n_round
        self.n_iter = n_iter
        self.noise = None
        self.decoder = None

    @abstractmethod
    def _init_parameters(self) -> None:
        pass

    @abstractmethod
    def _init_iter(self) -> None:
        pass

    @abstractmethod
    def _init_round(self) -> None:
        pass


class SurfaceGKPSimulator(Simulator):
    def __init__(self, d: int, n_round: int, n_iter: int, sigma, sigma_GKP, sigma_idle, with_info=True, clean=False, fix_seed=False, verbose=False, check_counter=False, final_noise=False) -> None:
        super().__init__(d, n_round, n_iter)
        self.sigma = sigma
        self.sigma_GKP = sigma_GKP
        self.sigma_idle = sigma_idle
        self.noise = SurfaceGKPNoiseModel(d, n_round, sigma, sigma_GKP, sigma_idle)
        self.decoder = SurfaceGKPDecoder(d, n_round)
        self.with_info = with_info
        self.clean = clean
        self.fix_seed = fix_seed
        self.seed_idx = 0
        self.seed_list = None
        self.verbose = verbose
        self.final_noise = final_noise
        self._init_parameters()
        self._make_connect_data_syndrome_table()

        self.check_counter = check_counter
        self.n_decoding_node = 0
        self.n_idle_decoding_node = 0
        self.n_decoding_node_diff = 0
        self.miss_count = 0
        self.wrong_count = 0
        self.det_node_fn = 0
        self.det_node_fp = 0
        self.det_node_tp = 0
        self.det_node_mismatch = 0
        self.det_node_ideal_total = 0
        self.det_node_obs_total = 0

        self.prev_ideal_Z_synd = None
        self.prev_ideal_X_synd = None
        self._counter_ideal_Z_curr = None
        self._counter_ideal_X_curr = None

    def set_seed(self, seed_list):
        self.seed_list = list(seed_list) if seed_list is not None else None
        self.seed_idx = 0

    def _next_seed(self) -> int:
        if self.seed_list is None:
            raise ValueError("seed_list is undefined.")
        if self.seed_idx >= len(self.seed_list):
            raise IndexError("seed_list is exhausted.")
        s = self.seed_list[self.seed_idx]
        self.seed_idx += 1
        return int(s)

    def _rand_Gvec(self, cov, I):
        if not self.fix_seed:
            return rand_Gvec(cov, I)
        seed = 42
        return rand_Gvec(cov, I, rng=np.random.default_rng(seed))

    def _make_connect_data_syndrome_table(self):
        n_syndrome_1d = (self.d + 1) // 2

        def Z1(k):
            x = k % n_syndrome_1d
            y = (k - x) / n_syndrome_1d
            if y % 2 == 0:
                return int(self.d * y + 2 * x)
            else:
                if x != n_syndrome_1d - 1:
                    return int(self.d * y + 2 * x + 1)
            return - 1

        def Z2(k):
            x = k % n_syndrome_1d
            y = (k - x) / n_syndrome_1d
            if y % 2 == 0:
                return int(self.d * (y + 1) + 2 * x)
            else:
                if x != n_syndrome_1d - 1:
                    return int(self.d * (y + 1) + 2 * x + 1)
            return -1

        def Z3(k):
            x = k % n_syndrome_1d
            y = (k - x) / n_syndrome_1d
            if y % 2 == 0:
                if x != 0:
                    return self.d * y + 2 * x - 1
            else:
                return self.d * y + 2 * x
            return -1

        def Z4(k):
            x = k % n_syndrome_1d
            y = (k - x) / n_syndrome_1d
            if y % 2 == 1:
                return self.d * (y + 1) + 2 * x
            else:
                if x != 0:
                    return self.d * (y + 1) + 2 * x - 1
            return -1

        def X1(k):
            x = k % n_syndrome_1d
            y = (k - x) / n_syndrome_1d
            if y % 2 == 0:
                return 2 * self.d * x + y + 1
            else:
                if x != 0:
                    return self.d * (2 * x - 1) + y + 1
            return -1

        def X2(k):
            x = k % n_syndrome_1d
            y = (k - x) / n_syndrome_1d
            if y % 2 == 0:
                return 2 * self.d * x + y
            else:
                if x != 0:
                    return self.d * (2 * x - 1) + y
            return -1

        def X3(k):
            x = k % n_syndrome_1d
            y = (k - x) / n_syndrome_1d
            if y % 2 == 1:
                return 2 * self.d * x + y + 1
            else:
                if x != n_syndrome_1d - 1:
                    return self.d * (2 * x + 1) + y + 1
            return -1

        def X4(k):
            x = k % n_syndrome_1d
            y = (k - x) / n_syndrome_1d
            if y % 2 == 1:
                return 2 * self.d * x + y
            else:
                if x != n_syndrome_1d - 1:
                    return self.d * (2 * x + 1) + y
            return -1

        self.Z_mat = [np.zeros((self.n_data_qubit, self.n_syndrome_2d), dtype = np.uint8) for _ in range(4)]
        self.X_mat = [np.zeros((self.n_data_qubit, self.n_syndrome_2d), dtype = np.uint8) for _ in range(4)]
        Z_func = [Z1, Z2, Z3, Z4]
        X_func = [X1, X2, X3, X4]
        for j in range(self.n_data_qubit):
            for k in range(self.n_syndrome_2d):
                for step, func in enumerate(Z_func):
                    if j == func(k):
                        self.Z_mat[step][j, k] = 1
                for step, func in enumerate(X_func):
                    if j == func(k):
                        self.X_mat[step][j, k] = 1

    def _init_parameters(self):
        self.n_data_qubit = self.d ** 2
        self.n_syndrome_2d = (self.d ** 2 - 1) // 2
        self.n_virtual_2d = self.d + 1
        self.n_node_2d = self.n_syndrome_2d + self.n_virtual_2d

        self.n_space_edge_2d = self.d ** 2 + self.d
        self.n_time_edge_2d = self.n_node_2d
        self.n_edge_3d = self.n_space_edge_2d * (self.n_round + 1) + self.n_time_edge_2d * self.n_round

        self.D, self.O1, self.O2, self.O3, self.O4 = self.noise.get_error_matrix()

        self.I = np.eye(2)
        self.cov_circ = (self.sigma ** 2) * self.I
        self.cov_idle = (self.sigma_idle ** 2) * self.I
        self.I_qubit = np.eye(self.n_data_qubit)
        self.I_stab = np.eye(self.n_syndrome_2d)
        self.cov_circ_qubit = (self.sigma ** 2)  * self.I_qubit
        self.cov_circ_stab = (self.sigma ** 2)  * self.I_stab
        self.cov_idle_qubit = (self.sigma_idle ** 2)  * self.I_qubit
        self.cov_idle_stab = (self.sigma_idle ** 2)  * self.I_stab
        self.cov_GKP_qubit = (self.sigma_GKP ** 2)  * self.I_qubit
        self.cov_GKP_stab = (self.sigma_GKP ** 2)  * self.I_stab

        self.prev_Z_synd = None
        self.prev_X_synd = None

        self.Z_lattice_graph = self.decoder.z_synd_graph.get_lattice_graph()
        self.X_lattice_graph = self.decoder.x_synd_graph.get_lattice_graph()

    def _init_iter(self):
        self.decoder = SurfaceGKPDecoder(self.d, self.n_round)

        self.data_q = np.zeros((self.n_data_qubit))
        self.data_p = np.zeros((self.n_data_qubit))
        self.prev_Z_synd = np.zeros((self.n_node_2d))
        self.prev_X_synd = np.zeros((self.n_node_2d))

        self.prev_ideal_Z_synd = np.zeros((self.n_node_2d), dtype=int)
        self.prev_ideal_X_synd = np.zeros((self.n_node_2d), dtype=int)
        self._counter_ideal_Z_curr = None
        self._counter_ideal_X_curr = None

        self.n_z_check = 0
        self.n_x_check = 0
        self.n_z_idle_check = 0
        self.n_x_idle_check = 0

        self._init_round()

    def _init_round(self):
        self.synd_q = np.zeros((self.n_data_qubit))
        self.synd_p = np.zeros((self.n_data_qubit))
        self.Z_synd_q = np.zeros((self.n_syndrome_2d))
        self.Z_synd_p = np.zeros((self.n_syndrome_2d))
        self.X_synd_q = np.zeros((self.n_syndrome_2d))
        self.X_synd_p = np.zeros((self.n_syndrome_2d))

        self.Z_space_weight = - np.ones((self.n_space_edge_2d))
        self.X_space_weight = - np.ones((self.n_space_edge_2d))
        self.Z_time_weight = - np.ones((self.n_node_2d))
        self.X_time_weight = - np.ones((self.n_node_2d))

        self.Z_synd = np.zeros((self.n_node_2d))
        self.X_synd = np.zeros((self.n_node_2d))

    def _apply_GKP_noise(self, data=False, GKP=False, surface=False, idx_list=None):
        if data:
            self.data_q += self._rand_Gvec(self.cov_GKP_qubit, self.I_qubit)
            self.data_p += self._rand_Gvec(self.cov_GKP_qubit, self.I_qubit)
        if GKP:
            self.synd_q += self._rand_Gvec(self.cov_GKP_qubit, self.I_qubit)
            self.synd_p += self._rand_Gvec(self.cov_GKP_qubit, self.I_qubit)
        if surface:
            self.Z_synd_q += self._rand_Gvec(self.cov_GKP_stab, self.I_stab)
            self.Z_synd_p += self._rand_Gvec(self.cov_GKP_stab, self.I_stab)
            self.X_synd_q += self._rand_Gvec(self.cov_GKP_stab, self.I_stab)
            self.X_synd_p += self._rand_Gvec(self.cov_GKP_stab, self.I_stab)

    def _apply_circuit_noise(self, data=False, GKP=False, surface=False, surface_end=False):
        if data:
            self.data_q += self._rand_Gvec(self.cov_circ_qubit, self.I_qubit)
            self.data_p += self._rand_Gvec(self.cov_circ_qubit, self.I_qubit)
        if GKP:
            self.synd_q += self._rand_Gvec(self.cov_circ_qubit, self.I_qubit)
            self.synd_p += self._rand_Gvec(self.cov_circ_qubit, self.I_qubit)
        if surface:
            self.Z_synd_q += self._rand_Gvec(self.cov_circ_stab, self.I_stab)
            self.Z_synd_p += self._rand_Gvec(self.cov_circ_stab, self.I_stab)
            self.X_synd_q += self._rand_Gvec(self.cov_circ_stab, self.I_stab)
            self.X_synd_p += self._rand_Gvec(self.cov_circ_stab, self.I_stab)

    def _apply_idle_noise(self, data=False, GKP=False, surface=False):
        if data:
            self.data_q += self._rand_Gvec(self.cov_idle_qubit, self.I_qubit)
            self.data_p += self._rand_Gvec(self.cov_idle_qubit, self.I_qubit)
        if GKP:
            self.synd_q += self._rand_Gvec(self.cov_idle_qubit, self.I_qubit)
            self.synd_p += self._rand_Gvec(self.cov_idle_qubit, self.I_qubit)
        if surface:
            self.Z_synd_q += self._rand_Gvec(self.cov_idle_stab, self.I_stab)
            self.Z_synd_p += self._rand_Gvec(self.cov_idle_stab, self.I_stab)
            self.X_synd_q += self._rand_Gvec(self.cov_idle_stab, self.I_stab)
            self.X_synd_p += self._rand_Gvec(self.cov_idle_stab, self.I_stab)

    def _surface_measure(self, step: int, faulty: bool):
        if step in [0, 3]:
            self.data_q += self.X_mat[step] @ self.X_synd_q
            self.data_p -= self.Z_mat[step] @ self.Z_synd_p
            self.Z_synd_q += self.Z_mat[step].T @ self.data_q
            self.X_synd_p -= self.X_mat[step].T @ self.data_p

            if faulty:
                for i in range(self.n_syndrome_2d):
                    Z_access_idx = find_one(self.Z_mat[step][:, i])
                    if Z_access_idx != -1:
                        rand_O1 = self._rand_Gvec(self.D, self.O1)
                        rand_O2 = self._rand_Gvec(self.D, self.O2)
                        self.data_q[Z_access_idx] += rand_O1[0]
                        self.Z_synd_q[i] += rand_O1[1]
                        self.data_p[Z_access_idx] += rand_O2[0]
                        self.Z_synd_p[i] += rand_O2[1]
                    else:
                        rand_G = self._rand_Gvec(self.cov_circ, self.I)
                        self.Z_synd_q[i] += rand_G[0]
                        self.Z_synd_p[i] += rand_G[1]

                    X_access_idx = find_one(self.X_mat[step][:, i])
                    if X_access_idx != -1:
                        rand_O3 = self._rand_Gvec(self.D, self.O3)
                        rand_O4 = self._rand_Gvec(self.D, self.O4)
                        self.data_q[X_access_idx] += rand_O3[0]
                        self.X_synd_q[i] += rand_O3[1]
                        self.data_p[X_access_idx] += rand_O4[0]
                        self.X_synd_p[i] += rand_O4[1]
                    else:
                        rand_G = self._rand_Gvec(self.cov_circ, self.I)
                        self.X_synd_q[i] += rand_G[0]
                        self.X_synd_p[i] += rand_G[1]

        elif step in [1, 2]:
            self.data_q -= self.X_mat[step] @ self.X_synd_q
            self.data_p -= self.Z_mat[step] @ self.Z_synd_p
            self.Z_synd_q += self.Z_mat[step].T @ self.data_q
            self.X_synd_p += self.X_mat[step].T @ self.data_p

            if faulty:
                for i in range(self.n_syndrome_2d):
                    Z_access_idx = find_one(self.Z_mat[step][:, i])
                    if Z_access_idx != -1:
                        rand_O1 = self._rand_Gvec(self.D, self.O1)
                        rand_O2 = self._rand_Gvec(self.D, self.O2)
                        self.data_q[Z_access_idx] += rand_O1[0]
                        self.Z_synd_q[i] += rand_O1[1]
                        self.data_p[Z_access_idx] += rand_O2[0]
                        self.Z_synd_p[i] += rand_O2[1]
                    else:
                        rand_G = self._rand_Gvec(self.cov_circ, self.I)
                        self.Z_synd_q[i] += rand_G[0]
                        self.Z_synd_p[i] += rand_G[1]

                    X_access_idx = find_one(self.X_mat[step][:, i])
                    if X_access_idx != -1:
                        rand_O1 = self._rand_Gvec(self.D, self.O1)
                        rand_O2 = self._rand_Gvec(self.D, self.O2)
                        self.data_q[X_access_idx] += rand_O2[0]
                        self.X_synd_q[i] += rand_O2[1]
                        self.data_p[X_access_idx] += rand_O1[0]
                        self.X_synd_p[i] += rand_O1[1]
                    else:
                        rand_G = self._rand_Gvec(self.cov_circ, self.I)
                        self.X_synd_q[i] += rand_G[0]
                        self.X_synd_p[i] += rand_G[1]

        else:
            raise ValueError("Step index allowed only in [0-3]")

        if faulty:
            for i in range(self.n_data_qubit):
                if np.sum(self.Z_mat[step][i, :] + self.X_mat[step][i, :]) == 0:
                    rand_G = self._rand_Gvec(self.cov_circ, self.I)
                    self.data_q[i] += rand_G[0]
                    self.data_p[i] += rand_G[1]

    def _GKP_measure(self, step: int, faulty: bool):
        if step == 0:
            self.data_q -= keep_odd(self.synd_q)
            self.data_p -= keep_even(self.synd_p)
            self.synd_q += keep_even(self.data_q)
            self.synd_p += keep_odd(self.data_p)

            if faulty:
                for i in range(self.n_data_qubit):
                    rand_O1 = self._rand_Gvec(self.D, self.O1)
                    rand_O2 = self._rand_Gvec(self.D, self.O2)
                    if i % 2 == 0:
                        self.data_q[i] += rand_O1[0]
                        self.synd_q[i] += rand_O1[1]
                        self.data_p[i] += rand_O2[0]
                        self.synd_p[i] += rand_O2[1]
                    else:
                        self.data_q[i] += rand_O2[0]
                        self.synd_q[i] += rand_O2[1]
                        self.data_p[i] += rand_O1[0]
                        self.synd_p[i] += rand_O1[1]

                self._apply_idle_noise(data=True, GKP=False)
                self._apply_circuit_noise(data=False, GKP=True)

            even_q = keep_even(self.synd_q)
            odd_p = keep_odd(self.synd_p)
            for i in range(self.n_data_qubit):
                self.synd_q[i] = transform_centered_mod(np.sqrt(np.pi), even_q[i])
                self.synd_p[i] = transform_centered_mod(np.sqrt(np.pi), odd_p[i])

        elif step == 1:
            self.data_q -= keep_even(self.synd_q)
            self.data_p -= keep_odd(self.synd_p)
            self.synd_q += keep_odd(self.data_q)
            self.synd_p += keep_even(self.data_p)

            if faulty:
                for i in range(self.n_data_qubit):
                    rand_O1 = self._rand_Gvec(self.D, self.O1)
                    rand_O2 = self._rand_Gvec(self.D, self.O2)
                    if i % 2 == 0:
                        self.data_q[i] += rand_O2[0]
                        self.synd_q[i] += rand_O2[1]
                        self.data_p[i] += rand_O1[0]
                        self.synd_p[i] += rand_O1[1]
                    else:
                        self.data_q[i] += rand_O1[0]
                        self.synd_q[i] += rand_O1[1]
                        self.data_p[i] += rand_O2[0]
                        self.synd_p[i] += rand_O2[1]

                self._apply_idle_noise(data=True, GKP=False)
                self._apply_circuit_noise(data=False, GKP=True)

            odd_q = keep_odd(self.synd_q)
            even_p = keep_even(self.synd_p)
            for i in range(self.n_data_qubit):
                self.synd_q[i] = transform_centered_mod(np.sqrt(np.pi), odd_q[i])
                self.synd_p[i] = transform_centered_mod(np.sqrt(np.pi), even_p[i])

        else:
            raise ValueError("Step index allowed only in [0-1]")

    def _GKP_correction(self, step):
        if step == 0:
            self.data_q -= keep_even(self.synd_q)
            self.data_p -= keep_odd(self.synd_p)
        elif step == 1:
            self.data_q -= keep_odd(self.synd_q)
            self.data_p -= keep_even(self.synd_p)
        else:
            raise ValueError("Step index allowed only in [0-1]")

    def _surface_correction(self, x_correction, z_correction):
        data_q_corr = (self.data_q + np.sqrt(np.pi) * x_correction) / np.sqrt(np.pi)
        data_p_corr = (self.data_p + np.sqrt(np.pi) * z_correction) / np.sqrt(np.pi)
        return data_q_corr, data_p_corr

    def _cal_surface_syndrome(self):
        for i in range(self.n_syndrome_2d):
            self.Z_synd_q[i] = transform_centered_mod(2 * np.sqrt(np.pi), self.Z_synd_q[i])
            self.X_synd_p[i] = transform_centered_mod(2 * np.sqrt(np.pi), self.X_synd_p[i])

        for i in range(self.n_syndrome_2d):
            if abs(self.Z_synd_q[i]) > np.sqrt(np.pi) / 2:
                self.Z_synd[i] = 1
            if abs(self.X_synd_p[i]) > np.sqrt(np.pi) / 2:
                self.X_synd[i] = 1

    def _cal_space_weight(self, step:int, t: int, with_info=True):
        if with_info:
            for i in range(self.n_data_qubit):
                if i % 2 == step:
                    self.Z_space_weight[i] = -np.log2(cond_err_prob(self.noise.get_Z_sigma_space(i, t, step), self.synd_q[i]))
                else:
                    self.X_space_weight[i] = -np.log2(cond_err_prob(self.noise.get_X_sigma_space(i, t, step), self.synd_p[i]))
        else:
            for i in range(self.n_data_qubit):
                if i % 2 == step:
                    self.Z_space_weight[i] = -np.log2(err_prob(self.noise.get_Z_sigma_space(i, t, step)))
                else:
                    self.X_space_weight[i] = -np.log2(err_prob(self.noise.get_X_sigma_space(i, t, step)))

    def _cal_time_weight(self, with_info=True):
        if with_info:
            for i in range(self.n_syndrome_2d):
                self.Z_time_weight[i] = -np.log2(cond_err_prob(self.noise.get_Z_sigma_time(i), transform_centered_mod(np.sqrt(np.pi), self.Z_synd_q[i])))
                self.X_time_weight[i] = -np.log2(cond_err_prob(self.noise.get_X_sigma_time(i), transform_centered_mod(np.sqrt(np.pi), self.X_synd_p[i])))
        else:
            for i in range(self.n_syndrome_2d):
                self.Z_time_weight[i] = -np.log2(err_prob(self.noise.get_Z_sigma_time(i)))
                self.X_time_weight[i] = -np.log2(err_prob(self.noise.get_X_sigma_time(i)))

    def _find_nearest_virtual(self, Z_check, X_check):
        Z_virtual, X_virtual = 0, 0

        if Z_check:
            Z_synd_path, best_len = None, float("inf")
            source_nodes = np.flatnonzero(self.Z_synd == 1)
            target_nodes = [n for n in self.Z_lattice_graph.nodes if n >= self.n_syndrome_2d]

            for s in source_nodes:
                if s >= self.n_syndrome_2d:
                    continue
                lengths = nx.single_source_shortest_path_length(self.Z_lattice_graph, s)
                for t in target_nodes:
                    if t in lengths and lengths[t] < best_len:
                        best_len = lengths[t]
                        Z_synd_path = nx.shortest_path(self.Z_lattice_graph, s, t)
                        Z_virtual = t

        if X_check:
            X_synd_path, best_len = None, float("inf")
            source_nodes = np.flatnonzero(self.X_synd == 1)
            target_nodes = [n for n in self.X_lattice_graph.nodes if n >= self.n_syndrome_2d]
            nodes = self.X_lattice_graph.nodes

            for s in source_nodes:
                if s >= self.n_syndrome_2d:
                    continue
                lengths = nx.single_source_shortest_path_length(self.X_lattice_graph, s)
                for t in target_nodes:
                    if t in lengths and lengths[t] < best_len:
                        best_len = lengths[t]
                        X_synd_path = nx.shortest_path(self.X_lattice_graph, s, t)
                        X_virtual = t

        return Z_virtual, X_virtual

    def _attach_virtual_nodes_to_arrays(self, Z_synd: np.ndarray, X_synd: np.ndarray):
        Z_full = np.array(Z_synd, copy=True, dtype=int)
        X_full = np.array(X_synd, copy=True, dtype=int)

        def nearest_virtual(synd_arr: np.ndarray, lattice_graph):
            if np.sum(synd_arr[:self.n_syndrome_2d]) % 2 == 0:
                return 0

            best_virtual = 0
            best_len = float("inf")

            source_nodes = np.flatnonzero(synd_arr[:self.n_syndrome_2d] == 1)
            target_nodes = [n for n in lattice_graph.nodes if n >= self.n_syndrome_2d]

            for s in source_nodes:
                lengths = nx.single_source_shortest_path_length(lattice_graph, s)
                for v in target_nodes:
                    if v in lengths and lengths[v] < best_len:
                        best_len = lengths[v]
                        best_virtual = v
            return best_virtual

        z_virtual = nearest_virtual(Z_full, self.Z_lattice_graph)
        x_virtual = nearest_virtual(X_full, self.X_lattice_graph)

        if z_virtual != 0:
            Z_full[z_virtual] = 1
        if x_virtual != 0:
            X_full[x_virtual] = 1

        return Z_full, X_full

    def get_detection_event_stats(self):
        total_slots = max(1, self.n_iter * (self.n_round + 1) * 2 * self.n_node_2d)

        precision = self.det_node_tp / max(self.det_node_obs_total, 1)
        recall = self.det_node_tp / max(self.det_node_ideal_total, 1)
        f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall / (precision + recall))

        return {
            "avg_det_mismatch": self.det_node_mismatch / self.n_iter,
            "avg_det_fn": self.det_node_fn / self.n_iter,
            "avg_det_fp": self.det_node_fp / self.n_iter,
            "det_mismatch_rate": self.det_node_mismatch / total_slots,
            "det_precision": precision,
            "det_recall": recall,
            "det_f1": f1,
        }

    def measure(self, t: int, faulty: bool):
        if faulty:
            self._apply_idle_noise(data=True)
            self._apply_GKP_noise(GKP=True)

        if self.clean:
            self._GKP_measure(step=0, faulty=False)
        else:
            self._GKP_measure(step=0, faulty=faulty)

        self._cal_space_weight(step=0, t=t, with_info=self.with_info)
        self._GKP_correction(step=0)

        self.synd_q = np.zeros((self.n_data_qubit))
        self.synd_p = np.zeros((self.n_data_qubit))
        if faulty:
            self._apply_idle_noise(data=True)
            self._apply_GKP_noise(GKP=True)

        if self.clean:
            self._GKP_measure(step=1, faulty=False)
        else:
            self._GKP_measure(step=1, faulty=faulty)

        self._cal_space_weight(step=1, t=t, with_info=self.with_info)
        self._GKP_correction(step=1)

        if faulty:
            self._apply_idle_noise(data=True)
            self._apply_GKP_noise(surface=True)

        self._surface_measure(step=0, faulty=faulty)
        self._surface_measure(step=1, faulty=faulty)
        self._surface_measure(step=2, faulty=faulty)
        self._surface_measure(step=3, faulty=faulty)

        if faulty:
            self._apply_idle_noise(data=True)
            self._apply_circuit_noise(surface=True)

        self._cal_surface_syndrome()

        if faulty and t < self.n_round:
            self._cal_time_weight(with_info=self.with_info)

        n_measure = 0
        if t < self.n_round:
            n_measure = self.n_syndrome_2d * 2

        if self.verbose:
            print("================== Round", t, "=================")
            print("Selected Z syndromes:", self.select_Z_synd_list)
            print("Selected X syndromes:", self.select_X_synd_list)
            print()

            print("Surface code ancilla values (Z_q and X_p):")
            print(self.Z_synd_q)
            print(self.X_synd_p)
            print()

            print("Space Weights and Time Weights (Z and X):")
            print(self.Z_space_weight)
            print(self.X_space_weight)
            print(self.Z_time_weight)
            print(self.X_time_weight)
            print()
            print()

        return n_measure

    def get_weight(self):
        return self.Z_space_weight, self.X_space_weight, self.Z_time_weight, self.X_time_weight

    def run(self, verbose, save_fig, log):
        self.n_measure, self.n_Z_err, self.n_X_err, self.n_Y_err = 0, 0, 0, 0

        if self.check_counter:
            self.n_decoding_node = 0
            self.n_idle_decoding_node = 0
            self.n_idle_decoding_diff = 0
            self.miss_count = 0
            self.wrong_count = 0

            self.det_node_fn = 0
            self.det_node_fp = 0
            self.det_node_tp = 0
            self.det_node_mismatch = 0
            self.det_node_ideal_total = 0
            self.det_node_obs_total = 0

        for iter in range(self.n_iter):
            n_Z_err, n_X_err, n_Y_err = 0, 0, 0
            self._init_iter()
            for t in range(self.n_round + 1):
                self._init_round()
                if t == self.n_round:
                    n_measure = self.measure(t, faulty=self.final_noise)
                else:
                    n_measure = self.measure(t, faulty=True)

                Z_space_weight, X_space_weight, Z_time_weight, X_time_weight = self.get_weight()

                self.decoder.update_weight(Z_space_weight, X_space_weight, Z_time_weight, X_time_weight, t)

                Z_check, X_check = np.sum(self.Z_synd) % 2, np.sum(self.X_synd) % 2
                Z_virtual_idx, X_virtual_idx = self._find_nearest_virtual(Z_check, X_check)
                if Z_virtual_idx != 0:
                    self.Z_synd[Z_virtual_idx] = 1
                if X_virtual_idx != 0:
                    self.X_synd[X_virtual_idx] = 1

                Z_diff = (self.Z_synd + self.prev_Z_synd) % 2
                X_diff = (self.X_synd + self.prev_X_synd) % 2
                self.n_z_check += int(np.sum(Z_diff))
                self.n_x_check += int(np.sum(X_diff))

                if self.verbose:
                    print(f"Round {t}: Z diff: {Z_diff}, X diff: {X_diff}")
                self.decoder.update_syndrome(Z_diff, X_diff, t)

                self.prev_Z_synd, self.prev_X_synd = self.Z_synd, self.X_synd

                self.n_measure += n_measure

            self.decoder.make_decoding_graph(self.n_z_check, self.n_x_check)

            x_correction, z_correction = self.decoder.decode(verbose)
            x_corr, z_corr = self._surface_correction(x_correction, z_correction)

            x_int = np.rint(x_corr).astype(int)
            z_int = np.rint(z_corr).astype(int)

            total_x = int(np.sum(x_int) % 2)
            total_z = int(np.sum(z_int) % 2)

            if total_x == 1 and total_z == 0:
                n_X_err += 1
            elif total_x == 0 and total_z == 1:
                n_Z_err += 1
            elif total_x == 1 and total_z == 1:
                n_Y_err += 1

            self.n_X_err += n_X_err
            self.n_Z_err += n_Z_err
            self.n_Y_err += n_Y_err

            if self.verbose:
                print(f'in {iter} iteration, X error: {n_X_err}, Z error: {n_Z_err}, Y error: {n_Y_err}')

        X_err_rate = self.n_X_err / self.n_iter
        Z_err_rate = self.n_Z_err / self.n_iter
        Y_err_rate = self.n_Y_err / self.n_iter
        avg_n_measure = self.n_measure / self.n_iter

        avg_n_decoding_node, avg_n_decoding_node_diff, avg_missing, avg_wrong = 0, 0, 0, 0

        return X_err_rate, Z_err_rate, Y_err_rate, avg_n_measure, avg_n_decoding_node, avg_n_decoding_node_diff, avg_missing, avg_wrong
