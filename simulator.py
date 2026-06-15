import numpy as np

from noise import SurfaceGKPNoiseTrackingModel
from simulator_base import SurfaceGKPSimulator
from utils import *

EPS = 1e-300


class SurfaceGKPWeightTrackingSimulator(SurfaceGKPSimulator):
    def __init__(
        self,
        d: int,
        n_round: int,
        n_iter: int,
        sigma,
        sigma_GKP,
        sigma_idle,
        with_info,
        clean=False,
        fix_seed=False,
        verbose=False,
        check_counter=False,
        final_noise=False,
        is_skip=False
    ) -> None:
        super().__init__(
            d,
            n_round,
            n_iter,
            sigma,
            sigma_GKP,
            sigma_idle,
            with_info,
            clean,
            fix_seed,
            verbose,
            check_counter,
            final_noise,
        )
        self.noise = SurfaceGKPNoiseTrackingModel(d, n_round, sigma, sigma_GKP, sigma_idle)
        self.is_skip = is_skip
        self.D, self.O1, self.O2, self.O3, self.O4 = self.noise.get_error_matrix()

    def _init_iter(self):
        """Reset trajectory-level closed-form state once per Monte-Carlo iteration.

        _Dq_var/_Dp_var must persist across rounds: GKP weights in round t
        depend on the surface action committed at the end of round t-1.
        """
        super()._init_iter()
        try:
            self.noise.reset_round(reset_data=True)
        except TypeError:
            self.noise.reset_round()


    def _init_round(self):
        self.synd_q = np.zeros(self.n_data_qubit)
        self.synd_p = np.zeros(self.n_data_qubit)
        self.Z_synd_q = np.zeros(self.n_syndrome_2d)
        self.Z_synd_p = np.zeros(self.n_syndrome_2d)
        self.X_synd_q = np.zeros(self.n_syndrome_2d)
        self.X_synd_p = np.zeros(self.n_syndrome_2d)

        self.Z_space_weight = -np.ones(self.n_space_edge_2d)
        self.X_space_weight = -np.ones(self.n_space_edge_2d)
        self.Z_time_weight = -np.ones(self.n_node_2d)
        self.X_time_weight = -np.ones(self.n_node_2d)

        self.Z_synd = np.zeros(self.n_node_2d)
        self.X_synd = np.zeros(self.n_node_2d)

        try:
            self.noise.reset_round(reset_data=False)
        except TypeError:
            self.noise.reset_round()

    def _surface_selection_masks_or_full(self):
        z_mask = getattr(self, "select_Z_mask", None)
        x_mask = getattr(self, "select_X_mask", None)
        if z_mask is None:
            z_mask = np.ones(self.n_syndrome_2d, dtype=bool)
        else:
            z_mask = np.asarray(z_mask, dtype=bool)
        if x_mask is None:
            x_mask = np.ones(self.n_syndrome_2d, dtype=bool)
        else:
            x_mask = np.asarray(x_mask, dtype=bool)
        return z_mask, x_mask

    def _add_independent_noise_to_mask(self, arr: np.ndarray, mask: np.ndarray, variance: float) -> None:
        mask = np.asarray(mask, dtype=bool)
        n = int(np.count_nonzero(mask))
        if n <= 0 or variance == 0.0:
            return
        I = np.eye(n)
        arr[mask] += self._rand_Gvec(variance * I, I)

    def _apply_gkp_readout_noise(self, step: int) -> None:
        """GKP readout slot under the operation-local convention.

        Only the actually measured GKP-ancilla quadrature receives circuit/readout
        noise.  Data qubits and unmeasured GKP-ancilla quadratures receive idle
        noise while the readout is performed.
        """
        self._apply_idle_noise(data=True)

        idx = np.arange(self.n_data_qubit, dtype=int)
        if step == 0:
            q_meas = (idx % 2) == 0
            p_meas = (idx % 2) == 1
        elif step == 1:
            q_meas = (idx % 2) == 1
            p_meas = (idx % 2) == 0
        else:
            raise ValueError("Step index allowed only in [0-1]")

        circ_var = self.sigma * self.sigma
        idle_var = self.sigma_idle * self.sigma_idle
        self._add_independent_noise_to_mask(self.synd_q, q_meas, circ_var)
        self._add_independent_noise_to_mask(self.synd_q, ~q_meas, idle_var)
        self._add_independent_noise_to_mask(self.synd_p, p_meas, circ_var)
        self._add_independent_noise_to_mask(self.synd_p, ~p_meas, idle_var)

    def _apply_surface_prep_noise(self) -> None:
        """Surface-ancilla preparation slot under the operation-local convention."""
        z_mask, x_mask = self._surface_selection_masks_or_full()
        idle_var = self.sigma_idle * self.sigma_idle
        gkp_var = self.sigma_GKP * self.sigma_GKP

        # Data qubits wait while selected surface ancillas are prepared.
        self._apply_idle_noise(data=True)

        # Selected surface ancillas are prepared as GKP states; skipped ancillas idle.
        for arr in (self.Z_synd_q, self.Z_synd_p):
            self._add_independent_noise_to_mask(arr, z_mask, gkp_var)
            self._add_independent_noise_to_mask(arr, ~z_mask, idle_var)
        for arr in (self.X_synd_q, self.X_synd_p):
            self._add_independent_noise_to_mask(arr, x_mask, gkp_var)
            self._add_independent_noise_to_mask(arr, ~x_mask, idle_var)

    def _apply_surface_readout_noise(self) -> None:
        """Surface readout slot under the operation-local convention.

        Z-type ancillas are read out in q, and X-type ancillas are read out in p.
        Those measured quadratures receive circuit/readout noise.  Other modes,
        including all data qubits, receive idle noise.
        """
        z_mask, x_mask = self._surface_selection_masks_or_full()
        idle_var = self.sigma_idle * self.sigma_idle
        circ_var = self.sigma * self.sigma

        self._apply_idle_noise(data=True)

        self._add_independent_noise_to_mask(self.Z_synd_q, z_mask, circ_var)
        self._add_independent_noise_to_mask(self.Z_synd_q, ~z_mask, idle_var)
        self._add_independent_noise_to_mask(self.Z_synd_p, np.ones(self.n_syndrome_2d, dtype=bool), idle_var)

        self._add_independent_noise_to_mask(self.X_synd_q, np.ones(self.n_syndrome_2d, dtype=bool), idle_var)
        self._add_independent_noise_to_mask(self.X_synd_p, x_mask, circ_var)
        self._add_independent_noise_to_mask(self.X_synd_p, ~x_mask, idle_var)

    def _GKP_measure(self, step: int, faulty: bool):
        """Perform one GKP-stabilizer measurement step.

        Gate noise is the usual correlated circuit noise.  In the readout slot,
        data qubits receive idle noise and only the measured GKP-ancilla mode
        receives circuit/readout noise.
        """
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
                self._apply_gkp_readout_noise(step=0)

            even_q = keep_even(self.synd_q)
            odd_p = keep_odd(self.synd_p)
            self.synd_q = transform_centered_mod(np.sqrt(np.pi), even_q)
            self.synd_p = transform_centered_mod(np.sqrt(np.pi), odd_p)

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
                self._apply_gkp_readout_noise(step=1)

            odd_q = keep_odd(self.synd_q)
            even_p = keep_even(self.synd_p)
            self.synd_q = transform_centered_mod(np.sqrt(np.pi), odd_q)
            self.synd_p = transform_centered_mod(np.sqrt(np.pi), even_p)
        else:
            raise ValueError("Step index allowed only in [0-1]")

    def _surface_measure(self, step: int, faulty: bool):
        """Surface gate slot with idle noise on non-participants."""
        if step not in (0, 1, 2, 3):
            raise ValueError("Step index allowed only in [0-3]")

        z_mask, x_mask = self._surface_selection_masks_or_full()
        masked_Z_mat = self.Z_mat[step] if not faulty else self.Z_mat[step] * z_mask
        masked_X_mat = self.X_mat[step] if not faulty else self.X_mat[step] * x_mask

        if step in (0, 3):
            self.data_q += masked_X_mat @ self.X_synd_q
            self.data_p -= masked_Z_mat @ self.Z_synd_p
            self.Z_synd_q += masked_Z_mat.T @ self.data_q
            self.X_synd_p -= masked_X_mat.T @ self.data_p
        else:
            self.data_q -= masked_X_mat @ self.X_synd_q
            self.data_p -= masked_Z_mat @ self.Z_synd_p
            self.Z_synd_q += masked_Z_mat.T @ self.data_q
            self.X_synd_p += masked_X_mat.T @ self.data_p

        if faulty:
            for i in range(self.n_syndrome_2d):
                Z_access_idx = find_one(masked_Z_mat[:, i])
                if Z_access_idx != -1:
                    rand_O1 = self._rand_Gvec(self.D, self.O1)
                    rand_O2 = self._rand_Gvec(self.D, self.O2)
                    self.data_q[Z_access_idx] += rand_O1[0]
                    self.Z_synd_q[i] += rand_O1[1]
                    self.data_p[Z_access_idx] += rand_O2[0]
                    self.Z_synd_p[i] += rand_O2[1]
                else:
                    rand_G = self._rand_Gvec(self.cov_idle, self.I)
                    self.Z_synd_q[i] += rand_G[0]
                    self.Z_synd_p[i] += rand_G[1]

                X_access_idx = find_one(masked_X_mat[:, i])
                if X_access_idx != -1:
                    if step in (0, 3):
                        rand_O3 = self._rand_Gvec(self.D, self.O3)
                        rand_O4 = self._rand_Gvec(self.D, self.O4)
                        self.data_q[X_access_idx] += rand_O3[0]
                        self.X_synd_q[i] += rand_O3[1]
                        self.data_p[X_access_idx] += rand_O4[0]
                        self.X_synd_p[i] += rand_O4[1]
                    else:
                        rand_O1 = self._rand_Gvec(self.D, self.O1)
                        rand_O2 = self._rand_Gvec(self.D, self.O2)
                        self.data_q[X_access_idx] += rand_O2[0]
                        self.X_synd_q[i] += rand_O2[1]
                        self.data_p[X_access_idx] += rand_O1[0]
                        self.X_synd_p[i] += rand_O1[1]
                else:
                    rand_G = self._rand_Gvec(self.cov_idle, self.I)
                    self.X_synd_q[i] += rand_G[0]
                    self.X_synd_p[i] += rand_G[1]

            for i in range(self.n_data_qubit):
                if np.sum(masked_Z_mat[i, :] + masked_X_mat[i, :]) == 0:
                    rand_G = self._rand_Gvec(self.cov_idle, self.I)
                    self.data_q[i] += rand_G[0]
                    self.data_p[i] += rand_G[1]


    def measure(self, t: int, faulty: bool):
        # GKP syndrome extraction and error correction
        if faulty:
            self._apply_idle_noise(data=True)
            self._apply_GKP_noise(GKP=True)

        self._GKP_measure(step=0, faulty=False if self.clean else faulty)
        self.noise.update_gkp_stabilizer_noise(step=0, faulty=faulty)
        self._cal_space_weight(step=0, t=t, with_info=self.with_info)
        self._GKP_correction(step=0)

        self.synd_q = np.zeros(self.n_data_qubit)
        self.synd_p = np.zeros(self.n_data_qubit)
        if faulty:
            self._apply_idle_noise(data=True)
            self._apply_GKP_noise(GKP=True)

        self._GKP_measure(step=1, faulty=False if self.clean else faulty)
        self.noise.update_gkp_stabilizer_noise(step=1, faulty=faulty)
        self._cal_space_weight(step=1, t=t, with_info=self.with_info)
        self._GKP_correction(step=1)

        # surface-code syndrome extraction and error correction
        if faulty:
            self._apply_surface_prep_noise()

        self._surface_measure(step=0, faulty=faulty)
        self._surface_measure(step=1, faulty=faulty)
        self._surface_measure(step=2, faulty=faulty)
        self._surface_measure(step=3, faulty=faulty)

        if faulty:
            self._apply_surface_readout_noise()
            
        self.noise.update_surface_stabilizer_noise(
            selected_Z_synd_list=None,
            selected_X_synd_list=None,
            faulty=faulty,
        )
        self._cal_surface_syndrome()

        if faulty and t < self.n_round:
            self._cal_time_weight(with_info=self.with_info)

        if self.verbose:
            print(f"===== round {t} : number of total measurement {self.n_syndrome_2d * 2 if t < self.n_round else 0} =====")
            print()

        return self.n_syndrome_2d * 2 if t < self.n_round else 0

    def _cal_space_weight(self, step: int, t: int, with_info: bool = True):
        if step not in (0, 1):
            raise ValueError("step must be 0 or 1")

        idx = np.arange(self.n_data_qubit, dtype=int)
        idxZ = idx[(idx % 2) == step]
        idxX = idx[(idx % 2) != step]

        sigmaZ_all = self.noise.get_Z_sigma_space_vec(step)
        sigmaX_all = self.noise.get_X_sigma_space_vec(step)

        if self.verbose:
            print("Sigma for gkp code (Z space):", sigmaZ_all**2)
            print("Sigma for gkp code (X space):", sigmaX_all**2)

        if self.is_skip and t != self.n_round:
            blocked_weight = 1e6
            self.Z_space_weight[idxZ] = blocked_weight
            self.X_space_weight[idxX] = blocked_weight
            return 
        
        if with_info:
            self.Z_space_weight[idxZ] = -np.log2(np.maximum(cond_err_prob_vec(sigmaZ_all[idxZ], self.synd_q[idxZ]), EPS))
            self.X_space_weight[idxX] = -np.log2(np.maximum(cond_err_prob_vec(sigmaX_all[idxX], self.synd_p[idxX]), EPS))
        else:
            self.Z_space_weight[idxZ] = -np.log2(np.maximum(err_prob_vec(sigmaZ_all[idxZ]), EPS))
            self.X_space_weight[idxX] = -np.log2(np.maximum(err_prob_vec(sigmaX_all[idxX]), EPS))

    def _cal_time_weight(self, with_info: bool = True):
        sigmaZt = self.noise.get_Z_sigma_time_vec()
        sigmaXt = self.noise.get_X_sigma_time_vec()

        if with_info:
            zZ = transform_centered_mod(np.sqrt(np.pi), self.Z_synd_q)
            zX = transform_centered_mod(np.sqrt(np.pi), self.X_synd_p)
            wZ = -np.log2(np.maximum(cond_err_prob_vec(sigmaZt, zZ), EPS))
            wX = -np.log2(np.maximum(cond_err_prob_vec(sigmaXt, zX), EPS))
        else:
            wZ = -np.log2(np.maximum(err_prob_vec(sigmaZt), EPS))
            wX = -np.log2(np.maximum(err_prob_vec(sigmaXt), EPS))

        self.Z_time_weight[: self.n_syndrome_2d] = wZ
        self.X_time_weight[: self.n_syndrome_2d] = wX


class SurfaceGKPPartialMeasureWeightTrackingSimulator(SurfaceGKPWeightTrackingSimulator):
    def __init__(
        self,
        d: int,
        n_round: int,
        n_iter: int,
        sigma,
        sigma_GKP,
        sigma_idle,
        with_info,
        clean=False,
        fix_seed=False,
        verbose=False,
        check_counter=False,
        final_noise=False,
        is_skip=False,
    ) -> None:
        super().__init__(
            d,
            n_round,
            n_iter,
            sigma,
            sigma_GKP,
            sigma_idle,
            with_info,
            clean,
            fix_seed,
            verbose,
            check_counter,
            final_noise,
            is_skip
        )
        self._all_syndromes = np.arange(self.n_syndrome_2d, dtype=int)
        self._reset_selection(full=True)

    def _init_round(self):
        super()._init_round()
        self._reset_selection(full=True)

    def _reset_selection(self, full: bool):
        if full:
            self.select_Z_synd_list = self._all_syndromes.copy()
            self.select_X_synd_list = self._all_syndromes.copy()
        else:
            self.select_Z_synd_list = np.array([], dtype=int)
            self.select_X_synd_list = np.array([], dtype=int)
            
        self._make_select_mask()

    def _select_measure(self, t):
        raise NotImplementedError

    def _make_select_mask(self):
        self.select_Z_mask = make_mask(self.select_Z_synd_list, self.n_syndrome_2d)
        self.select_X_mask = make_mask(self.select_X_synd_list, self.n_syndrome_2d)

    @staticmethod
    def _clip_prob(p, eps: float = 1e-12):
        return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)

    def _surface_measure(self, step: int, faulty: bool):
        if step not in (0, 1, 2, 3):
            raise ValueError("Step index allowed only in [0-3]")

        masked_Z_mat = self.Z_mat[step] if not faulty else self.Z_mat[step] * self.select_Z_mask
        masked_X_mat = self.X_mat[step] if not faulty else self.X_mat[step] * self.select_X_mask

        if step in (0, 3):
            self.data_q += masked_X_mat @ self.X_synd_q
            self.data_p -= masked_Z_mat @ self.Z_synd_p
            self.Z_synd_q += masked_Z_mat.T @ self.data_q
            self.X_synd_p -= masked_X_mat.T @ self.data_p
        else:
            self.data_q -= masked_X_mat @ self.X_synd_q
            self.data_p -= masked_Z_mat @ self.Z_synd_p
            self.Z_synd_q += masked_Z_mat.T @ self.data_q
            self.X_synd_p += masked_X_mat.T @ self.data_p

        if faulty:
            for i in range(self.n_syndrome_2d):
                Z_access_idx = find_one(masked_Z_mat[:, i])
                if Z_access_idx != -1:
                    rand_O1 = self._rand_Gvec(self.D, self.O1)
                    rand_O2 = self._rand_Gvec(self.D, self.O2)
                    self.data_q[Z_access_idx] += rand_O1[0]
                    self.Z_synd_q[i] += rand_O1[1]
                    self.data_p[Z_access_idx] += rand_O2[0]
                    self.Z_synd_p[i] += rand_O2[1]
                else:
                    rand_G = self._rand_Gvec(self.cov_idle, self.I)
                    self.Z_synd_q[i] += rand_G[0]
                    self.Z_synd_p[i] += rand_G[1]

                X_access_idx = find_one(masked_X_mat[:, i])
                if X_access_idx != -1:
                    if step in (0, 3):
                        rand_O3 = self._rand_Gvec(self.D, self.O3)
                        rand_O4 = self._rand_Gvec(self.D, self.O4)
                        self.data_q[X_access_idx] += rand_O3[0]
                        self.X_synd_q[i] += rand_O3[1]
                        self.data_p[X_access_idx] += rand_O4[0]
                        self.X_synd_p[i] += rand_O4[1]
                    else:
                        rand_O1 = self._rand_Gvec(self.D, self.O1)
                        rand_O2 = self._rand_Gvec(self.D, self.O2)
                        self.data_q[X_access_idx] += rand_O2[0]
                        self.X_synd_q[i] += rand_O2[1]
                        self.data_p[X_access_idx] += rand_O1[0]
                        self.X_synd_p[i] += rand_O1[1]
                else:
                    rand_G = self._rand_Gvec(self.cov_idle, self.I)
                    self.X_synd_q[i] += rand_G[0]
                    self.X_synd_p[i] += rand_G[1]

            for i in range(self.n_data_qubit):
                if np.sum(masked_Z_mat[i, :] + masked_X_mat[i, :]) == 0:
                    rand_G = self._rand_Gvec(self.cov_idle, self.I)
                    self.data_q[i] += rand_G[0]
                    self.data_p[i] += rand_G[1]

    def _cal_surface_syndrome(self):
        self.Z_synd[: self.n_syndrome_2d] = self.prev_Z_synd[: self.n_syndrome_2d]
        self.X_synd[: self.n_syndrome_2d] = self.prev_X_synd[: self.n_syndrome_2d]

        if np.any(self.select_Z_mask):
            self.Z_synd_q[self.select_Z_mask] = transform_centered_mod(2 * np.sqrt(np.pi), self.Z_synd_q[self.select_Z_mask])
            z_values = (np.abs(self.Z_synd_q[self.select_Z_mask]) > np.sqrt(np.pi) / 2).astype(float)
            self.Z_synd[np.flatnonzero(self.select_Z_mask)] = z_values

        if np.any(self.select_X_mask):
            self.X_synd_p[self.select_X_mask] = transform_centered_mod(2 * np.sqrt(np.pi), self.X_synd_p[self.select_X_mask])
            x_values = (np.abs(self.X_synd_p[self.select_X_mask]) > np.sqrt(np.pi) / 2).astype(float)
            self.X_synd[np.flatnonzero(self.select_X_mask)] = x_values

    def _cal_time_weight(self, with_info: bool = True):
        sigmaZt = self.noise.get_Z_sigma_time_vec()
        sigmaXt = self.noise.get_X_sigma_time_vec()

        if self.verbose:
            print("Sigma for surface code (Z time):", sigmaZt**2)
            print("Sigma for surface code (X time):", sigmaXt**2)

        if with_info:
            zZ = transform_centered_mod(np.sqrt(np.pi), self.Z_synd_q)
            zX = transform_centered_mod(np.sqrt(np.pi), self.X_synd_p)
            wZ = -np.log2(np.maximum(cond_err_prob_vec(sigmaZt, zZ), EPS))
            wX = -np.log2(np.maximum(cond_err_prob_vec(sigmaXt, zX), EPS))
        else:
            wZ = -np.log2(np.maximum(err_prob_vec(sigmaZt), EPS))
            wX = -np.log2(np.maximum(err_prob_vec(sigmaXt), EPS))

        wZ[~self.select_Z_mask] = 0.0
        wX[~self.select_X_mask] = 0.0

        self.Z_time_weight[: self.n_syndrome_2d] = wZ
        self.X_time_weight[: self.n_syndrome_2d] = wX

    def measure(self, t: int, faulty: bool):
        # GKP syndrome extraction and error correction
        if faulty:
            self._apply_idle_noise(data=True)
            self._apply_GKP_noise(GKP=True)

        self._GKP_measure(step=0, faulty=False if self.clean else faulty)
        self.noise.update_gkp_stabilizer_noise(step=0, faulty=faulty)
        self._cal_space_weight(step=0, t=t, with_info=self.with_info)
        self._GKP_correction(step=0)

        self.synd_q = np.zeros(self.n_data_qubit)
        self.synd_p = np.zeros(self.n_data_qubit)
        if faulty:
            self._apply_idle_noise(data=True)
            self._apply_GKP_noise(GKP=True)

        self._GKP_measure(step=1, faulty=False if self.clean else faulty)
        self.noise.update_gkp_stabilizer_noise(step=1, faulty=faulty)
        self._cal_space_weight(step=1, t=t, with_info=self.with_info)
        self._GKP_correction(step=1)

        # surface-code syndrome extraction and error correction
        if faulty:
            if t == self.n_round:
                self._reset_selection(full=True)
            else:
                self._select_measure(t)
            self._make_select_mask()
            self._apply_surface_prep_noise()
        else:
            self._reset_selection(full=True)

        # if self.verbose:
        #     print("self.data_p, self.data_q (before measurement)", self.data_p, self.data_q)

        self._surface_measure(step=0, faulty=faulty)
        self._surface_measure(step=1, faulty=faulty)
        self._surface_measure(step=2, faulty=faulty)
        self._surface_measure(step=3, faulty=faulty)

        if faulty:
            self._apply_surface_readout_noise()

        self.noise.update_surface_stabilizer_noise(
            self.select_Z_synd_list,
            self.select_X_synd_list,
            faulty=faulty,
        )

        # if self.verbose:
        #     print("self.data_p, self.data_q (after measurement)", self.data_p, self.data_q)

        self._cal_surface_syndrome()

        if faulty and t < self.n_round:
            self._cal_time_weight(with_info=self.with_info)

        n_measure = 0
        if t < self.n_round:
            n_measure = int(self.select_Z_mask.sum() + self.select_X_mask.sum())

        if self.verbose:
            print(f"===== round {t} : number of total measurement {n_measure} =====")
            print()

        return n_measure


class SurfaceGKPInfoMeasureWeightTrackingSimulator(SurfaceGKPPartialMeasureWeightTrackingSimulator):
    SUPPORTED_SCORE_METRICS = {"global_joint"}

    def __init__(
        self,
        d: int,
        n_round: int,
        n_iter: int,
        sigma,
        sigma_GKP,
        sigma_idle,
        with_info,
        clean=False,
        fix_seed=False,
        verbose=False,
        check_counter=False,
        final_noise=False,
        score_metric="global_joint",
        stab_error_prob="parity",
        is_skip=False,
    ) -> None:
        super().__init__(
            d,
            n_round,
            n_iter,
            sigma,
            sigma_GKP,
            sigma_idle,
            with_info,
            clean,
            fix_seed,
            verbose,
            check_counter,
            final_noise,
            is_skip,
        )
        if score_metric not in self.SUPPORTED_SCORE_METRICS:
            raise ValueError(f"Unknown score_metric: {score_metric}")
        if stab_error_prob not in {"parity", "one"}:
            raise ValueError(f"Unknown stab_error_prob: {stab_error_prob}")

        self.score_metric = score_metric
        self.stab_error_prob = stab_error_prob

    def _set_all_selection(self):
        self.select_Z_synd_list = self._all_syndromes.copy()
        self.select_X_synd_list = self._all_syndromes.copy()

    def _set_no_selection(self):
        self.select_Z_synd_list = np.array([], dtype=int)
        self.select_X_synd_list = np.array([], dtype=int)

    def _logical_error_probabilities(self):
        pZ = np.clip(2.0 ** (-self.Z_space_weight[: self.n_data_qubit]), 0.0, 1.0)
        pX = np.clip(2.0 ** (-self.X_space_weight[: self.n_data_qubit]), 0.0, 1.0)
        return pZ, pX

    def _syndrome_flip_probabilities(self):
        pZ, pX = self._logical_error_probabilities()
        Z_mat = sum(self.Z_mat).T
        X_mat = sum(self.X_mat).T

        if self.stab_error_prob == "parity":
            pZ_flip = 0.5 * (1.0 - np.prod(np.where(Z_mat, 1.0 - 2.0 * pZ, 1.0), axis=1))
            pX_flip = 0.5 * (1.0 - np.prod(np.where(X_mat, 1.0 - 2.0 * pX, 1.0), axis=1))
        elif self.stab_error_prob == "one":
            pZ_flip = 1.0 - np.prod(np.where(Z_mat, 1.0 - pZ, 1.0), axis=1)
            pX_flip = 1.0 - np.prod(np.where(X_mat, 1.0 - pX, 1.0), axis=1)
        else:
            raise ValueError(f"Unknown stab_error_prob: {self.stab_error_prob}")

        return np.clip(pZ_flip, 0.0, 1.0), np.clip(pX_flip, 0.0, 1.0)

    def _measurement_success_for_selection(self, selected_Z, selected_X):
        sigmaZ, sigmaX = self.noise.get_surface_time_sigma_vec_for_selection(selected_Z, selected_X)
        pZ_meas = 1.0 - err_prob_vec(sigmaZ)
        pX_meas = 1.0 - err_prob_vec(sigmaX)
        return self._clip_prob(pZ_meas), self._clip_prob(pX_meas)

    def _sum_log_prob(self, prob) -> float:
        return float(np.sum(np.log(self._clip_prob(prob))))

    def _apply_global_joint_policy(self, policy: str) -> None:
        empty = np.array([], dtype=int)

        if policy == "none":
            self._set_no_selection()
            return
        if policy == "z_only":
            self.select_Z_synd_list = self._all_syndromes.copy()
            self.select_X_synd_list = empty
            return
        if policy == "x_only":
            self.select_Z_synd_list = empty
            self.select_X_synd_list = self._all_syndromes.copy()
            return
        if policy == "both":
            self._set_all_selection()
            return

        raise ValueError(f"Unknown global_joint policy: {policy}")

    def _select_by_global_joint(self, skip_success_Z, skip_success_X) -> None:
        empty = np.array([], dtype=int)
        all_Z = self._all_syndromes
        all_X = self._all_syndromes

        pZ_z_only, _ = self._measurement_success_for_selection(all_Z, empty)
        _, pX_x_only = self._measurement_success_for_selection(empty, all_X)
        pZ_both, pX_both = self._measurement_success_for_selection(all_Z, all_X)

        scores = {
            "none": self._sum_log_prob(skip_success_Z) + self._sum_log_prob(skip_success_X),
            "z_only": self._sum_log_prob(pZ_z_only) + self._sum_log_prob(skip_success_X),
            "x_only": self._sum_log_prob(skip_success_Z) + self._sum_log_prob(pX_x_only),
            "both": self._sum_log_prob(pZ_both) + self._sum_log_prob(pX_both),
        }
        best_policy = max(scores, key=scores.get)
        self._last_global_joint_scores = scores
        self._last_global_joint_choice = best_policy
        self._apply_global_joint_policy(best_policy)

    def _select_measure(self, t):
        flip_prob_Z, flip_prob_X = self._syndrome_flip_probabilities()
        skip_success_Z = self._clip_prob(1.0 - flip_prob_Z)
        skip_success_X = self._clip_prob(1.0 - flip_prob_X)

        self._select_by_global_joint(skip_success_Z, skip_success_X)

        if self.verbose:
            print("skip_success_Z:", skip_success_Z)
            print("skip_success_X:", skip_success_X)
            print("global_joint_scores:", getattr(self, "_last_global_joint_scores", None))
            print("global_joint_choice:", getattr(self, "_last_global_joint_choice", None))
            print("selected Z:", self.select_Z_synd_list)
            print("selected X:", self.select_X_synd_list)


class SurfaceGKPInfoMeasureWeightTrackingMeasureSimulator(SurfaceGKPInfoMeasureWeightTrackingSimulator):
    def __init__(
        self,
        d: int,
        n_round: int,
        n_iter: int,
        sigma,
        sigma_GKP,
        sigma_idle,
        with_info,
        clean=False,
        fix_seed=False,
        verbose=False,
        check_counter=False,
        final_noise=False,
        score_metric="global_joint",
        stab_error_prob="parity",
        target_rounds=None,
        skip_rounds=None,
        is_skip=False,
    ) -> None:
        super().__init__(
            d,
            n_round,
            n_iter,
            sigma,
            sigma_GKP,
            sigma_idle,
            with_info,
            clean,
            fix_seed,
            verbose,
            check_counter,
            final_noise,
            score_metric,
            stab_error_prob,
            is_skip,
        )
        self.force_measure_rounds = self._normalize_rounds(target_rounds)
        self.force_skip_rounds = self._normalize_rounds(skip_rounds)
        overlap = self.force_measure_rounds & self.force_skip_rounds
        if overlap:
            raise ValueError(f"Round(s) present in both target_rounds and skip_rounds: {sorted(overlap)}")

    def _normalize_rounds(self, rounds):
        if rounds is None:
            return set()

        normalized = set()
        for round_idx in rounds:
            idx = int(round_idx)
            if idx < 0 or idx > self.n_round:
                raise ValueError(f"Round index out of range: {idx}. Allowed range is [0, {self.n_round}].")
            normalized.add(idx)
        return normalized

    def _select_measure(self, t):
        super()._select_measure(t)

        if t in self.force_measure_rounds:
            self._set_all_selection()
            return
        if t in self.force_skip_rounds:
            self._set_no_selection()
            return


class SurfaceGKPDefaultTrackingSimulator(SurfaceGKPPartialMeasureWeightTrackingSimulator):
    def _select_measure(self, t):
        self._reset_selection(full=True)