import numpy as np

from abc import ABC
from typing import Optional, Sequence


class NoiseModel(ABC):
    def __init__(self, d: int):
        self.d = int(d)


class SurfaceGKPNoiseModel(NoiseModel):
    def __init__(self, d: int, n_round: int, sigma: float, sigma_GKP: float, sigma_idle: float):
        super().__init__(d)
        self.n_round = int(n_round)
        self.sigma = float(sigma)
        self.sigma_GKP = float(sigma_GKP)
        self.sigma_idle = float(sigma_idle)
        self._make_error_matrix()

    def _make_error_matrix(self) -> None:
        s2 = self.sigma * self.sigma
        rt = np.sqrt
        self.D = np.array(
            [
                [((7.0 + np.sqrt(10.0)) / 6.0) * s2, 0.0],
                [0.0, ((7.0 - np.sqrt(10.0)) / 6.0) * s2],
            ],
            dtype=np.float64,
        )
        self.O1 = np.array(
            [
                [0.5 * rt(2.0 - np.sqrt(2.0 / 5.0)), -0.5 * rt(2.0 + np.sqrt(2.0 / 5.0))],
                [0.5 * rt(2.0 + np.sqrt(2.0 / 5.0)), 3.0 / rt(2.0 * (10.0 + np.sqrt(10.0)))],
            ],
            dtype=np.float64,
        )
        self.O2 = np.array(
            [
                [-0.5 * rt(2.0 + np.sqrt(2.0 / 5.0)), 0.5 * rt(2.0 - np.sqrt(2.0 / 5.0))],
                [3.0 / rt(2.0 * (10.0 + np.sqrt(10.0))), 0.5 * rt(2.0 + np.sqrt(2.0 / 5.0))],
            ],
            dtype=np.float64,
        )
        self.O3 = np.array(
            [
                [rt(2.0 + np.sqrt(2.0 / 5.0)) / 2.0, -3.0 / rt(2.0 * (10.0 + np.sqrt(10.0)))],
                [3.0 / rt(2.0 * (10.0 + np.sqrt(10.0))), rt(2.0 + np.sqrt(2.0 / 5.0)) / 2.0],
            ],
            dtype=np.float64,
        )
        self.O4 = np.array(
            [
                [-3.0 / rt(2.0 * (10.0 + np.sqrt(10.0))), rt(2.0 + np.sqrt(2.0 / 5.0)) / 2.0],
                [rt(2.0 + np.sqrt(2.0 / 5.0)) / 2.0, 3.0 / rt(2.0 * (10.0 + np.sqrt(10.0)))],
            ],
            dtype=np.float64,
        )

    def get_error_matrix(self):
        return self.D, self.O1, self.O2, self.O3, self.O4

    # These scalar helpers are kept for compatibility with non-tracking code paths.
    # They correspond to the sigma_idle=sigma limit of the original paper formulas.
    def get_Z_sigma_space(self, i: int, t: int, step: int) -> float:
        def sigmaZS(j: int) -> float:
            if j % self.d == 1:
                if int((j - 1) / self.d) % 2 == 0:
                    return np.sqrt(4.0 * self.sigma_GKP**2 + (52.0 / 3.0) * self.sigma**2)
                return np.sqrt(4.0 * self.sigma_GKP**2 + (58.0 / 3.0) * self.sigma**2)
            if j % self.d == 0:
                if int(j / self.d) % 2 == 0:
                    return np.sqrt(4.0 * self.sigma_GKP**2 + (49.0 / 3.0) * self.sigma**2)
                return np.sqrt(4.0 * self.sigma_GKP**2 + (55.0 / 3.0) * self.sigma**2)
            return np.sqrt(5.0 * self.sigma_GKP**2 + (59.0 / 3.0) * self.sigma**2)

        if t == 0:
            if step == 0:
                return np.sqrt(self.sigma_GKP**2 + (10.0 / 3.0) * self.sigma**2)
            if step == 1:
                return np.sqrt(2.0 * self.sigma_GKP**2 + (20.0 / 3.0) * self.sigma**2)
        elif t == self.n_round:
            if step == 0:
                return np.sqrt(sigmaZS(i + 1) ** 2 - self.sigma_GKP**2 - (10.0 / 3.0) * self.sigma**2)
            if step == 1:
                return np.sqrt(sigmaZS(i + 1) ** 2 - 2.0 * self.sigma_GKP**2 - (20.0 / 3.0) * self.sigma**2)
        return sigmaZS(i + 1)

    def get_X_sigma_space(self, i: int, t: int, step: int) -> float:
        def sigmaXS(j: int) -> float:
            if 1 <= j <= self.d:
                if j % 2 == 0:
                    return np.sqrt(4.0 * self.sigma_GKP**2 + (55.0 / 3.0) * self.sigma**2)
                return np.sqrt(4.0 * self.sigma_GKP**2 + (49.0 / 3.0) * self.sigma**2)
            if self.d**2 - self.d + 1 <= j <= self.d**2:
                if j % 2 == 0:
                    return np.sqrt(4.0 * self.sigma_GKP**2 + (52.0 / 3.0) * self.sigma**2)
                return np.sqrt(4.0 * self.sigma_GKP**2 + (58.0 / 3.0) * self.sigma**2)
            return np.sqrt(5.0 * self.sigma_GKP**2 + (59.0 / 3.0) * self.sigma**2)

        if t == 0:
            if step == 0:
                return np.sqrt(self.sigma_GKP**2 + (10.0 / 3.0) * self.sigma**2)
            if step == 1:
                return np.sqrt(2.0 * self.sigma_GKP**2 + (20.0 / 3.0) * self.sigma**2)
        elif t == self.n_round:
            if step == 0:
                return np.sqrt(sigmaXS(i + 1) ** 2 - self.sigma_GKP**2 - (10.0 / 3.0) * self.sigma**2)
            if step == 1:
                return np.sqrt(sigmaXS(i + 1) ** 2 - 2.0 * self.sigma_GKP**2 - (20.0 / 3.0) * self.sigma**2)
        return sigmaXS(i + 1)

    def get_Z_sigma_time(self, i: int) -> float:
        return float(self.get_Z_sigma_time_vec()[i])

    def get_X_sigma_time(self, i: int) -> float:
        return float(self.get_X_sigma_time_vec()[i])


class SurfaceGKPNoiseTrackingModel(SurfaceGKPNoiseModel):
    ACTIONS = ("both", "none", "z_only", "x_only")

    def __init__(self, d: int, n_round: int, sigma: float, sigma_GKP: float, sigma_idle: float):
        super().__init__(d, n_round, sigma, sigma_GKP, sigma_idle)
        self.n_data = self.d * self.d
        self.n_syn = (self.d * self.d - 1) // 2

        self._Dq_var = np.zeros(self.n_data, dtype=np.float64)
        self._Dp_var = np.zeros(self.n_data, dtype=np.float64)
        self._Aq_var = np.zeros((2, self.n_data), dtype=np.float64)
        self._Ap_var = np.zeros((2, self.n_data), dtype=np.float64)
        self._space_ready = {0: False, 1: False}

        self._Z_time_read_var: Optional[np.ndarray] = None
        self._X_time_read_var: Optional[np.ndarray] = None
        self._time_ready = False

        self._all_syn_idx = np.arange(self.n_syn, dtype=int)
        self._fixed_surface_table = self._build_fixed_surface_table()

    @staticmethod
    def _as_index_array(idx_list: Optional[Sequence[int]], n_syn: int) -> np.ndarray:
        if idx_list is None:
            return np.arange(n_syn, dtype=int)
        return np.asarray(idx_list, dtype=int).reshape(-1)

    def _selection_kind(self, idx_list: Optional[Sequence[int]]) -> Optional[str]:
        arr = self._as_index_array(idx_list, self.n_syn)
        if arr.size == 0:
            return "none"
        if arr.size == self.n_syn and np.array_equal(np.sort(arr), self._all_syn_idx):
            return "all"
        return None

    def global_surface_action_key(
        self,
        selected_Z_synd_list: Optional[Sequence[int]],
        selected_X_synd_list: Optional[Sequence[int]],
    ) -> Optional[str]:
        z_kind = self._selection_kind(selected_Z_synd_list)
        x_kind = self._selection_kind(selected_X_synd_list)
        if z_kind is None or x_kind is None:
            return None
        if z_kind == "all" and x_kind == "all":
            return "both"
        if z_kind == "none" and x_kind == "none":
            return "none"
        if z_kind == "all" and x_kind == "none":
            return "z_only"
        if z_kind == "none" and x_kind == "all":
            return "x_only"
        return None

    def get_surface_action_key(
        self,
        selected_Z_synd_list: Optional[Sequence[int]],
        selected_X_synd_list: Optional[Sequence[int]],
    ) -> str:
        action_key = self.global_surface_action_key(selected_Z_synd_list, selected_X_synd_list)
        if action_key is None:
            raise ValueError("Selection is not a global action: both, none, z_only, x_only.")
        return action_key

    def _xy_data_grid(self) -> tuple[np.ndarray, np.ndarray]:
        y, x = np.indices((self.d, self.d), dtype=int)
        return x, y

    def get_surface_time_sigma_vec_for_selection(
        self,
        selected_Z_synd_list: Optional[Sequence[int]],
        selected_X_synd_list: Optional[Sequence[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        action_key = self.get_surface_action_key(selected_Z_synd_list, selected_X_synd_list)
        return self.get_fixed_surface_time_sigma_vec(action_key)

    def _surface_time_variance_fixed_equation(self, action_key: str) -> tuple[np.ndarray, np.ndarray]:
        if action_key not in self.ACTIONS:
            raise ValueError(f"Unknown global surface action: {action_key!r}")

        g = self.sigma_GKP * self.sigma_GKP
        c = self.sigma * self.sigma
        idle = self.sigma_idle * self.sigma_idle

        idx = np.arange(self.n_syn, dtype=int)
        m = (self.d + 1) // 2
        u = idx % m
        v = idx // m
        v_last = self.d - 2

        Z = np.empty(self.n_syn, dtype=np.float64)
        X = np.empty(self.n_syn, dtype=np.float64)

        def idle_surface(arr: np.ndarray) -> None:
            arr[:] = 6.0 * idle

        def assign(arr: np.ndarray, mask: np.ndarray, A: float, B: float, C: float) -> None:
            arr[mask] = A * g + B * c + C * idle

        if action_key == "none":
            idle_surface(Z)
            idle_surface(X)
            return Z, X

        if action_key == "z_only":
            idle_surface(X)
            Z[:] = 7.0 * g + (61.0 / 3.0) * c + 16.0 * idle
            assign(Z, (u == 0) & ((v % 2) == 0), 4.0, 29.0 / 3.0, 9.0)
            assign(Z, (u == m - 1) & (v == v_last), 4.0, 32.0 / 3.0, 12.0)
            assign(Z, (u == m - 1) & ((v % 2) == 1) & (v != v_last), 4.0, 35.0 / 3.0, 11.0)
            assigned = (
                ((u == 0) & ((v % 2) == 0))
                | ((u == m - 1) & (v == v_last))
                | ((u == m - 1) & ((v % 2) == 1) & (v != v_last))
            )
            assign(Z, ((v == 0) | (v == v_last)) & ~assigned, 7.0, 58.0 / 3.0, 17.0)
            return Z, X

        if action_key == "x_only":
            idle_surface(Z)
            X[:] = 7.0 * g + (61.0 / 3.0) * c + 16.0 * idle
            assign(X, (u == m - 1) & ((v % 2) == 0), 4.0, 29.0 / 3.0, 9.0)
            assign(X, (u == 0) & (v == v_last), 4.0, 32.0 / 3.0, 12.0)
            assign(X, (u == 0) & ((v % 2) == 1) & (v != v_last), 4.0, 35.0 / 3.0, 11.0)
            assigned = (
                ((u == m - 1) & ((v % 2) == 0))
                | ((u == 0) & (v == v_last))
                | ((u == 0) & ((v % 2) == 1) & (v != v_last))
            )
            assign(X, ((v == 0) | (v == v_last)) & ~assigned, 7.0, 58.0 / 3.0, 17.0)
            return Z, X

        # action_key == "both"
        Z[:] = 7.0 * g + (80.0 / 3.0) * c + 12.0 * idle
        assign(Z, (u == 0) & ((v % 2) == 0), 4.0, 29.0 / 3.0, 9.0)
        assign(Z, (u == m - 1) & (v == v_last), 4.0, 43.0 / 3.0, 10.0)
        assign(Z, (u == m - 1) & ((v % 2) == 1) & (v != v_last), 4.0, 46.0 / 3.0, 9.0)
        assign(Z, (u == 0) & (v == v_last), 7.0, 22.0, 15.0)
        assign(Z, (u == 0) & ((v % 2) == 1) & (v != v_last), 7.0, 23.0, 14.0)
        assigned_Z = (
            ((u == 0) & ((v % 2) == 0))
            | ((u == m - 1) & (v == v_last))
            | ((u == m - 1) & ((v % 2) == 1) & (v != v_last))
            | ((u == 0) & (v == v_last))
            | ((u == 0) & ((v % 2) == 1) & (v != v_last))
        )
        assign(Z, ((v == 0) | (v == v_last)) & ~assigned_Z, 7.0, 77.0 / 3.0, 13.0)

        X[:] = 7.0 * g + (80.0 / 3.0) * c + 12.0 * idle
        assign(X, (u == m - 1) & ((v % 2) == 0), 4.0, 29.0 / 3.0, 9.0)
        assign(X, (u == 0) & (v == v_last), 4.0, 43.0 / 3.0, 10.0)
        assign(X, (u == 0) & ((v % 2) == 1) & (v != v_last), 4.0, 46.0 / 3.0, 9.0)
        assign(X, (u == m - 1) & (v == v_last), 7.0, 22.0, 15.0)
        assign(X, (u == m - 1) & ((v % 2) == 1) & (v != v_last), 7.0, 23.0, 14.0)
        assigned_X = (
            ((u == m - 1) & ((v % 2) == 0))
            | ((u == 0) & (v == v_last))
            | ((u == 0) & ((v % 2) == 1) & (v != v_last))
            | ((u == m - 1) & (v == v_last))
            | ((u == m - 1) & ((v % 2) == 1) & (v != v_last))
        )
        assign(X, ((v == 0) | (v == v_last)) & ~assigned_X, 7.0, 77.0 / 3.0, 13.0)
        return Z, X

    def _data_variance_fixed_equation(self, action_key: str) -> tuple[np.ndarray, np.ndarray]:
        if action_key not in self.ACTIONS:
            raise ValueError(f"Unknown global surface action: {action_key!r}")

        g = self.sigma_GKP * self.sigma_GKP
        c = self.sigma * self.sigma
        idle = self.sigma_idle * self.sigma_idle
        x, y = self._xy_data_grid()

        parity_even = ((x + y) % 2) == 0
        odd_x = (x % 2) == 1
        odd_y = (y % 2) == 1
        even_x = ~odd_x
        even_y = ~odd_y
        top = y == 0
        bottom = y == self.d - 1
        left = x == 0
        right = x == self.d - 1
        internal_y = ~(top | bottom)
        internal_x = ~(left | right)

        deg_Z = np.where(top | bottom, 1.0, 2.0)
        deg_X = np.where(left | right, 1.0, 2.0)

        Dq_g = np.where(parity_even, 2.0, 1.0)
        Dp_g = np.where(parity_even, 1.0, 2.0)
        Dq_c3 = np.where(parity_even, 11.0, 7.0)
        Dp_c3 = np.where(parity_even, 7.0, 11.0)
        Dq_i = np.where(parity_even, 9.0, 7.0)
        Dp_i = np.where(parity_even, 7.0, 9.0)

        def add_z_data_side(c3: np.ndarray, i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return c3 + 3.0 * deg_Z, i - deg_Z

        def add_x_data_side(c3: np.ndarray, i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return c3 + 3.0 * deg_X, i - deg_X

        def z_measured_dp_coeffs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            G = np.empty((self.d, self.d), dtype=np.float64)
            C3 = np.empty_like(G)
            I = np.empty_like(G)

            G[top | bottom] = np.where(odd_x[top | bottom], 3.0, 2.0)
            mask = internal_y & odd_y
            G[mask] = np.where(odd_x[mask], 3.0, 4.0)
            mask = internal_y & even_y
            G[mask] = np.where(odd_x[mask], 4.0, 3.0)

            C3[top] = np.where(odd_x[top], 21.0, 11.0)
            mask = internal_y & odd_y
            C3[mask] = np.where(odd_x[mask], 24.0, 28.0)
            C3[mask & right] = 22.0
            mask = internal_y & even_y
            C3[mask] = np.where(odd_x[mask], 28.0, 24.0)
            C3[mask & right] = 18.0
            C3[bottom] = np.where(odd_x[bottom], 18.0, 20.0)
            C3[bottom & right] = 14.0

            I[top] = np.where(odd_x[top], 8.0, 6.0)
            mask = internal_y & odd_y
            I[mask] = np.where(odd_x[mask], 5.0, 7.0)
            I[mask & right] = 9.0
            mask = internal_y & even_y
            I[mask] = np.where(odd_x[mask], 7.0, 5.0)
            I[mask & right] = 7.0
            I[bottom] = np.where(odd_x[bottom], 8.0, 6.0)
            I[bottom & right] = 8.0
            return G, C3, I

        def x_measured_dq_coeffs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            G = np.empty((self.d, self.d), dtype=np.float64)
            C3 = np.empty_like(G)
            I = np.empty_like(G)

            mask = left | right
            G[mask] = np.where(odd_y[mask], 2.0, 3.0)
            mask = internal_x & odd_x
            G[mask] = np.where(odd_y[mask], 4.0, 3.0)
            mask = internal_x & even_x
            G[mask] = np.where(odd_y[mask], 3.0, 4.0)

            C3[top] = np.where(odd_x[top], 18.0, 18.0)
            C3[top & internal_x & even_x] = 22.0
            C3[top & right] = 15.0
            mask = (~top) & odd_y
            C3[mask] = np.where(odd_x[mask], 28.0, 20.0)
            C3[mask & internal_x & even_x] = 24.0
            C3[mask & right] = 11.0
            mask = (~top) & even_y
            C3[mask] = np.where(odd_x[mask], 24.0, 18.0)
            C3[mask & internal_x & even_x] = 28.0
            C3[mask & right] = 21.0

            I[top] = np.where(odd_x[top], 7.0, 8.0)
            I[top & internal_x & even_x] = 9.0
            I[top & right] = 10.0
            mask = (~top) & odd_y
            I[mask] = np.where(odd_x[mask], 7.0, 6.0)
            I[mask & internal_x & even_x] = 5.0
            I[mask & right] = 6.0
            mask = (~top) & even_y
            I[mask] = np.where(odd_x[mask], 5.0, 8.0)
            I[mask & internal_x & even_x] = 7.0
            I[mask & right] = 8.0
            return G, C3, I

        if action_key == "z_only":
            Dq_c3, Dq_i = add_z_data_side(Dq_c3, Dq_i)
            Dp_g, Dp_c3, Dp_i = z_measured_dp_coeffs()
        elif action_key == "x_only":
            Dq_g, Dq_c3, Dq_i = x_measured_dq_coeffs()
            Dp_c3, Dp_i = add_x_data_side(Dp_c3, Dp_i)
        elif action_key == "both":
            Dq_g, Dq_c3, Dq_i = x_measured_dq_coeffs()
            Dq_c3, Dq_i = add_z_data_side(Dq_c3, Dq_i)
            Dp_g, Dp_c3, Dp_i = z_measured_dp_coeffs()
            Dp_c3, Dp_i = add_x_data_side(Dp_c3, Dp_i)

        Dq = Dq_g * g + (Dq_c3 / 3.0) * c + Dq_i * idle
        Dp = Dp_g * g + (Dp_c3 / 3.0) * c + Dp_i * idle
        return Dq.reshape(-1).copy(), Dp.reshape(-1).copy()

    def _build_fixed_surface_table(self) -> dict[str, dict[str, np.ndarray]]:
        table: dict[str, dict[str, np.ndarray]] = {}
        for action_key in self.ACTIONS:
            Dq, Dp = self._data_variance_fixed_equation(action_key)
            Z_time, X_time = self._surface_time_variance_fixed_equation(action_key)
            table[action_key] = {"Dq": Dq, "Dp": Dp, "Z_time": Z_time, "X_time": X_time}
        return table

    def get_fixed_surface_time_sigma_vec(self, action_key: str) -> tuple[np.ndarray, np.ndarray]:
        if action_key not in self._fixed_surface_table:
            raise ValueError(f"Unknown global surface action: {action_key!r}")
        entry = self._fixed_surface_table[action_key]
        return np.sqrt(entry["Z_time"].copy()), np.sqrt(entry["X_time"].copy())

    def reset_round(self, reset_data: bool = False) -> None:
        self._Aq_var[:] = 0.0
        self._Ap_var[:] = 0.0
        self._space_ready = {0: False, 1: False}
        self._Z_time_read_var = None
        self._X_time_read_var = None
        self._time_ready = False
        if reset_data:
            self._Dq_var[:] = 0.0
            self._Dp_var[:] = 0.0

    def update_gkp_stabilizer_noise(self, step: int, faulty: bool = False) -> None:
        if step not in (0, 1):
            raise ValueError("step must be 0 or 1")

        s2 = self.sigma * self.sigma if faulty else 0.0
        g2 = self.sigma_GKP * self.sigma_GKP if faulty else 0.0
        i2 = self.sigma_idle * self.sigma_idle if faulty else 0.0
        read = s2
        wait = i2

        if faulty:
            self._Dq_var += i2
            self._Dp_var += i2

        idx = np.arange(self.n_data)
        even = idx % 2 == 0
        odd = ~even

        Dq_pre = self._Dq_var.copy()
        Dp_pre = self._Dp_var.copy()

        Aq = np.empty(self.n_data, dtype=np.float64)
        Ap = np.empty(self.n_data, dtype=np.float64)
        Dq_new = np.empty(self.n_data, dtype=np.float64)
        Dp_new = np.empty(self.n_data, dtype=np.float64)

        if step == 0:
            Aq[even] = Dq_pre[even] + g2 + (4.0 / 3.0) * s2 + read
            Ap[even] = g2 + s2 + read  
            Dq_new[even] = g2 + (4.0 / 3.0) * s2 + read + wait
            Dp_new[even] = Dp_pre[even] + g2 + (4.0 / 3.0) * s2 + wait

            Aq[odd] = g2 + s2 + read  
            Ap[odd] = Dp_pre[odd] + g2 + (4.0 / 3.0) * s2 + read
            Dq_new[odd] = Dq_pre[odd] + g2 + (4.0 / 3.0) * s2 + wait
            Dp_new[odd] = g2 + (4.0 / 3.0) * s2 + read + wait
        else:
            Aq[even] = g2 + s2 + read  
            Ap[even] = Dp_pre[even] + g2 + (4.0 / 3.0) * s2 + read
            Dq_new[even] = Dq_pre[even] + g2 + (4.0 / 3.0) * s2 + wait
            Dp_new[even] = g2 + (4.0 / 3.0) * s2 + read + wait

            Aq[odd] = Dq_pre[odd] + g2 + (4.0 / 3.0) * s2 + read
            Ap[odd] = g2 + s2 + read
            Dq_new[odd] = g2 + (4.0 / 3.0) * s2 + read + wait
            Dp_new[odd] = Dp_pre[odd] + g2 + (4.0 / 3.0) * s2 + wait

        self._Aq_var[step] = Aq
        self._Ap_var[step] = Ap
        self._Dq_var = Dq_new
        self._Dp_var = Dp_new
        self._space_ready[step] = True

    def update_surface_stabilizer_noise(
        self,
        selected_Z_synd_list: Optional[Sequence[int]] = None,
        selected_X_synd_list: Optional[Sequence[int]] = None,
        faulty: bool = False,
    ) -> None:
        if not faulty:
            zeros = np.zeros(self.n_syn, dtype=np.float64)
            self._Z_time_read_var = zeros
            self._X_time_read_var = zeros.copy()
            self._time_ready = True
            return

        action_key = self.get_surface_action_key(selected_Z_synd_list, selected_X_synd_list)
        entry = self._fixed_surface_table[action_key]
        self._Dq_var = entry["Dq"].copy()
        self._Dp_var = entry["Dp"].copy()
        self._Z_time_read_var = entry["Z_time"].copy()
        self._X_time_read_var = entry["X_time"].copy()
        self._time_ready = True

    def get_Z_sigma_space(self, i: int, t: int, step: int) -> float:
        return float(self.get_Z_sigma_space_vec(step)[i])

    def get_X_sigma_space(self, i: int, t: int, step: int) -> float:
        return float(self.get_X_sigma_space_vec(step)[i])

    def get_Z_sigma_time(self, i: int) -> float:
        return float(self.get_Z_sigma_time_vec()[i])

    def get_X_sigma_time(self, i: int) -> float:
        return float(self.get_X_sigma_time_vec()[i])

    def get_Z_sigma_space_vec(self, step: int) -> np.ndarray:
        if not self._space_ready.get(step, False):
            raise RuntimeError(f"GKP space for step={step} is not updated yet.")
        return np.sqrt(self._Aq_var[step].copy())

    def get_X_sigma_space_vec(self, step: int) -> np.ndarray:
        if not self._space_ready.get(step, False):
            raise RuntimeError(f"GKP space for step={step} is not updated yet.")
        return np.sqrt(self._Ap_var[step].copy())

    def get_Z_sigma_time_vec(self) -> np.ndarray:
        if not self._time_ready or self._Z_time_read_var is None:
            raise RuntimeError("Surface time-like channels are not updated yet.")
        return np.sqrt(self._Z_time_read_var.copy())

    def get_X_sigma_time_vec(self) -> np.ndarray:
        if not self._time_ready or self._X_time_read_var is None:
            raise RuntimeError("Surface time-like channels are not updated yet.")
        return np.sqrt(self._X_time_read_var.copy())
