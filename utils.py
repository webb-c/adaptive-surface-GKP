import csv
import numpy as np

from argparse import ArgumentTypeError
from pathlib import Path
from scipy.special import erf

EPS = 1e-300


def str2bool(v):
    if isinstance(v, bool):
        return v
    value = v.lower()
    if value in ("true", "yes", "t"):
        return True
    if value in ("false", "no", "f"):
        return False
    raise ArgumentTypeError("Boolean value expected.")


def pos_int(x: str) -> int:
    try:
        value = int(x)
    except ValueError as exc:
        raise ArgumentTypeError(f"{x!r} is not a valid int") from exc
    if value <= 0:
        raise ArgumentTypeError(f"{value} is not a positive integer")
    return value


def pos_odd_int(x: str) -> int:
    value = pos_int(x)
    if value % 2 == 0:
        raise ArgumentTypeError(f"{value} is not an odd integer")
    return value


def rand_Gvec(D, O, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    std = np.sqrt(np.diag(D))
    return O @ (std * rng.standard_normal(len(std)))


def keep_odd(vec: np.ndarray) -> np.ndarray:
    out = np.zeros_like(vec)
    out[1::2] = vec[1::2]
    return out


def keep_even(vec: np.ndarray) -> np.ndarray:
    out = np.zeros_like(vec)
    out[::2] = vec[::2]
    return out


def transform_centered_mod(period: float, z):
    z = np.asarray(z)
    return z - period * np.floor(z / period + 0.5)


def cond_err_prob(sigma: float, z: float, K: int = 3) -> float:
    sigma_arr = np.asarray([sigma], dtype=float)
    z_arr = np.asarray([z], dtype=float)
    return float(cond_err_prob_vec(sigma_arr, z_arr, K=K)[0])


def err_prob(sigma: float, k_max: int = 3) -> float:
    return float(err_prob_vec(np.asarray([sigma], dtype=float), k_max=k_max)[0])


def find_one(arr: np.ndarray) -> int:
    idx = np.flatnonzero(arr == 1)
    return int(idx[0]) if idx.size else -1


def make_mask(idx_list, vec_len: int) -> np.ndarray:
    mask = np.zeros(vec_len, dtype=bool)
    if idx_list is not None:
        mask[np.asarray(idx_list, dtype=int)] = True
    if mask.size != vec_len:
        raise ValueError("Length error")
    return mask


def extract_edges_from_nx(graph):
    edges = []
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)
        if not isinstance(weight, float):
            weight = weight.item()
        edges.append([[u, v], weight])
    return edges


def print_edges_for_nx(graph):
    for item in extract_edges_from_nx(graph):
        print(item)


def save_results(time, n_iter, distance, n_round, sigma, sigma_GKP, sigma_idle, with_info, method, X_err_rate, Z_err_rate, Y_err_rate, avg_n_measure, csv_path_str):
    csv_path = Path(csv_path_str)
    header = [
        "time",
        "n_iter",
        "distance",
        "n_round",
        "sigma",
        "sigma_GKP",
        "sigma_idle",
        "with_info",
        "method",
        "X_err_rate",
        "Z_err_rate",
        "Y_err_rate",
        "n_measure",
    ]
    row = [
        time,
        n_iter,
        distance,
        n_round,
        f"{sigma:.4f}",
        f"{sigma_GKP:.4f}",
        f"{sigma_idle:.4f}",
        with_info,
        method,
        f"{X_err_rate:.15f}",
        f"{Z_err_rate:.15f}",
        f"{Y_err_rate:.15f}",
        avg_n_measure,
    ]

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def save_results_counter(time, n_iter, distance, n_round, sigma, sigma_GKP, sigma_idle, with_info, method, avg_node, avg_missing, avg_wrong, X_err_rate, Z_err_rate, Y_err_rate, avg_n_measure, csv_path_str):
    csv_path = Path(csv_path_str)
    header = [
        "time",
        "n_iter",
        "distance",
        "n_round",
        "sigma",
        "sigma_GKP",
        "sigma_idle",
        "with_info",
        "method",
        "avg_node",
        "avg_missing",
        "avg_wrong",
        "X_err_rate",
        "Z_err_rate",
        "Y_err_rate",
        "avg_n_measure",
    ]
    row = [
        time,
        n_iter,
        distance,
        n_round,
        f"{sigma:.4f}",
        f"{sigma_GKP:.4f}",
        f"{sigma_idle:.4f}",
        with_info,
        method,
        avg_node,
        avg_missing,
        avg_wrong,
        X_err_rate,
        Z_err_rate,
        Y_err_rate,
        avg_n_measure,
    ]

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def cond_err_prob_vec(sigma: np.ndarray, z: np.ndarray, K: int = 3) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    z = np.asarray(z, dtype=float)
    sigma, z = np.broadcast_arrays(sigma, z)

    out_shape = sigma.shape
    sigma = sigma.reshape(-1)
    z = z.reshape(-1)

    root_pi = np.sqrt(np.pi)
    k = np.arange(-K, K + 1, dtype=float)
    inv2 = 0.5 / np.maximum(sigma * sigma, 1e-30)
    zc = z[:, None]

    p_even = np.exp(-inv2[:, None] * (zc - (2.0 * k) * root_pi) ** 2).sum(axis=1)
    p_odd = np.exp(-inv2[:, None] * (zc - (2.0 * k + 1.0) * root_pi) ** 2).sum(axis=1)

    p_err = p_odd / np.maximum(p_even + p_odd, EPS)
    return np.clip(p_err, 0.0, 1.0).reshape(out_shape)


def err_prob_vec(sigma: np.ndarray, k_max: int = 3) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    k = np.arange(-k_max, k_max + 1, dtype=float)
    coeff = np.sqrt(np.pi / 2.0) / np.maximum(sigma, 1e-30)
    a = (2 * k + 1.5)[None, :] * coeff[:, None]
    b = (2 * k + 0.5)[None, :] * coeff[:, None]
    p = 0.5 * (erf(a) - erf(b))
    return p.sum(axis=1)
