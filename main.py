import argparse
import csv
import math
import multiprocessing as mp
import os
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from simulator import (
    SurfaceGKPDefaultTrackingSimulator,
    SurfaceGKPInfoMeasureWeightTrackingMeasureSimulator,
    SurfaceGKPInfoMeasureWeightTrackingSimulator,
)
from utils import pos_int, pos_odd_int, save_results, save_results_counter, str2bool

SUPPORTED_MODES = (
    "default_tracking",
    "adaptive",
    "adaptive_meas",
    "adaptive_round",
    "skip",
    "skip_round",
)
SUPPORTED_SCORE_METRICS = ("global_joint",)
SUPPORTED_STAB_ERROR_PROBS = ("one", "parity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--distance", type=pos_odd_int, default=3, help="code distance")
    parser.add_argument("-r", "--n_round", type=int, default=3, help="number of noisy surface-measurement rounds")
    parser.add_argument("-n", "--n_iter", type=pos_int, default=1, help="number of iterations")
    parser.add_argument("-s", "--sigma", type=float, default=1.0, help="circuit-level noise standard deviation")
    parser.add_argument("-g", "--sigma_GKP", type=float, default=1.0, help="GKP noise standard deviation")
    parser.add_argument("-i", "--sigma_idle", type=float, default=None, help="idle noise standard deviation")
    parser.add_argument("-w", "--with_info", type=str2bool, default=True)
    parser.add_argument("-c", "--clean", type=str2bool, default=False, help="inner code measurement is clean or not")
    parser.add_argument("-m", "--mode", type=str, choices=SUPPORTED_MODES, default="default_tracking")
    parser.add_argument("-o", "--score_metric", type=str, choices=SUPPORTED_SCORE_METRICS, default="global_joint")
    parser.add_argument(
        "--stab_error_prob",
        type=str,
        choices=SUPPORTED_STAB_ERROR_PROBS,
        default="parity",
        help=(
            "model for skipped stabilizer success probability: "
            "'parity' uses odd-parity syndrome-flip probability; "
            "'one' uses at-least-one-error probability"
        ),
    )
    parser.add_argument(
        "--target_rounds",
        type=str,
        default="",
        help="comma-separated round list. Keywords: first, middle, last, final, all, all_with_final",
    )
    parser.add_argument(
        "--skip_rounds",
        type=str,
        default="",
        help="comma-separated round list. Keywords: first, middle, last, final, all, all_with_final",
    )
    parser.add_argument("--final_noise", type=str2bool, default=False)
    parser.add_argument("--csv_path", type=str, default="results/results.csv", help="path of results csv file")
    parser.add_argument("--log", type=str2bool, default=True)
    parser.add_argument("--check", type=str2bool, default=False)
    parser.add_argument("--fix_seed", type=str2bool, default=False)
    parser.add_argument("--workers", type=int, default=0, help="number of processes for multiprocessing (0=auto, 1=serial)")
    parser.add_argument("--verbose", type=str2bool, default=False)
    args = parser.parse_args()
    if args.sigma_idle is None:
        args.sigma_idle = args.sigma
    args.target_round_list = _parse_round_spec(args.target_rounds, args.n_round)
    args.skip_round_list = _parse_round_spec(args.skip_rounds, args.n_round)
    return args


def _parse_round_spec(spec: str, n_round: int):
    if not spec:
        return []
    rounds = set()
    for raw_token in spec.split(","):
        token = raw_token.strip().lower()
        if not token or token == "none":
            continue
        if token == "first":
            rounds.add(0)
        elif token == "middle":
            rounds.add(n_round // 2)
        elif token == "last":
            rounds.add(max(n_round - 1, 0))
        elif token == "final":
            rounds.add(n_round)
        elif token == "all":
            rounds.update(range(n_round))
        elif token == "all_with_final":
            rounds.update(range(n_round + 1))
        else:
            value = int(token)
            if value < 0 or value > n_round:
                raise ValueError(f"Round index out of range: {value}. Allowed range is [0, {n_round}].")
            rounds.add(value)
    return sorted(rounds)


def _round_target_list(distance: int, n_round: int):
    return sorted(set(range(distance, n_round + 1, distance)) | {n_round})


def _build_common_args(args: argparse.Namespace, n_iter_override: int):
    return dict(
        d=args.distance,
        n_round=args.n_round,
        n_iter=n_iter_override,
        sigma=args.sigma,
        sigma_GKP=args.sigma_GKP,
        sigma_idle=args.sigma_idle,
        with_info=args.with_info,
        clean=args.clean,
        fix_seed=args.fix_seed,
        verbose=args.verbose,
        check_counter=args.check,
        final_noise=args.final_noise,
    )


def _build_simulator(args: argparse.Namespace, n_iter_override: int):
    common = _build_common_args(args, n_iter_override)
    if args.mode == "default_tracking":
        return SurfaceGKPDefaultTrackingSimulator(**common)
    if args.mode == "adaptive":
        return SurfaceGKPInfoMeasureWeightTrackingSimulator(
            **common,
            score_metric=args.score_metric,
            stab_error_prob=args.stab_error_prob,
        )
    if args.mode == "adaptive_meas":
        return SurfaceGKPInfoMeasureWeightTrackingMeasureSimulator(
            **common,
            score_metric=args.score_metric,
            stab_error_prob=args.stab_error_prob,
            target_rounds=args.target_round_list,
            skip_rounds=args.skip_round_list,
        )
    if args.mode in {"skip"}:
        return SurfaceGKPInfoMeasureWeightTrackingMeasureSimulator(
            **common,
            score_metric=args.score_metric,
            stab_error_prob=args.stab_error_prob,
            target_rounds=[args.n_round],
            skip_rounds=list(range(args.n_round)),
            is_skip=False,                              
            # argument for testing skipped information (False: using the skipped information)
        )
    if args.mode == "adaptive_round":
        return SurfaceGKPInfoMeasureWeightTrackingMeasureSimulator(
            **common,
            score_metric=args.score_metric,
            stab_error_prob=args.stab_error_prob,
            target_rounds=_round_target_list(args.distance, args.n_round),
            skip_rounds=[],
        )
    if args.mode == "skip_round":
        all_rounds = list(range(args.n_round + 1))
        target_round_list = _round_target_list(args.distance, args.n_round)
        skip_round_list = [r for r in all_rounds if r not in target_round_list]
        return SurfaceGKPInfoMeasureWeightTrackingMeasureSimulator(
            **common,
            score_metric=args.score_metric,
            stab_error_prob=args.stab_error_prob,
            target_rounds=target_round_list,
            skip_rounds=skip_round_list,
        )
    raise ValueError(f"Unknown mode: {args.mode}")


def _child_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _worker_chunk(args_tuple):
    args, chunk_iter = args_tuple
    sim = None
    try:
        sim = _build_simulator(args, chunk_iter)
        sim.run(verbose=args.verbose, save_fig=False, log=args.log)
        return (
            sim.n_X_err,
            sim.n_Z_err,
            sim.n_Y_err,
            sim.n_measure,
            sim.n_decoding_node,
            sim.n_decoding_node_diff,
            sim.miss_count,
            sim.wrong_count,
            sim.n_iter,
        )
    finally:
        del sim


def _run_parallel(args: argparse.Namespace):
    total_iters = args.n_iter
    workers = (os.cpu_count() or 1) if args.workers <= 0 else max(1, args.workers)
    workers = min(workers, max(1, total_iters))
    chunk = math.ceil(total_iters / workers)
    tasks = []
    for worker_idx in range(workers):
        start = worker_idx * chunk
        end = min((worker_idx + 1) * chunk, total_iters)
        if start < end:
            tasks.append((args, end - start))

    Xc = Zc = Yc = meas_total = iters_total = 0
    n_decoding_node_total = n_decoding_node_diff_total = 0
    miss_count_total = wrong_count_total = 0
    ctx = mp.get_context("forkserver")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_child_init) as ex:
            futures = [ex.submit(_worker_chunk, task) for task in tasks]
            for future in as_completed(futures):
                x, z, y, m, n_decoding_node, n_decoding_node_diff, miss_count, wrong_count, it = future.result()
                Xc += x
                Zc += z
                Yc += y
                meas_total += m
                iters_total += it
                n_decoding_node_total += n_decoding_node
                n_decoding_node_diff_total += n_decoding_node_diff
                miss_count_total += miss_count
                wrong_count_total += wrong_count
    except Exception:
        try:
            ex.shutdown(wait=False, cancel_futures=True)  # type: ignore[name-defined]
        except Exception:
            pass
        raise
    return (
        Xc / iters_total,
        Zc / iters_total,
        Yc / iters_total,
        meas_total / iters_total,
        n_decoding_node_total / iters_total,
        n_decoding_node_diff_total / iters_total,
        miss_count_total / iters_total,
        wrong_count_total / iters_total,
    )


def _run_serial(args: argparse.Namespace):
    sim = _build_simulator(args, args.n_iter)
    return sim.run(verbose=args.verbose, save_fig=False, log=args.log), sim


def _format_rounds(round_list):
    return "-".join(str(x) for x in round_list) if round_list else "none"


def _build_method_name(args: argparse.Namespace) -> str:
    parts = [args.mode]
    if args.mode in {"adaptive", "adaptive_meas", "adaptive_round"}:
        parts.extend([args.score_metric, "success", args.stab_error_prob, "zero"])
    if args.mode in {"skip", "skip_round"}:
        parts.append("zero")
    if args.mode == "adaptive_meas":
        parts.append(f"TARGET_{_format_rounds(args.target_round_list)}")
        parts.append(f"SKIP_{_format_rounds(args.skip_round_list)}")
    if args.mode == "adaptive_round":
        parts.append(f"TARGET_{_format_rounds(_round_target_list(args.distance, args.n_round))}")
    if args.mode == "skip_round":
        parts.append(f"TARGET_{_format_rounds(_round_target_list(args.distance, args.n_round))}")
    method = "_".join(parts)
    if args.clean:
        method += "_clean"
    return method


def main():
    args = parse_args()
    start_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
    if args.workers == 1:
        data, sim = _run_serial(args)
        X_err_rate, Z_err_rate, Y_err_rate, avg_n_measure, avg_n_decoding_node, avg_n_decoding_node_diff, avg_missing, avg_wrong = data
        if args.check:
            print(sim.get_detection_event_stats())
    else:
        X_err_rate, Z_err_rate, Y_err_rate, avg_n_measure, avg_n_decoding_node, avg_n_decoding_node_diff, avg_missing, avg_wrong = _run_parallel(args)
    method = _build_method_name(args)
    print(method)
    print(f"X: {X_err_rate:.5f}, Z: {Z_err_rate:.5f}, Y: {Y_err_rate:.5f}, # measure: {avg_n_measure}")
    if args.check:
        print(
            f"avg # decoding nodes: {avg_n_decoding_node:.2f}, "
            f"avg # avg_n_decoding_node_diff: {avg_n_decoding_node_diff:.2f}, "
            f"avg missing: {avg_missing:.2f}, avg wrong: {avg_wrong:.2f}"
        )
    if args.log:
        try:
            if args.check:
                save_results_counter(
                    start_time,
                    args.n_iter,
                    args.distance,
                    args.n_round,
                    args.sigma,
                    args.sigma_GKP,
                    args.sigma_idle,
                    args.with_info,
                    method,
                    avg_n_decoding_node,
                    avg_missing,
                    avg_wrong,
                    X_err_rate,
                    Z_err_rate,
                    Y_err_rate,
                    avg_n_measure,
                    args.csv_path,
                )
            else:
                save_results(
                    start_time,
                    args.n_iter,
                    args.distance,
                    args.n_round,
                    args.sigma,
                    args.sigma_GKP,
                    args.sigma_idle,
                    args.with_info,
                    method,
                    X_err_rate,
                    Z_err_rate,
                    Y_err_rate,
                    avg_n_measure,
                    args.csv_path,
                )
        except (OSError, csv.Error):
            print("(*) Saving results is failed.")


if __name__ == "__main__":
    main()
