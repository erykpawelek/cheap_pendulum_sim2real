import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import simulation.envs


def parse_args():
    """Parses command-line arguments for flexible evaluation modes and CSV logging."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO controller on FurutaPendulum-v0."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of inference episodes to run (ignored if --infinite is set).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable real-time 3D MuJoCo visualization during evaluation.",
    )
    parser.add_argument(
        "--infinite",
        action="store_true",
        help="Run infinite real-time visual loop without episode limit (forces --render).",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable metric calculation, summary table, and CSV saving.",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="training/logs",
        help="Directory where evaluation CSV log files will be saved.",
    )
    return parser.parse_args()


def warmup_window(eval_env, frames=45):
    """Pumps OS window events without stepping physics to prevent initial visual stutter."""
    for _ in range(frames):
        eval_env.render()
        time.sleep(1.0 / 60.0)


def save_metrics_to_csv(episode_rows, results, csv_dir):
    """Saves per-episode metrics and summary statistics (MEAN, STD) to a timestamped CSV file."""
    log_dir = Path(csv_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = log_dir / f"eval_metrics_{timestamp}.csv"

    fieldnames = [
        "episode",
        "success",
        "stalled",
        "return",
        "settling_time_s",
        "upright_ratio_pct",
        "rmse_total_rad",
        "rmse_dynamic_rad",
        "rmse_balanced_rad",
        "iae_total_rad_s",
        "control_effort",
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Write individual episode rows
        for row in episode_rows:
            writer.writerow(row)

        # Write aggregated summary rows (MEAN and STD) at the bottom
        if episode_rows:
            n_eps = len(episode_rows)
            sr = (results["successes"] / n_eps) * 100.0
            stall_rate = (results["stalls"] / n_eps) * 100.0
            mean_settle = (
                f"{np.mean(results['settling_times']):.4f}"
                if results["settling_times"]
                else "NaN"
            )
            std_settle = (
                f"{np.std(results['settling_times']):.4f}"
                if results["settling_times"]
                else "NaN"
            )

            writer.writerow({
                "episode": "MEAN",
                "success": f"{sr:.2f}%",
                "stalled": f"{stall_rate:.2f}%",
                "return": f"{np.mean(results['returns']):.4f}",
                "settling_time_s": mean_settle,
                "upright_ratio_pct": f"{np.mean(results['upright_ratios']):.2f}",
                "rmse_total_rad": f"{np.mean(results['rmse_total']):.4f}",
                "rmse_dynamic_rad": f"{np.mean(results['rmse_Dynamic']):.4f}",
                "rmse_balanced_rad": f"{np.mean(results['rmse_balanced']):.4f}",
                "iae_total_rad_s": f"{np.mean(results['iae_total']):.4f}",
                "control_effort": f"{np.mean(results['control_effort']):.4f}",
            })

            writer.writerow({
                "episode": "STD",
                "success": "-",
                "stalled": "-",
                "return": f"{np.std(results['returns']):.4f}",
                "settling_time_s": std_settle,
                "upright_ratio_pct": f"{np.std(results['upright_ratios']):.2f}",
                "rmse_total_rad": f"{np.std(results['rmse_total']):.4f}",
                "rmse_dynamic_rad": f"{np.std(results['rmse_Dynamic']):.4f}",
                "rmse_balanced_rad": f"{np.std(results['rmse_balanced']):.4f}",
                "iae_total_rad_s": f"{np.std(results['iae_total']):.4f}",
                "control_effort": f"{np.std(results['control_effort']):.4f}",
            })

    return csv_path


def main():
    args = parse_args()
    if args.infinite:
        args.render = True

    env_id = "FurutaPendulum-v0"
    model_path = "training/logs/ppo_furuta_pendulum"
    stats_path = "training/logs/vec_normalize.pkl"

    render_mode = "human" if args.render else None

    # Initialize vectorized environment with dynamic render_mode
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode=render_mode)])

    try:
        eval_env = VecNormalize.load(stats_path, eval_env)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Normalization file '{stats_path}' not found. Train the model first."
        )

    # Freeze normalization statistics during inference
    eval_env.training = False
    eval_env.norm_reward = False

    try:
        model = PPO.load(model_path, env=eval_env)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{model_path}.zip' not found.")

    # Extract physics timestep (dt) from the unwrapped MuJoCo environment
    try:
        dt = eval_env.envs[0].unwrapped.dt
    except AttributeError:
        dt = 0.02

    # Thresholds for Furuta Pendulum benchmark metrics
    balance_angle_tol = 0.20      # ~11.5 deg from upright (pi)
    balance_vel_tol = 1.0         # rad/s max velocity to consider "settled"
    min_settled_steps = int(1.0 / dt)  # Must hold upright for at least 1.0 simulated second

    print(f"Loaded '{model_path}.zip' | dt = {dt}s | Render: {args.render} | Infinite: {args.infinite}")

    obs = eval_env.reset()
    if args.render:
        print("Warming up OpenGL window...")
        warmup_window(eval_env, frames=60)

    # Storage for aggregated benchmark metrics across episodes
    results = {
        "returns": [],
        "rmse_total": [],
        "rmse_Dynamic": [],
        "rmse_balanced": [],
        "iae_total": [],
        "control_effort": [],
        "settling_times": [],
        "upright_ratios": [],
        "successes": 0,
        "stalls": 0,
    }
    episode_rows = []

    ep_idx = 0
    wall_start = time.perf_counter()

    try:
        while True:
            # Variables for single episode tracking
            ep_return = 0.0
            ep_steps = 0
            settled_streak = 0
            settling_step = None
            stalled_in_ep = False

            angle_errors_sq = []
            dynamic_errors_sq = []
            balanced_errors_sq = []
            iae_accum = 0.0
            effort_accum = 0.0
            upright_steps = 0

            done = [False]
            while not done[0]:
                step_start = time.perf_counter()

                # Query Actor network in deterministic mode
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)

                if not args.no_metrics:
                    # Retrieve raw unnormalized observations: [sin, cos, pend_vel, arm_pos, arm_vel]
                    raw_obs = eval_env.get_original_obs()[0]
                    sin_th, cos_th, pend_vel = raw_obs[0], raw_obs[1], raw_obs[2]

                    # Compute exact angular deviation from upright pi in range [0, pi]
                    angle_err = abs(np.arctan2(sin_th, -cos_th))
                    act_val = float(action[0][0]) if np.ndim(action) > 1 else float(action[0])

                    ep_return += float(reward[0])
                    ep_steps += 1

                    # Accumulate whole-episode integrals
                    angle_errors_sq.append(angle_err ** 2)
                    iae_accum += angle_err * dt
                    effort_accum += (act_val ** 2) * dt

                    # Check if currently inside balance zone
                    if angle_err < balance_angle_tol:
                        upright_steps += 1
                        balanced_errors_sq.append(angle_err ** 2)
                    else:
                        dynamic_errors_sq.append(angle_err ** 2)

                    # Detect settling time (transition from dynamic swing-up to steady balance)
                    if angle_err < balance_angle_tol and abs(pend_vel) < balance_vel_tol:
                        settled_streak += 1
                        if settled_streak >= min_settled_steps and settling_step is None:
                            settling_step = ep_steps - min_settled_steps + 1
                    else:
                        settled_streak = 0

                    # Detect motor stall from info dict or early termination before max horizon
                    if info[0].get("stalled", False) or (done[0] and not info[0].get("TimeLimit.truncated", False)):
                        stalled_in_ep = True

                # Real-time synchronization when rendering is active
                if args.render:
                    compute_time = time.perf_counter() - step_start
                    if compute_time < dt:
                        time.sleep(dt - compute_time)

            ep_idx += 1

            # Record episode metrics
            if not args.no_metrics and ep_steps > 0:
                ep_rmse = float(np.sqrt(np.mean(angle_errors_sq)))
                dyn_rmse = float(np.sqrt(np.mean(dynamic_errors_sq))) if dynamic_errors_sq else 0.0
                bal_rmse = float(np.sqrt(np.mean(balanced_errors_sq))) if balanced_errors_sq else np.pi
                up_ratio = (upright_steps / ep_steps) * 100.0
                is_success = settling_step is not None and not stalled_in_ep
                settle_time_val = settling_step * dt if settling_step is not None else None

                results["returns"].append(ep_return)
                results["rmse_total"].append(ep_rmse)
                results["rmse_Dynamic"].append(dyn_rmse)
                results["rmse_balanced"].append(bal_rmse)
                results["iae_total"].append(iae_accum)
                results["control_effort"].append(effort_accum)
                results["upright_ratios"].append(up_ratio)

                if is_success:
                    results["successes"] += 1
                    results["settling_times"].append(settle_time_val)
                if stalled_in_ep:
                    results["stalls"] += 1

                # Store formatted row for CSV export
                episode_rows.append({
                    "episode": ep_idx,
                    "success": int(is_success),
                    "stalled": int(stalled_in_ep),
                    "return": f"{ep_return:.4f}",
                    "settling_time_s": f"{settle_time_val:.4f}" if settle_time_val is not None else "NaN",
                    "upright_ratio_pct": f"{up_ratio:.2f}",
                    "rmse_total_rad": f"{ep_rmse:.4f}",
                    "rmse_dynamic_rad": f"{dyn_rmse:.4f}",
                    "rmse_balanced_rad": f"{bal_rmse:.4f}",
                    "iae_total_rad_s": f"{iae_accum:.4f}",
                    "control_effort": f"{effort_accum:.4f}",
                })

                if args.render or (ep_idx % 10 == 0) or (ep_idx == args.episodes):
                    t_set_str = f"{settle_time_val:.2f}s" if settle_time_val is not None else "N/A"
                    print(
                        f"Ep {ep_idx:3d} | Return: {ep_return:7.1f} | "
                        f"Swing-Up Time: {t_set_str:>6} | "
                        f"Upright: {up_ratio:5.1f}% | "
                        f"RMSE: {ep_rmse:.3f} rad | "
                        f"Stalled: {stalled_in_ep}"
                    )

            # Reset for next episode
            obs = eval_env.reset()
            if args.render:
                warmup_window(eval_env, frames=25)

            # Check exit condition for finite mode
            if not args.infinite and ep_idx >= args.episodes:
                break

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        eval_env.close()

    # Print final aggregated benchmark report and save CSV
    if not args.no_metrics and ep_idx > 0:
        wall_elapsed = time.perf_counter() - wall_start
        sr = (results["successes"] / ep_idx) * 100.0
        stall_rate = (results["stalls"] / ep_idx) * 100.0
        mean_settle = np.mean(results["settling_times"]) if results["settling_times"] else float("nan")
        std_settle = np.std(results["settling_times"]) if results["settling_times"] else float("nan")

        print("\n" + "=" * 65)
        print(f" FURUTA PENDULUM BENCHMARK REPORT ({ep_idx} EPISODES in {wall_elapsed:.2f}s)")
        print("=" * 65)
        print(f" Success Rate (Settled >= 1s) : {sr:6.1f} %")
        print(f" Stepper Motor Stall Rate     : {stall_rate:6.1f} %")
        print(f" Mean Episode Return          : {np.mean(results['returns']):6.2f} +/- {np.std(results['returns']):.2f}")
        print("-" * 65)
        print(" 1. DYNAMIC PHASE (SWING-UP):")
        print(f"    Settling Time (T_s)       : {mean_settle:6.2f} s   +/- {std_settle:.2f} s")
        print(f"    Dynamic Phase RMSE        : {np.mean(results['rmse_Dynamic']):6.3f} rad")
        print("-" * 65)
        print(" 2. FULL EPISODE & STABILIZATION:")
        print(f"    Time Spent Upright (<11°) : {np.mean(results['upright_ratios']):6.1f} %")
        print(f"    Whole-Episode Angle RMSE  : {np.mean(results['rmse_total']):6.3f} rad")
        print(f"    Balanced Zone Angle RMSE  : {np.mean(results['rmse_balanced']):6.3f} rad")
        print(f"    Integral Abs Error (IAE)  : {np.mean(results['iae_total']):6.2f} rad*s")
        print(f"    Control Effort (int u^2)  : {np.mean(results['control_effort']):6.2f} (rad/s)^2*s")
        print("=" * 65)

        csv_file = save_metrics_to_csv(episode_rows, results, args.csv_dir)
        print(f"Saved evaluation metrics to CSV: '{csv_file}'")


if __name__ == "__main__":
    main()