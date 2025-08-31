# cv_baseline.py
# Constant-velocity baseline + minADE/minFDE on nuScenes (mini by default)

import argparse
import numpy as np
from tqdm import tqdm

import csv, os
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper


def constant_velocity_forecast(past_xy: np.ndarray, horizon_s: float, dt: float) -> np.ndarray:
    """
    past_xy: [T_p, 2] in agent frame, last row = most recent (t=0^-)
    Returns: [T_f, 2] for future times (t = dt, 2dt, ..., horizon_s)
    """
    if past_xy.shape[0] < 2:
        # not enough history: predict staying still
        last = past_xy[-1]
        H = int(round(horizon_s / dt))
        return np.tile(last[None, :], (H, 1))

    v = past_xy[-1] - past_xy[-2]                  # simple finite difference
    H = int(round(horizon_s / dt))
    preds = [past_xy[-1] + (i + 1) * v for i in range(H)]
    return np.stack(preds, axis=0)                 # [H,2]

def rotate(v, theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[c, -s],[s, c]], dtype=np.float32)
    return (R @ v.reshape(2,1)).reshape(2,)

def kmodal_cv(past_xy, horizon_s, dt, theta_deg=8.0):
    v = past_xy[-1] - past_xy[-2]
    H = int(round(horizon_s/dt))
    thetas = [0.0, np.deg2rad(theta_deg), -np.deg2rad(theta_deg)]
    trajs = []
    for th in thetas:
        v_rot = rotate(v, th)
        preds = [past_xy[-1] + (i+1)*v_rot for i in range(H)]
        trajs.append(np.stack(preds, 0))
    return np.stack(trajs, 0)  # [3,H,2]


def minade_minfde(preds: np.ndarray, gt: np.ndarray):
    """
    preds: [K, H, 2]  (K modes; for CV K=1)
    gt:    [H, 2]
    Returns: (minADE, minFDE)
    """
    # broadcast: [K,H,2] - [1,H,2]
    errs = np.linalg.norm(preds - gt[None, ...], axis=-1)  # [K,H]
    ade = errs.mean(axis=1).min()
    fde = errs[:, -1].min()
    return float(ade), float(fde)


def main(args):
    nusc = NuScenes(version=f'v1.0-{args.version}', dataroot=args.dataroot, verbose=False)
    helper = PredictHelper(nusc)

    seconds_past   = args.seconds_past
    seconds_future = args.seconds_future
    dt             = args.dt
    Tp = int(round(seconds_past / dt))
    Tf = int(round(seconds_future / dt))

    total = 0
    ades, fdes = [], []
    rows = [] if args.save_csv else None

    # Iterate all samples; for each visible annotation (agent), try building a (past, future) pair
    for sample in tqdm(nusc.sample, desc=f'Building CV baseline ({args.version})'):
        s_tok = sample['token']
        for ann_tok in sample['anns']:
            ann = nusc.get('sample_annotation', ann_tok)
            inst_tok = ann['instance_token']

            # Get history & future in the AGENT frame (so x,y are relative to the agent at t=0)
            past_xy = helper.get_past_for_agent(
                inst_tok, s_tok, seconds=seconds_past,
                in_agent_frame=True, just_xy=True
            )
            future_xy = helper.get_future_for_agent(
                inst_tok, s_tok, seconds=seconds_future,
                in_agent_frame=True, just_xy=True
            )

            # Skip if not enough history/future around this sample
            if past_xy is None or future_xy is None:
                continue
            if len(past_xy) != Tp or len(future_xy) != Tf:
                continue

            past_xy   = np.asarray(past_xy, dtype=np.float32)
            future_xy = np.asarray(future_xy, dtype=np.float32)

            pred = kmodal_cv(past_xy, horizon_s=seconds_future, dt=dt)    # [3, Tf,2]
            # ade, fde = minade_minfde(pred[None, ...], future_xy)                          # K=1
            ade, fde = minade_minfde(pred, future_xy)  # <-- pass pred directly (no None/extra dim)

            ades.append(ade); fdes.append(fde)
            if rows is not None:
                rows.append({"sample_token": s_tok, "instance_token": inst_tok, "ade": ade, "fde": fde})

            # optional viz
            if args.save_viz_dir and (total % args.viz_every == 0):
                os.makedirs(args.save_viz_dir, exist_ok=True)
                fig = plt.figure(figsize=(4,4))
                # past (blue), GT future (green), CV pred (red)
                px, py = past_xy[:,0], past_xy[:,1]
                gx, gy = future_xy[:,0], future_xy[:,1]
                rx, ry = pred[:,0], pred[:,1]
                plt.plot(px, py, '-o', label='past', alpha=0.6)
                plt.plot(gx, gy, '-o', label='gt', alpha=0.8)
                plt.plot(rx, ry, '-o', label='cv', alpha=0.8)
                plt.legend(); plt.axis('equal'); plt.grid(True)
                plt.title(f"{s_tok[:6]}… | ADE {ade:.2f} FDE {fde:.2f}")
                fig.savefig(os.path.join(args.save_viz_dir, f"{total:06d}.png"), bbox_inches='tight')
                plt.close(fig)

            total += 1

            if args.max_pairs > 0 and total >= args.max_pairs:
                break
        if args.max_pairs > 0 and total >= args.max_pairs:
            break
    
    if rows is not None and len(rows):
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Saved CSV to {args.save_csv}")


    if total == 0:
        print("No valid (past,future) pairs found. Try reducing seconds_past/future or check your dataset path/version.")
        return

    print(f"\nEvaluated pairs: {total}")
    print(f"Constant-Velocity baseline on nuScenes-{args.version}:")
    print(f"  ADE  (mean) = {np.mean(ades):.3f} m")
    print(f"  FDE  (mean) = {np.mean(fdes):.3f} m")

    if args.histogram:
        try:
            rows = 1; cols = 2
            plt.figure(figsize=(10, 3))
            plt.subplot(rows, cols, 1); plt.hist(ades, bins=50); plt.title('ADE distribution'); plt.xlabel('meters')
            plt.subplot(rows, cols, 2); plt.hist(fdes, bins=50); plt.title('FDE distribution'); plt.xlabel('meters')
            plt.tight_layout()
            plt.savefig("err_distribution.png", dpi=150)
            plt.show()
        except Exception as e:
            print(f"(Optional histogram skipped: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True,
                        help="Path with maps/, samples/, sweeps/, v1.0-mini/ ...")
    parser.add_argument("--version", type=str, default="mini",
                        choices=["mini", "trainval", "test"],
                        help="nuScenes table version suffix (v1.0-<version>)")
    parser.add_argument("--seconds_past", type=float, default=2.0)
    parser.add_argument("--seconds_future", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.5, help="Keyframe interval (nuScenes uses 0.5s)")
    parser.add_argument("--max_pairs", type=int, default=2000,
                        help="Limit for quick runs (set 0 for all possible pairs)")
    parser.add_argument("--histogram", action="store_true", help="Show ADE/FDE histograms")
    # add args
    parser.add_argument("--save_csv", type=str, default="", help="Path to save per-pair ADE/FDE CSV")
    parser.add_argument("--save_viz_dir", type=str, default="", help="Dir to save a few GT vs pred plots")
    parser.add_argument("--viz_every", type=int, default=200, help="Save 1 plot every N samples")
    args = parser.parse_args()
    main(args)

    # python cv_mini.py --dataroot "./data/nuscenes" --version mini --seconds_past 2.0 --seconds_future 6.0 --dt 0.5 --max_pairs 2000 --histogram --save_csv results_cv_mini.csv --save_viz_dir figs_cv --viz_every 300
