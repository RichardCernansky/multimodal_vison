import os, pickle, argparse, numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion.quaternion import Quaternion

def quat_yaw(q):
    return Quaternion(q).yaw_pitch_roll[0]

def to_local(xs, origin_xy, yaw):
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s],[s, c]], dtype=np.float32)
    return ((xs - origin_xy[None,:]) @ R.T).astype(np.float32)

def collect_instance_tracks(nusc, split_scenes):
    # collect samples from chosen scenes (or all)
    scene_tokens = [s['token'] for s in nusc.scene if s['name'] in split_scenes] if split_scenes else [s['token'] for s in nusc.scene]
    sample_tokens = []
    for st in scene_tokens:
        sd = nusc.get('scene', st)
        tok = sd['first_sample_token']
        while tok:
            sample_tokens.append(tok)
            tok = nusc.get('sample', tok)['next']
    # group annotation tokens by instance
    inst2annos = {}
    for st in sample_tokens:
        s = nusc.get('sample', st)
        for ann_tok in s['anns']:
            ann = nusc.get('sample_annotation', ann_tok)
            inst2annos.setdefault(ann['instance_token'], []).append(ann_tok)
    # order annotations per instance (prev/next or timestamp)
    for k in list(inst2annos.keys()):
        seq = inst2annos[k]
        seq_sorted, ann_set = [], set(seq)
        heads = [t for t in seq if nusc.get('sample_annotation', t)['prev'] == '']
        for h in heads:
            t = h
            while t and t != '' and t in ann_set:
                seq_sorted.append(t)
                t = nusc.get('sample_annotation', t)['next']
        if not seq_sorted:
            seq_sorted = sorted(seq, key=lambda t: nusc.get('sample', nusc.get('sample_annotation', t)['sample_token'])['timestamp'])
        inst2annos[k] = seq_sorted
    return inst2annos

def build_pairs(nusc, Tp, Tf, include_classes, stride=1,
                split_scenes=None, max_pairs=None,
                bev_out=None, bev_size=256, bev_extent=30.0,
                bev_mode="toy_point", bev_channels=1):
    """
    Returns list of dicts per window:
      past_xy [Tp,2], future_xy [Tf,2],
      optionally: bev_seq_paths [Tp] (each saved .npy is [C,H,W])

    bev_mode:
      - "toy_point": draw only the current past point at step t
      - "toy_cumulative": draw all past points 0..t
    """
    inst2annos = collect_instance_tracks(nusc, split_scenes)
    pairs, saved = [], 0

    H = W = bev_size
    s = (H-1)/(2*bev_extent)

    def to_px(xy):
        x, y = xy[:,0], xy[:,1]
        u = (W/2 + x*s).round().astype(int)
        v = (H/2 - y*s).round().astype(int)
        m = (u>=0)&(u<W)&(v>=0)&(v<H)
        return u[m], v[m]

    for inst, seq in inst2annos.items():
        if len(seq) < Tp + Tf:
            continue

        for i in range(Tp, len(seq) - Tf, stride):
            ann_c = nusc.get('sample_annotation', seq[i])
            if include_classes and ann_c['category_name'].split('.')[0] not in include_classes:
                continue

            # collect global XY + yaw
            window = seq[i - Tp : i + Tf]
            trans, yaws = [], []
            for t in window:
                a = nusc.get('sample_annotation', t)
                trans.append(np.array(a['translation'][:2], dtype=np.float32))
                yaws.append(quat_yaw(a['rotation']))
            trans = np.stack(trans)  # [Tp+Tf,2]

            # localize to ego at last past step
            yaw0 = yaws[Tp-1]
            origin = trans[Tp-1]
            loc = to_local(trans, origin, yaw0)
            past_xy, future_xy = loc[:Tp], loc[Tp:]

            item = {"past_xy": past_xy, "future_xy": future_xy}

            # sequential BEVs (toy) — one per past step
            if bev_out is not None:
                os.makedirs(bev_out, exist_ok=True)
                if bev_channels != 1:
                    raise NotImplementedError("Template writes 1-channel toy BEVs; extend for multi-channel.")
                bev_paths = []
                cum_img = np.zeros((H, W), np.float32)
                for t in range(Tp):
                    img = cum_img if bev_mode == "toy_cumulative" else np.zeros((H, W), np.float32)
                    u, v = to_px(past_xy[t:t+1])
                    if u.size > 0:
                        img[v, u] = 1.0
                    arr = img[None, ...].astype(np.float32)  # [C,H,W], C=1
                    p = os.path.join(bev_out, f"{inst}_{seq[i]}_{t}.npy")
                    np.save(p, arr)
                    bev_paths.append(p)
                    if bev_mode == "toy_cumulative":
                        cum_img = img
                item["bev_seq_paths"] = bev_paths

            pairs.append(item)
            saved += 1
            if max_pairs and saved >= max_pairs:
                return pairs
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--Tp", type=int, default=8)
    ap.add_argument("--Tf", type=int, default=12)
    ap.add_argument("--out_pkl", default="mini_sample_instance_pairs_seq.pkl")
    ap.add_argument("--classes", nargs="*", default=["vehicle","human"],
                    help='Use "all" to include every class, or list like: vehicle human')
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--split_scenes", nargs="*", default=None)
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--bev_out", default=None)
    ap.add_argument("--bev_size", type=int, default=256)
    ap.add_argument("--bev_extent_m", type=float, default=30.0)
    ap.add_argument("--bev_mode", choices=["toy_point","toy_cumulative"], default="toy_point")
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    include_classes = None if (len(args.classes)==1 and args.classes[0].lower()=="all") else set(args.classes)

    pairs = build_pairs(
        nusc=nusc, Tp=args.Tp, Tf=args.Tf,
        include_classes=include_classes, stride=args.stride,
        split_scenes=args.split_scenes, max_pairs=args.max_pairs,
        bev_out=args.bev_out, bev_size=args.bev_size, bev_extent=args.bev_extent_m,
        bev_mode=args.bev_mode,
    )

    with open(args.out_pkl, "wb") as f:
        pickle.dump(pairs, f)
    print(f"Saved {len(pairs)} pairs -> {args.out_pkl}")

if __name__ == "__main__":
    main()
