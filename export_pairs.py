# build_pairs.py
import argparse, pickle, numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper

def main(args):
    nusc = NuScenes(version=f'v1.0-{args.version}', dataroot=args.dataroot, verbose=False)
    helper = PredictHelper(nusc)

    dt = 0.5
    Tp = int(round(args.seconds_past / dt))
    Tf = int(round(args.seconds_future / dt))
    out = []

    for sample in tqdm(nusc.sample, desc="Collecting pairs"):
        s_tok = sample['token']
        for ann_tok in sample['anns']:
            ann = nusc.get('sample_annotation', ann_tok)
            inst = ann['instance_token']

            past = helper.get_past_for_agent(inst, s_tok, seconds=args.seconds_past,
                                             in_agent_frame=True, just_xy=True)
            fut  = helper.get_future_for_agent(inst, s_tok, seconds=args.seconds_future,
                                               in_agent_frame=True, just_xy=True)
            if past is None or fut is None: 
                continue
            if len(past) != Tp or len(fut) != Tf:
                continue

            out.append({
                "past_xy": np.asarray(past, np.float32),
                "future_xy": np.asarray(fut,  np.float32),
                "instance": ann['instance_token'],
                "sample": s_tok
            })
            if args.max_pairs and len(out) >= args.max_pairs:
                break
        if args.max_pairs and len(out) >= args.max_pairs:
            break

    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved {len(out)} pairs to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="mini", choices=["mini","trainval","test"])
    ap.add_argument("--seconds_past", type=float, default=2.0)
    ap.add_argument("--seconds_future", type=float, default=6.0)
    ap.add_argument("--max_pairs", type=int, default=10000)
    ap.add_argument("--out", default="mini_pairs.pkl")
    args = ap.parse_args()
    main(args)


# python export_pairs.py  --dataroot "./data/nuscenes" --version mini --max_pairs 15000 --out mini_pairs.pkl
