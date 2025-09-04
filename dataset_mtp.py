import pickle, numpy as np, torch
from torch.utils.data import Dataset

class MTPTrajSeqPathsDataset(Dataset):
    """
    Loads:
      - past_xy: [Tp,2]
      - future_xy: [Tf,2]
      - If present: bev_seq_paths -> loads [Tp,C,H,W]
      - Else: builds toy BEV seq from past_xy (C=1)

    Notes:
      - Expects all saved BEV frames to have identical shape [C,H,W].
      - If your exporter later writes multi-channel BEV per t, this loader still works.
    """
    def __init__(self, pairs_pkl, H=128, W=128, extent_m=30.0):
        self.items = pickle.load(open(pairs_pkl, "rb"))
        self.H, self.W = H, W
        self.s = (H - 1) / (2 * extent_m)

    def __len__(self):
        return len(self.items)

    def _px(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        u = (self.W / 2 + x * self.s).round().astype(int)
        v = (self.H / 2 - y * self.s).round().astype(int)
        m = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        return u[m], v[m]

    def __getitem__(self, i):
        it = self.items[i]
        past = torch.tensor(it["past_xy"], dtype=torch.float32)   # [Tp,2]
        fut  = torch.tensor(it["future_xy"], dtype=torch.float32) # [Tf,2]
        Tp = past.shape[0]

        # Preferred path: load stored sequential BEVs
        if "bev_seq_paths" in it:
            frames = []
            for p in it["bev_seq_paths"]:
                arr = np.load(p)              # [C,H,W]
                frames.append(torch.from_numpy(arr).float())
            bev_seq = torch.stack(frames, dim=0)  # [Tp,C,H,W]
            return {"past_xy": past, "future_xy": fut, "bev_seq": bev_seq}

        # Fallback: toy BEV sequence built from past_xy (C=1)
        bev_seq = []
        for t in range(Tp):
            img = np.zeros((self.H, self.W), np.float32)
            u, v = self._px(past[t:t+1].numpy())
            if u.size > 0: img[v, u] = 1.0
            bev_seq.append(torch.from_numpy(img[None, ...]))  # [1,H,W]
        bev_seq = torch.stack(bev_seq, dim=0)                 # [Tp,1,H,W]
        return {"past_xy": past, "future_xy": fut, "bev_seq": bev_seq}
