# tiny_mtp.py
from dataclasses import dataclass
import argparse, pickle, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from dataset_mtp import MTPTrajDataset

@torch.no_grad()
def minade_minfde(preds: torch.Tensor, gt: torch.Tensor):
    errs = torch.linalg.vector_norm(preds - gt.unsqueeze(1), dim=-1) # shape [B, K, H]
    min_ade = errs.mean(-1).min(dim=1).values.mean().item()
    min_fde = errs[..., -1].min(dim=1).values.mean().item()
    return min_ade, min_fde

@torch.no_grad()
def best_mode_idx(preds: torch.Tensor, gt: torch.Tensor, by: str = "fde"):
    errs = torch.linalg.vector_norm(preds - gt.unsqueeze(1), dim=-1) #shape [B, 1, H, 2], so it can broadcast against [B, K, H, 2] ->  error across take the L2 norm across x/y → shape [B, K, H] (for every point)
    score = errs.mean(-1) if by == "ade" else errs[..., -1] # (tensor indexing) take the mean error across the horizon (ADE) or get the error- last element from the last axis (-1) (FDE)
    return score.argmin(dim=1) # get indexof the minimum error mode

def huber(x: torch.Tensor, delta: float = 1.0):
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * ax * ax, delta * (ax - 0.5 * delta))

def mtp_loss(preds: torch.Tensor, logits: torch.Tensor, gt: torch.Tensor, beta: float = 1.0, pick: str = "fde"):
    B, K, H, _ = preds.shape # preds shape = [B, K, H, 2]
    with torch.no_grad():
        k_best = best_mode_idx(preds, gt, by=pick) # get the index of best mode (for the whole batch), shape [B]
    idx = k_best.view(B, 1, 1, 1).expand(B, 1, H, 2) # expand to pred dimesnsion to extract the best K (mode)
    best_pred = preds.gather(1, idx).squeeze(1) # squeeze to -> [B, H, 2]
    reg = huber(best_pred - gt).mean() # Huber on (pred − gt), averaged over all elements.
    ce = F.cross_entropy(logits, k_best) # Cross-entropy between per-mode logits and the “label” k_best.
    return reg + beta * ce, reg.item(), ce.item()


class MLP(nn.Module):
    """
    Forward:
    ⟦B,din⟧ --Linear(din→hidden)--> ⟦B,hidden⟧ --ReLU--> ⟦B,hidden⟧
    --( Linear(hidden→hidden) · ReLU · Dropout )^(n_layers-1)--> ⟦B,hidden⟧
    --Linear(hidden→dout)--> ⟦B,dout⟧
    """
    def __init__(self, din, hidden, dout, nlayers=2, pdrop=0.0):
        super().__init__()
        layers = [nn.Linear(din, hidden), nn.ReLU()]
        for _ in range(nlayers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            if pdrop > 0:
                layers += [nn.Dropout(pdrop)]
        layers += [nn.Linear(hidden, dout)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class TinyBEVEncoder(nn.Module):
    """
    Forward (with shapes):
    ⟦B,1,H,W⟧
    --Conv2d(1→16,3,p1)--> ⟦B,16,H,W⟧ --ReLU--> ⟦B,16,H,W⟧
    --MaxPool2d(2)--> ⟦B,16,H/2,W/2⟧
    --Conv2d(16→32,3,p1)--> ⟦B,32,H/2,W/2⟧ --ReLU--> ⟦B,32,H/2,W/2⟧
    --MaxPool2d(2)--> ⟦B,32,H/4,W/4⟧
    --Conv2d(32→64,3,p1)--> ⟦B,64,H/4,W/4⟧ --ReLU--> ⟦B,64,H/4,W/4⟧
    --AdaptiveAvgPool2d(1×1)--> ⟦B,64,1,1⟧
    --View--> ⟦B,64⟧
    --Linear(64→out_dim)--> ⟦B,out_dim⟧
    """
    def __init__(self, out_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(64, out_dim)
    def forward(self, bev: torch.Tensor):
        x = self.backbone(bev)
        x = x.view(x.shape[0], 64)
        return self.proj(x)

#MODELS
class TinyMTPPastOnly(nn.Module):
    def __init__(self, Tp, Tf, K=3, hidden=256, layers=2, pdrop=0.0):
        super().__init__()
        self.Tp, self.Tf, self.K = Tp, Tf, K
        din = Tp * 2
        dout = K * Tf * 2 + K
        self.mlp = MLP(din, hidden, dout, nlayers=layers, pdrop=pdrop)
    def forward(self, past_xy: torch.Tensor):
        B = past_xy.shape[0]
        x = past_xy.reshape(B, -1)
        out = self.mlp(x)
        coord = out[:, : self.K * self.Tf * 2]
        logits = out[:, self.K * self.Tf * 2:]
        trajs = coord.view(B, self.K, self.Tf, 2)
        return trajs, logits

class TinyMTPLidarFusion(nn.Module):
    """
    Inputs:
    past_xy ⟦B,Tp,2⟧
    bev     ⟦B,1,H,W⟧
    Pipeline:
    feat_past := View(past_xy → ⟦B, Tp*2⟧)
    feat_bev  := TinyBEVEncoder(bev)                  ⇒ ⟦B, bev_feat⟧
    x         := Concat(feat_past, feat_bev, dim=1)   ⇒ ⟦B, Tp*2 + bev_feat⟧
    out       := MLP(x)                                ⇒ ⟦B, K*Tf*2 + K⟧
    Split(out → coord:⟦B,K*Tf*2⟧ , logits:⟦B,K⟧)
    trajs     := View(coord → ⟦B,K,Tf,2⟧)

    Return:
    (trajs ⟦B,K,Tf,2⟧ , logits ⟦B,K⟧)
    """
    def __init__(self, Tp, Tf, K=3, hidden=256, bev_feat=64, layers=2, pdrop=0.0):
        super().__init__()
        self.Tp, self.Tf, self.K = Tp, Tf, K
        self.bev_enc = TinyBEVEncoder(out_dim=bev_feat)
        din = Tp * 2 + bev_feat
        dout = K * Tf * 2 + K
        self.mlp = MLP(din, hidden, dout, nlayers=layers, pdrop=pdrop)
    def forward(self, past_xy: torch.Tensor, bev: torch.Tensor):
        B = past_xy.shape[0]
        feat_past = past_xy.reshape(B, -1)
        feat_bev = self.bev_enc(bev)
        x = torch.cat([feat_past, feat_bev], dim=1)
        out = self.mlp(x)
        coord = out[:, : self.K * self.Tf * 2]
        logits = out[:, self.K * self.Tf * 2:]
        trajs = coord.view(B, self.K, self.Tf, 2)
        return trajs, logits

def pick_device(gpu: int):
    use_cuda = torch.cuda.is_available() and gpu >= 0
    return torch.device(f"cuda:{gpu}") if use_cuda else torch.device("cpu")

def detect_bev_key(sample: dict, explicit: str | None):
    if explicit:
        return explicit
    for k in ["bev", "bev_rast", "lidar_bev", "bev_tensor"]:
        if k in sample:
            return k
    raise ValueError("BEV tensor key not found in dataset item. Provide --bev_key or ensure item includes one of: bev, bev_rast, lidar_bev, bev_tensor.")

def unpack_batch(batch, bev_key: str | None):
    if isinstance(batch, dict):
        past = batch["past_xy"]
        fut = batch["future_xy"]
        bev = batch[bev_key] if bev_key is not None else None
        return past, fut, bev
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            return batch[0], batch[1], None
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
    raise RuntimeError("Unsupported batch structure from DataLoader.")

def main(args):
    torch.manual_seed(0)
    device = pick_device(args.gpu)

    dataset = MTPTrajDataset(
        pairs_pkl=args.pairs,
        bev_dir=getattr(args, "bev_dir", None),
        can_dir=getattr(args, "can_dir", None),
        mapfeat_dir=getattr(args, "mapfeat_dir", None),
    )

    n = len(dataset)
    n_train = max(1, int(0.9 * n))
    train_ds, val_ds = random_split(dataset, [n_train, n - n_train], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=args.bsz, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.bsz, shuffle=False)

    sample = dataset[0] if isinstance(dataset[0], dict) else {"past_xy": dataset[0][0], "future_xy": dataset[0][1]}
    Tp = sample["past_xy"].shape[0]
    Tf = sample["future_xy"].shape[0]

    if args.model == "past":
        model = TinyMTPPastOnly(Tp, Tf, K=args.K, hidden=args.hidden).to(device)
        bev_key = None
    else:
        bev_key = detect_bev_key(sample, args.bev_key)
        model = TinyMTPLidarFusion(Tp, Tf, K=args.K, hidden=args.hidden, bev_feat=args.bev_feat).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            past, fut, bev = unpack_batch(batch, bev_key)
            past, fut = past.to(device), fut.to(device)
            if bev_key is None:
                preds, logits = model(past)
            else:
                preds, logits = model(past, bev.to(device))
            loss, reg, ce = mtp_loss(preds, logits, fut, beta=args.beta, pick=args.pick)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss), reg=reg, ce=ce)

        model.eval()
        with torch.no_grad():
            ades, fdes = [], []
            for batch in val_loader:
                past, fut, bev = unpack_batch(batch, bev_key)
                past, fut = past.to(device), fut.to(device)
                if bev_key is None:
                    preds, logits = model(past)
                else:
                    preds, logits = model(past, bev.to(device))
                ade, fde = minade_minfde(preds, fut)
                ades.append(ade); fdes.append(fde)
        print(f"Val: minADE={np.mean(ades):.3f}  minFDE={np.mean(fdes):.3f}")

    torch.save({"state_dict": model.state_dict(), "Tp": Tp, "Tf": Tf, "K": args.K, "model": args.model}, args.out)
    print(f"Saved -> {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="mini_sample_instance_pairs.pkl")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--bev_feat", type=int, default=64)
    ap.add_argument("--bsz", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--pick", choices=["fde","ade"], default="fde")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="tiny_mtp.pth")
    ap.add_argument("--model", choices=["past","lidar"], default="past")
    ap.add_argument("--bev_dir", default=None)
    ap.add_argument("--can_dir", default=None)
    ap.add_argument("--mapfeat_dir", default=None)
    ap.add_argument("--bev_key", default=None)
    args = ap.parse_args()
    main(args)
