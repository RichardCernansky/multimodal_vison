# train_tiny_mtp.py
import argparse, pickle, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from dataset_mtp import MTPTrajDataset

from tqdm import tqdm


def minade_minfde(preds, gt):  # preds: [B,K,H,2], gt: [B,H,2]
    errs = torch.linalg.vector_norm(preds - gt.unsqueeze(1), dim=-1)  # [B,K,H]
    ade = errs.mean(-1).min(dim=1).values.mean().item()
    fde = errs[:,:,-1].min(dim=1).values.mean().item()
    return ade, fde


class TinyMTP(nn.Module):
    def __init__(self, Tp, Tf, K=3, hidden=256):
        super().__init__()
        Din = Tp*2
        Dout = K*Tf*2 + K  # trajectories + logits
        self.net = nn.Sequential(
            nn.Linear(Din, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, Dout)
        )
        self.Tf, self.K = Tf, K
    def forward(self, past_xy):  # past_xy: [B,Tp,2]
        x = past_xy.reshape(past_xy.shape[0], -1)
        out = self.net(x)
        trajs = out[:, : self.K*self.Tf*2].reshape(-1, self.K, self.Tf, 2)
        logits = out[:, self.K*self.Tf*2 :]  # [B,K]
        return trajs, logits

def best_mode_idx(preds, gt):  # preds: [B,K,H,2]
    errs = torch.linalg.vector_norm(preds - gt.unsqueeze(1), dim=-1)  # [B,K,H]
    return errs[:,:,-1].argmin(dim=1)  # by FDE

def huber(x, delta=1.0):
    absx = torch.abs(x)
    return torch.where(absx <= delta, 0.5*absx**2, delta*(absx - 0.5*delta))

def mtp_loss(preds, logits, gt, beta=1.0):
    """
    preds: [B,K,H,2], logits: [B,K], gt: [B,H,2]
    Choose best mode per sample (by final-point L2), regress with Huber, and CE for confidences.
    """
    B,K,H,_ = preds.shape
    with torch.no_grad():
        k_best = best_mode_idx(preds, gt)  # [B]
    # regression loss (best mode)
    idx = k_best[:,None,None,None].expand(B,1,H,2)
    best_pred = preds.gather(1, idx).squeeze(1)  # [B,H,2]
    reg = huber(best_pred - gt).mean()

    # classification (encourage high prob on best mode)
    ce = nn.CrossEntropyLoss()(logits, k_best)

    return reg + beta*ce, reg.item(), ce.item()

def main(args):
    # pick device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    # load dataset (past_xy + future_xy are always in the pickle,
    # optional bev_dir/can_dir/mapfeat_dir if you have them)
    dataset = MTPTrajDataset(
        pairs_pkl=args.pairs,
        bev_dir=args.bev_dir if hasattr(args, "bev_dir") else None,
        can_dir=args.can_dir if hasattr(args, "can_dir") else None,
        mapfeat_dir=args.mapfeat_dir if hasattr(args, "mapfeat_dir") else None,
    )

    # split train/val (90/10)
    n = len(dataset)
    n_train = int(0.9 * n)
    train_ds, val_ds = random_split(dataset, [n_train, n - n_train], generator=torch.Generator().manual_seed(0))

    # build loaders
    train_loader = DataLoader(train_ds, batch_size=args.bsz, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.bsz, shuffle=False)
    
    sample = dataset[0]   # one item = dict
    Tp = sample["past_xy"].shape[0]
    Tf = sample["future_xy"].shape[0]

    model = TinyMTP(Tp, Tf, K=args.K, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    for epoch in range(args.epochs):
        model.train()
        for past, fut in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            past, fut = past.to(device), fut.to(device)
            preds, logits = model(past)
            loss, reg, ce = mtp_loss(preds, logits, fut, beta=args.beta)
            opt.zero_grad(); loss.backward(); opt.step()

        # validation
        model.eval()
        with torch.no_grad():
            all_ade, all_fde = [], []
            for past, fut in val_loader:
                past, fut = past.to(device), fut.to(device)
                preds, logits = model(past)
                ade, fde = minade_minfde(preds, fut)
                all_ade.append(ade); all_fde.append(fde)
        print(f"Val: minADE={np.mean(all_ade):.3f}  minFDE={np.mean(all_fde):.3f}")

    # save
    # torch.save(model.state_dict(), args.out)
    # print("Saved model to", args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="mini_sample_instance_pairs.pkl")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--bsz", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0, help="weight for CE (mode probs)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="tiny_mtp.pth")
    args = ap.parse_args()
    main(args)



# 1) build pairs if not done
# python mtp_mini.py
