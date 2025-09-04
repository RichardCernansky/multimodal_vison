import argparse, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset_mtp_seq_paths import MTPTrajSeqPathsDataset
from models_seq import MTP_Head

@torch.no_grad()
def minade_minfde(preds, gt):
    errs = torch.linalg.vector_norm(preds - gt.unsqueeze(1), dim=-1)
    min_ade = errs.mean(-1).min(dim=1).values.mean().item()
    min_fde = errs[...,-1].min(dim=1).values.mean().item()
    return min_ade, min_fde

@torch.no_grad()
def best_mode_idx(preds, gt, by="fde"):
    errs = torch.linalg.vector_norm(preds - gt.unsqueeze(1), dim=-1)
    score = errs.mean(-1) if by=="ade" else errs[...,-1]
    return score.argmin(dim=1)

def huber(x, delta=1.0):
    ax = x.abs()
    return torch.where(ax<=delta, 0.5*ax*ax, delta*(ax-0.5*delta))

def mtp_loss(preds, logits, gt, beta=1.0, pick="fde"):
    B,K,H,_ = preds.shape
    with torch.no_grad():
        k_best = best_mode_idx(preds, gt, by=pick)
    idx = k_best.view(B,1,1,1).expand(B,1,H,2)
    best_pred = preds.gather(1, idx).squeeze(1)
    reg = huber(best_pred-gt).mean()
    ce  = F.cross_entropy(logits, k_best)
    return reg + beta*ce, reg.item(), ce.item()

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu>=0 else "cpu")

    ds = MTPTrajSeqPathsDataset(args.pairs, H=args.H, W=args.W, extent_m=args.extent_m)
    n = len(ds); n_train = max(1, int(0.9*n))
    train_ds, val_ds = random_split(ds, [n_train, n-n_train], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=args.bsz, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.bsz, shuffle=False)

    sample = ds[0]
    Tp = sample["past_xy"].shape[0]
    Tf = sample["future_xy"].shape[0]
    # infer input channels C from data
    C = sample["bev_seq"].shape[1]  # [Tp,C,H,W]

    model = MTP_Head(Tp, Tf, K=args.K, hidden=args.hidden, encoder=args.encoder, c_in=C).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            past  = batch["past_xy"].to(device)
            fut   = batch["future_xy"].to(device)
            bev_s = batch["bev_seq"].to(device)
            preds, logits = model(past, bev_s)
            loss, reg, ce = mtp_loss(preds, logits, fut, beta=args.beta, pick=args.pick)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss), reg=reg, ce=ce)

        model.eval(); ades,fdes=[],[]
        with torch.no_grad():
            for batch in val_loader:
                past  = batch["past_xy"].to(device)
                fut   = batch["future_xy"].to(device)
                bev_s = batch["bev_seq"].to(device)
                preds, logits = model(past, bev_s)
                ade,fde = minade_minfde(preds, fut)
                ades.append(ade); fdes.append(fde)
        print(f"Val: minADE={np.mean(ades):.3f}  minFDE={np.mean(fdes):.3f}")

    torch.save({"state_dict": model.state_dict(), "Tp":Tp, "Tf":Tf, "K":args.K,
                "encoder":args.encoder, "C":C}, args.out)
    print(f"Saved -> {args.out}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="mini_sample_instance_pairs_seq.pkl")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--pick", choices=["fde","ade"], default="fde")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="mtp_seq_mem.pth")
    ap.add_argument("--encoder", choices=["gru","mem"], default="gru")
    # raster/grid params (must match exporter or fallback toy settings)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--extent_m", type=float, default=30.0)
    args = ap.parse_args()
    main(args)


# python train_seq.py --pairs mini_sample_instance_pairs_seq.pkl --encoder gru --epochs 5
# python train_seq.py --pairs mini_sample_instance_pairs_seq.pkl --encoder mem --epochs 5

