import torch, torch.nn as nn

# ---------- per-frame encoder (supports C input channels) ----------
class TinyPerFrameBEV(nn.Module):
    def __init__(self, c_in=1, out_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(c_in,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),   nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)), # [B, 64, 1, 1] for FC with 64 features
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):              # x: [B,C,H,W]
        y = self.backbone(x)           # [B,64,1,1] 
        y = y.view(y.size(0), 64)
        return self.fc(y)              # [B,out_dim]

# ---------- GRU memory encoder ----------
class BEV_GRU_Encoder(nn.Module):
    #feat_dim (per-frame embedding size), hid (GRU state), layers (stacked GRUs).
    #Pros: simple, strong baseline. Cost: O(Tp·hid); grows linearly with Tp.
    def __init__(self, c_in=1, feat_dim=128, hid=128, layers=1):
        super().__init__()
        self.frame = TinyPerFrameBEV(c_in=c_in, out_dim=feat_dim)
        self.gru   = nn.GRU(input_size=feat_dim, hidden_size=hid, num_layers=layers, batch_first=True)

    def forward(self, bev_seq):        # [B,Tp,C,H,W]
        B,T,C,H,W = bev_seq.shape
        f = self.frame(bev_seq.view(B*T, C, H, W))   # [B*T,feat_dim]
        f = f.view(B, T, -1)                         # [B,Tp,feat_dim] Now you have a sequence of feature vectors per sample
        _, hT = self.gru(f)                          # [layers,B,hid] GRU rolls through time dimension (Tp steps), _ not interested in the ones before last
        return hT[-1]                                # [B,hid]
    


# ---------- XMem-like attention memory encoder ----------
class TinyAttentionMemory(nn.Module):
    def __init__(self, c_in=1, feat_dim=128, mem_slots=6, key_dim=64, val_dim=128):
        super().__init__()
        self.frame = TinyPerFrameBEV(c_in=c_in, out_dim=feat_dim)
        self.q_proj = nn.Linear(feat_dim, key_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, key_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, val_dim, bias=False)
        self.mem_slots = mem_slots

    def _attend(self, q, K, V, mask):
        scores = torch.einsum("bid,bsd->bis", q, K) / (K.size(-1) ** 0.5)  # [B,1,S]
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn = scores.softmax(dim=-1)                                      # [B,1,S]
        ctx  = torch.bmm(attn, V)                                          # [B,1,Dv]
        return ctx.squeeze(1)                                              # [B,Dv]

    def forward(self, bev_seq):                                            # [B,Tp,C,H,W]
        B,T,C,H,W = bev_seq.shape
        Dk, Dv = self.q_proj.out_features, self.v_proj.out_features
        Kmem = torch.zeros(B, self.mem_slots, Dk, device=bev_seq.device)
        Vmem = torch.zeros(B, self.mem_slots, Dv, device=bev_seq.device)
        Mmask= torch.zeros(B, self.mem_slots, dtype=torch.bool, device=bev_seq.device)
        ptr  = torch.zeros(B, dtype=torch.long, device=bev_seq.device)

        summary = torch.zeros(B, Dv, device=bev_seq.device)
        for t in range(T):
            f = self.frame(bev_seq[:,t])           # [B,feat_dim]
            q = self.q_proj(f).unsqueeze(1)        # [B,1,Dk]
            k = self.k_proj(f).unsqueeze(1)
            v = self.v_proj(f).unsqueeze(1)

            if Mmask.any():
                summary = self._attend(q, Kmem, Vmem, Mmask)
            else:
                summary = self.v_proj(f)

            # FIFO write (could switch to reservoir/age-based)
            for b in range(B):
                j = ptr[b].item()
                Kmem[b,j] = k[b,0]; Vmem[b,j] = v[b,0]; Mmask[b,j] = True
                ptr[b] = (ptr[b] + 1) % self.mem_slots

        return summary   # [B,Dv]

# ---------- Full MTP model ----------
class MTP_Head(nn.Module):
    def __init__(self, Tp, Tf, K=3, past_feat=128, hidden=256, encoder="gru", c_in=1):
        super().__init__()
        self.Tf, self.K = Tf, K
        self.past_mlp = nn.Sequential(nn.Linear(Tp*2, past_feat), nn.ReLU())
        if encoder == "gru":
            self.bev_enc = BEV_GRU_Encoder(c_in=c_in, feat_dim=128, hid=128)
            bev_out = 128
        elif encoder == "mem":
            self.bev_enc = TinyAttentionMemory(c_in=c_in, feat_dim=128, mem_slots=6)
            bev_out = 128
        else:
            raise ValueError("encoder must be 'gru' or 'mem'")
        self.head = nn.Sequential(
            nn.Linear(past_feat + bev_out, hidden), nn.ReLU(),
            nn.Linear(hidden, K*Tf*2 + K)
        )

    def forward(self, past_xy, bev_seq):
        B = past_xy.size(0)
        p = self.past_mlp(past_xy.view(B,-1))
        b = self.bev_enc(bev_seq)
        out = self.head(torch.cat([p,b], dim=1))
        coord = out[:, : self.K*self.Tf*2].view(B, self.K, self.Tf, 2)
        logits= out[:, self.K*self.Tf*2:]
        return coord, logits
