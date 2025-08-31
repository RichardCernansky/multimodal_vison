# dataset_mtp.py
import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

class MTPTrajDataset(Dataset):
    """
    Generic dataset for multimodal trajectory prediction.
    Loads from prebuilt pickle of instance pairs, with optional extra features.
    Each item returns a dict with:
      - past_xy   [Tp,2]
      - future_xy [Tf,2]
      - bev       [1,H,W]   (if available)
      - can       [D]       (if available)
      - mapfeat   [F]       (if available)
    """

    def __init__(self, pairs_pkl, bev_dir=None, can_dir=None, mapfeat_dir=None, device="cpu"):
        """
        pairs_pkl : str  path to pickle with [{"past_xy":..., "future_xy":..., "sample_token":..., "instance_token":...}]
        bev_dir   : str  folder with BEV rasters saved as .npy (optional)
        can_dir   : str  folder with CAN signals per sample_token (optional)
        mapfeat_dir: str folder with map features per sample_token (optional)
        """
        self.items = pickle.load(open(pairs_pkl, "rb"))
        self.bev_dir = bev_dir
        self.can_dir = can_dir
        self.mapfeat_dir = mapfeat_dir
        self.device = device

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]

        past_xy   = torch.from_numpy(it["past_xy"]).float()     # [Tp,2]
        future_xy = torch.from_numpy(it["future_xy"]).float()   # [Tf,2]

        sample_token = it.get("sample_token", None)

        # load extras if directories provided
        bev = None
        if self.bev_dir and sample_token:
            bev_path = os.path.join(self.bev_dir, f"{sample_token}.npy")
            bev = torch.from_numpy(np.load(bev_path)).float()   # [1,H,W]

        can = None
        if self.can_dir and sample_token:
            can_path = os.path.join(self.can_dir, f"{sample_token}.npy")
            can = torch.from_numpy(np.load(can_path)).float()   # [D]

        mapfeat = None
        if self.mapfeat_dir and sample_token:
            mf_path = os.path.join(self.mapfeat_dir, f"{sample_token}.npy")
            mapfeat = torch.from_numpy(np.load(mf_path)).float()   # [F]

        return {
            "past_xy": past_xy,
            "future_xy": future_xy,
            "bev": bev,
            "can": can,
            "mapfeat": mapfeat,
            "sample_token": sample_token,
            "instance_token": it.get("instance_token", None)
        }
