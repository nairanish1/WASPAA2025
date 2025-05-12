import numpy as np
import pandas as pd
import torch
import scipy.io as sio
from torch.utils.data import Dataset

# after preprocessing, CSV is already 93x37; .mat has 96 subjects
VALID_SUBJ = [*range(0, 17), *range(18, 78), *range(79, 91), *range(92, 96)]  # 93 total

class HUTUBS(Dataset):
    """
    Leave-one-out HUTUBS dataset (93 subjects after filtering).

    Each sample = (head_anthro, ear_anthro, sht_flat, hrtf_flat,
            subj_idx, freq_idx, ear_side)
    """

    def __init__(self, args, val: bool):
        super().__init__()
        self.val = val

        # 1) anthropometry (already pruned CSV)
        df = pd.read_csv(args.anthro_mat_path)
        anthro = df.values.astype(np.float32)        # [93, 37]
        assert 0 <= args.val_idx < anthro.shape[0], "val_idx must be 0..92"
        hold = args.val_idx

        # split train / val
        meas_val = anthro[[hold]]                   # [1, 37]
        meas_tr  = np.delete(anthro, hold, axis=0)   # [92, 37]

        # normalization
        if args.norm_anthro:
            mu, sd = meas_tr.mean(0, keepdims=True), meas_tr.std(0, keepdims=True)
            if args.anthro_norm_method == 'standard':
                meas_tr = (meas_tr - mu) / sd
                meas_val = (meas_val - mu) / sd
            else:  # chun2017
                meas_tr  = 1.0 / (1.0 + np.exp((meas_tr  - mu) / sd))
                meas_val = 1.0 / (1.0 + np.exp((meas_val - mu) / sd))

        # split head / ear-L / ear-R
        self.head_tr, self.dl_tr, self.dr_tr = (
            meas_tr[:, :13], meas_tr[:, 13:25], meas_tr[:, 25:37]
        )
        self.head_va, self.dl_va, self.dr_va = (
            meas_val[:, :13], meas_val[:, 13:25], meas_val[:, 25:37]
        )

        # 2) load acoustic .mat and filter subjects
        mat       = sio.loadmat(args.hrtf_SHT_mat_path)
        freq_idx  = mat['freq_logind'].squeeze()
        hrtf_96   = mat['hrtf_freq_allDB'].astype(np.float32)  # [96, F, H, W, E]
        sht_96    = mat['hrtf_SHT_dBmat'].astype(np.float32)  # [96, F, H, W, E]

        # keep only VALID_SUBJ
        hrtf_93 = hrtf_96[VALID_SUBJ]  # [93, F, H, W, E]
        sht_93  = sht_96 [VALID_SUBJ]

        # select freq bins
        self.hrtf_all = hrtf_93[:, freq_idx, ...]  # [93, F, H, W, E]
        self.sht_all  = sht_93 [:, freq_idx, ...]

        # split train / val along subject
        self.hrtf_va = self.hrtf_all[[hold]]
        self.sht_va  = self.sht_all [[hold]]
        self.hrtf_tr = np.delete(self.hrtf_all, hold, axis=0)
        self.sht_tr  = np.delete(self.sht_all,  hold, axis=0)

    def __len__(self):
        hrtf = self.hrtf_va if self.val else self.hrtf_tr
        S, F = hrtf.shape[:2]
        return S * F * 2  # two ears

    def __getitem__(self, idx: int):
        # choose split
        head_arr, dl_arr, dr_arr = (
            (self.head_va, self.dl_va, self.dr_va) if self.val
            else (self.head_tr, self.dl_tr, self.dr_tr)
        )
        hrtf_arr, sht_arr = (
            (self.hrtf_va, self.sht_va) if self.val
            else (self.hrtf_tr, self.sht_tr)
        )

        S, F = hrtf_arr.shape[:2]
        subj = idx // (F * 2)
        freq = (idx // 2) % F
        ear  = idx % 2  # 0=left, 1=right

        # anthropometry
        head_vec = head_arr[subj]
        ear_vec  = dl_arr[subj] if ear == 0 else dr_arr[subj]

        # acoustic: take the slice and pick ear dimension, then flatten
        # hrtf_arr[subj,freq] shape: [H, W, E]
        raw_hrtf = hrtf_arr[subj, freq]  # [H, W, E]
        raw_sht  = sht_arr [subj, freq]  # [H, W, E]

        # select the requested ear and flatten spatial grid
        hrtf_slice = raw_hrtf[..., ear].reshape(-1)  # [H*W]
        sht_slice  = raw_sht [..., ear].reshape(-1)

        return (
            torch.from_numpy(head_vec),
            torch.from_numpy(ear_vec),
            torch.from_numpy(sht_slice),   # target vector [C]
            torch.from_numpy(hrtf_slice),  # auxiliary [C]
            subj, freq, ear
        )
