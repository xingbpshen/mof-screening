import torch
import numpy as np
from torch.utils.data import Dataset


class MOFDataset(Dataset):
    def __init__(self, x_path, wc, sel, x_stats, wc_stats, sel_stats, input_normalization=False,
                 target_normalization=False, weights_wc=None, weights_sel=None):
        self.x_path = x_path
        self.wc = wc
        self.sel = sel
        self.x_stats = x_stats
        self.wc_stats = wc_stats
        self.sel_stats = sel_stats
        self.input_normalization = input_normalization
        self.target_normalization = target_normalization
        self.weights_wc = weights_wc
        self.weights_sel = weights_sel

    def __len__(self):
        return len(self.x_path)

    def __getitem__(self, index):
        x, wc, sel = np.load(self.x_path[index]), self.wc[index], self.sel[index]
        # Convert x to float32
        x = x.astype(np.float32)
        if self.weights_wc is not None:
            wc_weight = float(self.weights_wc[index])
        else:
            wc_weight = 1.0
        if self.weights_sel is not None:
            sel_weight = float(self.weights_sel[index])
        else:
            sel_weight = 1.0
        if self.input_normalization:
            x = (x - self.x_stats[0]) / self.x_stats[1]
        if self.target_normalization:
            wc = (wc - self.wc_stats[0]) / self.wc_stats[1]
            sel = (sel - self.sel_stats[0]) / self.sel_stats[1]
        return torch.from_numpy(x), torch.tensor(wc), torch.tensor(sel), wc_weight, sel_weight
