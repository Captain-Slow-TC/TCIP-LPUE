import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

def msw_to_onehot_sequence(msw_list):

    def get_category(msw):
        if msw < 10.8:
            return 0
        elif msw < 17.1:
            return 1
        elif msw < 24.4:
            return 2
        elif msw < 32.6:
            return 3
        elif msw < 41.4:
            return 4
        elif msw < 50.9:
            return 5
        elif msw < 61.1:
            return 6
        elif msw < 71.4:
            return 7
        else:
            return 8
    
    onehot = np.zeros((9, 5), dtype=np.float32)
    for t_idx, msw in enumerate(msw_list):
        cat = get_category(msw)
        onehot[cat, t_idx] = 1.0
    return onehot

class TyphoonLifecycleDataset(Dataset):

    def __init__(self, data_dir, tc_folders):

        self.data_dir = data_dir
        self.samples = []

        for folder in tc_folders:
            folder_path = os.path.join(data_dir, folder)
            npy_files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
            self.samples.extend(npy_files)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
            x_img (Tensor):  (10, 400, 400)。
            x_ERA (Tensor):  (3, 5, 400, 400)。
            x_onehot (Tensor):  (9, 5)。
            y_true (Tensor):  (1,)。
        """
        file_path = self.samples[idx]
        try:

            data = np.load(file_path, allow_pickle=True).item()

            x_img = torch.tensor(data['tc_img'], dtype=torch.float32)

            x_ERA = torch.tensor(data['tc_era'], dtype=torch.float32).view(3, 5, 400, 400)

            msw_hist = data['tc_intensity']
            x_onehot = torch.tensor(msw_to_onehot_sequence(msw_hist), dtype=torch.float32)
            
            target_msw = float(data['target'][0])
            y_true = torch.tensor([target_msw], dtype=torch.float32)
            
        except Exception as e:
            x_img = torch.zeros((10, 400, 400), dtype=torch.float32)
            x_ERA = torch.zeros((3, 5, 400, 400), dtype=torch.float32)
            x_onehot = torch.zeros((9, 5), dtype=torch.float32)
            y_true = torch.zeros((1,), dtype=torch.float32)
            
        return x_img, x_ERA, x_onehot, y_true