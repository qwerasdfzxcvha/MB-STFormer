import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from scipy.signal import butter, lfilter
import scipy.io
from scipy import signal
import re

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

class EEGdataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data_per_subject = []
        self.label_per_subject = []

        data_files = sorted(os.listdir(data_path), key=natural_key)
        label_files = sorted(os.listdir(label_path), key=natural_key)

        filter_bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 51)]

        for data_file, label_file in zip(data_files, label_files):
            data = scipy.io.loadmat(os.path.join(data_path, data_file))['data']  # (N, C, T)
            labels = scipy.io.loadmat(os.path.join(label_path, label_file))['perclos']
            labels = np.where(labels > 0.35, 1, 0).squeeze()

            data = data.reshape(-1, 2400, 25).transpose(0, 2, 1)  # (N, C, T)

            EEG_bank = []
            for low, high in filter_bands:
                b, a = signal.butter(5, [low, high], btype='bandpass', fs=300)
                filtered = signal.lfilter(b, a, data, axis=-1)
                EEG_bank.append(filtered)

            EEG_combined = np.concatenate(EEG_bank, axis=1)
            self.data_per_subject.append(torch.tensor(EEG_combined, dtype=torch.float32))
            self.label_per_subject.append(torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.data_per_subject)

    def __getitem__(self, idx):
        return self.data_per_subject[idx], self.label_per_subject[idx]
if __name__ == '__main__':
    data_path = r"F:\sll\dataset\VLA_VRW\lab\EEG_new"
    label_path = r"F:\sll\dataset\VLA_VRW\lab\perclos"

    dataset = EEGdataset(data_path, label_path)
    print(len(dataset))  # 打印数据集大小