from torch.utils.data import Dataset
import torch
import numpy as np
import os
import scipy.io
from scipy import signal


class EEGdataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data_per_subject = []
        self.label_per_subject = []

        data_files = sorted(os.listdir(data_path))
        label_files = sorted(os.listdir(label_path))

        filter_bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 51)]

        for data_file, label_file in zip(data_files, label_files):
            # 加载数据与标签
            data = scipy.io.loadmat(os.path.join(data_path, data_file))['EEG']['data'][0][0]  # shape: (N, 1600, 17)
            labels = scipy.io.loadmat(os.path.join(label_path, label_file))['perclos']  # shape: (N, 1)
            labels = np.where(labels > 0.35, 1, 0).squeeze()  # shape: (N,)

            data = data.reshape(-1, 1600, 17)  # shape: (N, T, C)
            data = data.transpose(0, 2, 1)     # shape: (N, C, T)

            EEG_bank = []
            for low, high in filter_bands:
                b, a = signal.butter(N=5, Wn=[low, high], btype='bandpass', fs=200)
                filtered = signal.lfilter(b, a, data, axis=-1)  # shape: (N, C, T)
                EEG_bank.append(filtered)

            # 多频段拼接：沿 C 通道维拼接
            EEG_combined = np.concatenate(EEG_bank, axis=1)  # shape: (N, F×C, T)

            EEG_tensor = torch.tensor(EEG_combined, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.long)

            self.data_per_subject.append(EEG_tensor)
            self.label_per_subject.append(label_tensor)

    def __len__(self):
        return len(self.data_per_subject)

    def __getitem__(self, idx):
        return self.data_per_subject[idx], self.label_per_subject[idx]
if __name__ == '__main__':
    data_path = "C:/Users/SLL/dataset/SEED_VIG/Raw_Data/"
    label_path = 'C:/Users/SLL/dataset/SEED_VIG/perclos_labels/'

    dataset = EEGdataset(data_path, label_path)
    print(len(dataset))  # 打印数据集大小