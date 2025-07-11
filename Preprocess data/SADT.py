import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal


class EEGdataset(Dataset):
    def __init__(self, data_path):
        self.data_per_subject = []  # 每个被试的数据
        self.label_per_subject = []  # 每个被试的标签

        filter_bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 51)]

        data_files = sorted(os.listdir(data_path))  # 每个文件一个被试
        for data_file in data_files:
            full_path = os.path.join(data_path, data_file)
            with open(full_path, 'rb') as f:
                sample = pickle.load(f)
                data = np.array(sample['data'])  # shape: [N, 1, C, T] or [N, C, T]
                label = np.array(sample['label']).astype(int)  # shape: [N, 1] or [N]

            data = np.squeeze(data, axis=1)  # 去掉中间无意义维度，变为 [N, C, T]
            label = np.squeeze(label)

            # 多频段滤波
            EEG_bank = []
            for band in filter_bands:
                b, a = signal.butter(N=5, Wn=[band[0], band[1]], btype='bandpass', fs=128)
                filtered = signal.lfilter(b, a, data, axis=-1)  # shape: [N, C, T]
                EEG_bank.append(filtered)

            # 将多个频段结果拼接到 channel 维度：shape => [N, F*C, T]
            EEG_combined = np.concatenate(EEG_bank, axis=1).astype(np.float32)
            EEG_tensor = torch.from_numpy(EEG_combined).float()
            label_tensor = torch.from_numpy(label).long()

            self.data_per_subject.append(EEG_tensor)
            self.label_per_subject.append(label_tensor)

    def __len__(self):
        # 返回总被试数（也即文件数）
        return len(self.data_per_subject)

    def __getitem__(self, idx):
        # 返回第 idx 个被试的数据和标签
        return self.data_per_subject[idx], self.label_per_subject[idx]


if __name__ == '__main__':
    data_path = "C:/Users/SLL/dataset/data_eeg_FATIG_FTG"
    # label_path = 'C:/Users/SLL/dataset/SEED_VIG/perclos_labels/'

    # 创建数据集
    dataset = EEGdataset(data_path)

