from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
import torch
class Custom_Data_loader(Dataset):
    def __init__(self):
        y = np.loadtxt('data-diabetes.csv', delimiter=',', dtype=np.float32)
        self.len = y.shape[0]
        self.x_data = torch.from_numpy(y[:, 0:-1])
        self.y_data = torch.from_numpy(y[:, [-1]])

def __len__(self):
    return self.len

def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]


dataset = Custom_Data_loader()
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)