import torch
from torch.utils.data import Dataset

class CustomData(Dataset):
  def __init__(self):
    self.x_data = [[10.0], [20.0], [30.0], [40.0], [50.0], [60.0], [70.0], [80.0], [90.0], [100.0]]
    self.y_data = [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y
