import torch
from torch.utils.data import Dataset

class data(Dataset):
  def __init__(self):
    self.x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
    self.y_data = [[0], [0], [0], [1], [1], [1]]

  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y
