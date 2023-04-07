import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

#... 대충 데이터

#모델 생성 클래스
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = Model()

#...대충 학습