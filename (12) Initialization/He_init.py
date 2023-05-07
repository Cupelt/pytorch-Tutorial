import torch
import torch.nn as nn
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear_1 = nn.Linear(1024, 1024, bias=False)
        self.linear_2 = nn.Linear(1024, 512, bias=False)
        self.linear_3 = nn.Linear(512, 10, bias=True)
        self.initialize_weights()
          
    def forward(self, x):
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        x = torch.relu(x)
        x = self.linear_3(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

model = Model()
print(model.linear_1.weight.data)