import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

z = torch.rand(3, 5, requires_grad=True)
y = torch.randint(5, (3,)).long()

print(y)

y_pred = F.softmax(z, dim=1)

y_one_hot = torch.zeros_like(y_pred) 
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
print(y_one_hot)