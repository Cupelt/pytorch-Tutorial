import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 

nb_epochs = 20
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    # print(batch_idx)
    # print(samples)
    x_train, y_train = samples
    # H(x) 계산
    y_pred = model(x_train)

    # loss 계산
    loss = F.mse_loss(y_pred, y_train)

    # loss로 H(x) 계산
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Batch {}/{} loss: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        loss.item()
        ))

prediction = torch.FloatTensor([[73,  80,  75]])
print(model(x_result))