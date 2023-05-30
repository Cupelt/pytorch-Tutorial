import torch
import torch.nn as nn

input_size = 5 # 입력의 크기
hidden_size = 8 # 은닉 상태의 크기

# (batch_size, time_steps, input_size)
inputs = torch.Tensor(1, 10, 5)

cell = nn.LSTM(input_size, hidden_size, batch_first=True)

outputs, _status = cell(inputs)

print(outputs.shape) # 모든 time-step의 hidden_state
print(_status[0].shape) # 최종 time-step의 hidden_state
print(_status[1].shape) # 최종 time-step의 cell_state