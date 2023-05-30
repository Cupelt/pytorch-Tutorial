import numpy as np

timesteps = 2 # 시점의 수. NLP에서는 보통 문장의 길이가 된다.
input_size = 5 # 입력의 차원. NLP에서는 보통 단어 벡터의 차원이 된다.
hidden_size = 8 # 은닉 상태의 크기. 메모리 셀의 용량이다.

inputs = np.random.random((timesteps, input_size)) # 입력에 해당되는 2D 텐서

hidden_state_t = np.zeros((hidden_size,)) # 초기 은닉 상태는 0(벡터)로 초기화
cell_state_t = np.zeros((hidden_size,)) # 초기 셀 상태는 0(벡터)로 초기화
# 은닉 상태의 크기 hidden_size로 은닉 상태를 만듬.

#시그모이드 함수 정의
def sigmoid(x):
    return 1 / (1 +np.exp(-x))

#region weights
Wxi = np.random.random((hidden_size, input_size))
Wxg = np.random.random((hidden_size, input_size))
Wxf = np.random.random((hidden_size, input_size))
Wxo = np.random.random((hidden_size, input_size))

Whi = np.random.random((hidden_size, hidden_size))
Whg = np.random.random((hidden_size, hidden_size))
Whf = np.random.random((hidden_size, hidden_size))
Who = np.random.random((hidden_size, hidden_size))

bi = np.random.random((hidden_size,))
bg = np.random.random((hidden_size,))
bf = np.random.random((hidden_size,))
bo = np.random.random((hidden_size,))
#endregion

total_hidden_states = []
total_cell_states = []

# 메모리 셀 동작
for input_t in inputs: # 각 시점에 따라서 입력값이 입력됨.

    # 입력 게이트
    it = sigmoid(np.dot(Wxi, input_t) + np.dot(Whi, hidden_state_t) + bi)
    gt = np.tanh(np.dot(Wxg, input_t) + np.dot(Whg, hidden_state_t) + bg)

    # 삭제 게이트
    ft = sigmoid(np.dot(Wxf, input_t) + np.dot(Whf, hidden_state_t) + bf)

    # 장기 상태
    ct = ft * cell_state_t + it * gt

    # 단기 상태
    ot = sigmoid(np.dot(Wxo, input_t) + np.dot(Who, hidden_state_t) + bo)
    output_t = ot * np.tanh(ct)

    # 최종 값 업데이트
    cell_state_t = ct # 셀 상태를 업데이트
    total_cell_states.append(cell_state_t)

    hidden_state_t = output_t # 은닉 상태를 업데이트
    total_hidden_states.append(hidden_state_t)
    

total_hidden_states = np.stack(total_hidden_states, axis = 0)
total_cell_states = np.stack(total_hidden_states, axis = 0) 
# 출력 시 값을 깔끔하게 해준다.

print(np.shape(output_t))
print(np.shape(total_hidden_states))
print(np.shape(total_cell_states))