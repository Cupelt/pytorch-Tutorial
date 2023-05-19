import torch
import torch.nn.functional as F

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum(dim=0) # only difference

scores = torch.tensor([-1.1487,  0.2165, -0.6754, -0.1675, -1.4490])
print(scores)

probabilities = F.softmax(scores, dim=0)
print(probabilities)