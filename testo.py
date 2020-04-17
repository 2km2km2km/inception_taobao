import torch
outputs=torch.tensor([[1,5,3],[9,3,4]])
_, prediction = torch.max(outputs.data, 1)
print(prediction)