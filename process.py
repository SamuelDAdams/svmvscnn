import torch
import numpy as np

def np_to_tensor(path):
    data = np.load(path)
    name = path.split('.')[0]
    data = torch.FloatTensor(data)
    torch.save(data, name + '_tensor.pth')

np_to_tensor('test/bin_labels.npy')
np_to_tensor('test/embeddings.npy')

