from torchmdnet.models.model import load_model
import torch
import numpy as np

model = load_model('model1.ckpt')

#model = load_model('model1.ckpt', device=torch.device('cuda:0'))

typeDict = {('Br', -1): 0, ('Br', 0): 1, ('C', -1): 2, ('C', 0): 3, ('C', 1): 4, ('Ca', 2): 5, ('Cl', -1): 6,
            ('Cl', 0): 7, ('F', -1): 8, ('F', 0): 9, ('H', 0): 10, ('I', -1): 11, ('I', 0): 12, ('K', 1): 13,
            ('Li', 1): 14, ('Mg', 2): 15, ('N', -1): 16, ('N', 0): 17, ('N', 1): 18, ('Na', 1): 19, ('O', -1): 20,
            ('O', 0): 21, ('O', 1): 22, ('P', 0): 23, ('P', 1): 24, ('S', -1): 25, ('S', 0): 26, ('S', 1): 27}


types = torch.tensor([6, 19], dtype=torch.long)
pos = torch.tensor([[0, 0, 0], [0, 3, 0]], dtype=torch.float32)
energy, forces = model.forward(types, pos)

print(energy, forces)
