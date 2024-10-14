import numpy as np
import torch

# Polyhedra Graph
phi = (np.sqrt(5)+1) / 2
pos_dict = {}

# Tetrahedron
pos_dict[4] = torch.FloatTensor([
    [ 1, 1, 1],         [-1,-1, 1],
    [-1, 1,-1],         [ 1,-1,-1],
])

# Hexahedron
pos_dict[6] = torch.FloatTensor([
    [ 1, 1, 1],         [ 1, 1,-1],         [ 1,-1, 1],         [-1, 1, 1],
    [ 1,-1,-1],         [-1, 1,-1],         [-1,-1, 1],         [-1,-1,-1],
])

# Octahedron
pos_dict[8] = torch.FloatTensor([
    [ 0, 0, 1],         [ 0, 0,-1],         [ 0, 1, 0],         [ 0,-1, 0],
    [ 1, 0, 0],         [-1, 0, 0],
])

# Dodecahedron
pos_dict[12] = torch.FloatTensor([
    [0, phi, 1/phi],    [0, phi,-1/phi],    [0,-phi, 1/phi],    [0,-phi,-1/phi],
    [ 1/phi,0, phi],    [-1/phi,0, phi],    [ 1/phi,0,-phi],    [-1/phi,0,-phi],
    [ phi, 1/phi,0],    [-phi, 1/phi,0],    [ phi,-1/phi,0],    [-phi,-1/phi,0], 
    [ 1, 1, 1],         [ 1, 1,-1],         [ 1,-1, 1],         [-1, 1, 1],
    [ 1,-1,-1],         [-1, 1,-1],         [-1,-1, 1],         [-1,-1,-1],
])

# Icosahedron
pos_dict[20] = torch.FloatTensor([
    [ 0, 1, phi],       [ 0, 1,-phi],       [ 0,-1, phi],       [ 0,-1,-phi],
    [ 1, phi, 0],       [-1, phi, 0],       [ 1,-phi, 0],       [-1,-phi, 0],
    [ phi, 0, 1],       [ phi, 0,-1],       [-phi, 0, 1],       [-phi, 0,-1],
])

ver_num_dict = {4:4, 6:8, 8:6, 12:20, 20:12}