import torch
from einops import repeat 
import json

def positions(b, n, device = None):
    pos = repeat(torch.arange(n, device=device), "n -> b n", b=b)
    return torch.clamp(pos, min=0)

def save_dict_to_json(dico, filepath) :
    with open(filepath, 'w') as fp:
        json.dump(dico, fp, indent=4)