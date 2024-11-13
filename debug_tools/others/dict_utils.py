import numpy as np
import os
import json
from datetime import datetime

def save_dict(data_dict, filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"./output/{filename}_{timestamp}.txt"
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    data_dict = {k: [{'real': v.real, 'imag': v.imag} for v in arr] for k, arr in data_dict.items()}
    
    with open(filepath, 'w') as f:
        json.dump(data_dict, f)
    print(f'filename:{filepath}')


def load_dict(filepath):
    with open(filepath, 'r') as f:
        data_dict = json.load(f)
    

    data_dict = {k: np.array([complex(item['real'], item['imag']) for item in v]) for k, v in data_dict.items()}
    
    return data_dict

