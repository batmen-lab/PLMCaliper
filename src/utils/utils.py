import random
import os
import numpy as np
import pickle

def set_randseed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    
def load_data(data_type, decoy_type):
    prefix = f'../data/result_{data_type}_astral'
    param_prefix = "_blosum62_Q11R1" if data_type == "blastp" else ""

    url_homo_origin = f'{prefix}{param_prefix}_homo_hit1000.pkl'
    url_nonhomo_origin = f'{prefix}{param_prefix}_nonhomo_hit1000.pkl'
    url_homo = f"{prefix}_{decoy_type}{param_prefix}_homo_hit1000.pkl"
    url_nonhomo = f"{prefix}_{decoy_type}{param_prefix}_nonhomo_hit1000.pkl"  
    urls = [url_homo_origin, url_nonhomo_origin, url_homo, url_nonhomo]

    data = []
    for u in urls:
        assert os.path.exists(u), f'{u} not exist'    
        with open(u, 'rb') as f:
            data.append(pickle.load(f))
                
    return data