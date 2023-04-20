from pathlib import Path

import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split

from utils import construct_gnn_dataset

grid_cases = ['case30/'] # ['case30/', 'case118/', 'case300/']

DATA_PATH = 'data/matpower_format/'

SAVE_PATH = 'data/graph_structured/'

for i in range(len(grid_cases)):
    
    X = []
    A = []
    y = []
    y_dcpf = []
    
    for j in range(1, 16):
        # Import the generated grids
        mpc_data = io.loadmat(DATA_PATH + grid_cases[i] + 'grids_pre_soln_batch' + str(j))['mpc_array']
        acpf_data = io.loadmat(DATA_PATH + grid_cases[i] + 'grids_post_acpf_batch' + str(j))['res_acpf_array']
        dcpf_data= io.loadmat(DATA_PATH + grid_cases[i] + 'grids_post_dcpf_batch' + str(j))['res_dcpf_array']
        # Construct dataset
        X_, A_, y_, y_dcpf_ = construct_gnn_dataset(mpc_data, acpf_data, dcpf_data)
        
        X.extend(X_)
        A.extend(A_)
        y.extend(y_)
        y_dcpf.extend(y_dcpf_)
        
        print('Batch {} added'.format(j))
    
    X_train, X_test, \
    A_train, A_test, \
    y_train, y_test, \
    y_dcpf_train, y_dcpf_test = train_test_split(X, A, y, y_dcpf, test_size = 0.30, random_state = 1)
    
    X_train, X_val, \
    A_train, A_val, \
    y_train, y_val, \
    y_dcpf_train, y_dcpf_val = train_test_split(X_train, A_train, y_train, y_dcpf_train, test_size = 0.20, random_state = 1)
    

    Path(SAVE_PATH + grid_cases[i]).mkdir(parents=True, exist_ok=True)

    np.save(SAVE_PATH + grid_cases[i] + 'X_train.npy', X_train)
    np.save(SAVE_PATH + grid_cases[i] + 'A_train.npy', A_train)
    np.save(SAVE_PATH + grid_cases[i] + 'y_train.npy', y_train)
    np.save(SAVE_PATH + grid_cases[i] + 'y_dcpf_train.npy', y_dcpf_train)
    
    np.save(SAVE_PATH + grid_cases[i] + 'X_val.npy', X_val)
    np.save(SAVE_PATH + grid_cases[i] + 'A_val.npy', A_val)
    np.save(SAVE_PATH + grid_cases[i] + 'y_val.npy', y_val)
    np.save(SAVE_PATH + grid_cases[i] + 'y_dcpf_val.npy', y_dcpf_val)
    
    np.save(SAVE_PATH + grid_cases[i] + 'X_test.npy', X_test)
    np.save(SAVE_PATH + grid_cases[i] + 'A_test.npy', A_test)
    np.save(SAVE_PATH + grid_cases[i] + 'y_test.npy', y_test)
    np.save(SAVE_PATH + grid_cases[i] + 'y_dcpf_test.npy', y_dcpf_test)


    print('Dataset saved')

