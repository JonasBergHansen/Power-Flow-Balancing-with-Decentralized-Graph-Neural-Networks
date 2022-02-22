import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils/'))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from training_utils import create_and_train_model_batch_mode, create_and_train_model_disjoint_mode
from testing_utils import compute_predictions, compute_nrmse
from models import Local_MLP, Global_MLP, ARMA_GNN
from spektral.utils.convolution import normalized_adjacency
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as kb
from collections import OrderedDict

def custom_mse(y_true, y_pred):
    return kb.mean(kb.square(y_true[y_true != 0] - y_pred[y_true != 0]))


# Tunables and other options
P = OrderedDict([
    ('model', 'global_mlp'),  # 'local_mlp', 'global_mlp' or 'arma_gnn'
    ('epochs', 250),
    ('learning_rate', 1e-3),
    ('batch_size', 32),
    ('early_stopping', True),
    ('patience', 250),
    ('best_val_weights', True),
    ('plot_loss', False),
    ('save_model', False)
])

SAVE_PATH = ''

model_classes = {'local_mlp': Local_MLP, 'global_mlp': Global_MLP,
                 'arma_gnn': ARMA_GNN}
adj_normalizer = {'arma_gnn': normalized_adjacency}


##########################################################################
# LOAD TRAINING AND VALIDATION DATASETS
##########################################################################
if P['model'] in ['local_mlp', 'arma_gnn']:
    data_format = '/graph_structured/'
elif P['model'] == 'global_mlp':
    data_format = '/flattened/'

DATA_PATH = '../data/experiment3/'


grid_cases = ['case9', 'case14', 'case_ieee30', 'case39', 'case89pegase', 'case118'];



rng = np.random.default_rng(2021) # For reproducible validation shuffling

if P['model'] == 'arma_gnn':
    X_train = []
    A_train = []
    y_train = []

    X_val = []
    A_val = []
    y_val = []
    for k, case in enumerate(grid_cases):
        X_train.extend(list(np.load(DATA_PATH + case + data_format + 'X_train.npy')))
        A_train.extend(np.load(DATA_PATH + case + data_format + 'A_train.npy', allow_pickle=True).tolist())
        y_train.extend(list(np.load(DATA_PATH + case + data_format + 'y_train.npy')))

        X_val.extend(list(np.load(DATA_PATH + case + data_format + 'X_val.npy')))
        A_val.extend(np.load(DATA_PATH + case + data_format + 'A_val.npy', allow_pickle=True).tolist())
        y_val.extend(list(np.load(DATA_PATH + case + data_format + 'y_val.npy')))




    L_train = [adj_normalizer[P['model']](A_train[i], symmetric = True) for i in range(len(A_train))]
    L_val = [adj_normalizer[P['model']](A_val[i], symmetric = True) for i in range(len(A_val))]

    # Shuffle the validation set
    shuffle_idx = np.arange(0, len(X_val), dtype = np.int32)
    rng.shuffle(shuffle_idx)
    X_val = [X_val[i] for i in shuffle_idx]
    L_val = [L_val[i] for i in shuffle_idx]
    y_val = [y_val[i] for i in shuffle_idx]

    train_inputs = [X_train, L_train]
    val_inputs = [X_val, L_val]

    train_targets = y_train
    val_targets = y_val

elif P['model'] == 'local_mlp':
    X_train = []
    y_train = []

    X_val = []
    y_val = []
    for k, case in enumerate(grid_cases):
        X_train.extend(list(np.load(DATA_PATH + case + data_format + 'X_train.npy')))
        y_train.extend(list(np.load(DATA_PATH + case + data_format + 'y_train.npy')))

        X_val.extend(list(np.load(DATA_PATH + case + data_format + 'X_val.npy')))
        y_val.extend(list(np.load(DATA_PATH + case + data_format + 'y_val.npy')))

    # Shuffle the validation set
    shuffle_idx = np.arange(0, len(X_val), dtype = np.int32)
    rng.shuffle(shuffle_idx)
    X_val = [X_val[i] for i in shuffle_idx]
    y_val = [y_val[i] for i in shuffle_idx]

    mean_num_branches = int(np.mean([X_train[i].shape[0] for i in range(len(X_train))]))

    # The local MLP works on a single branch/node at a time, so the grids are combined
    # into one and the batch size is increased accordingly (multiply by the mean number of branches)
    train_inputs = np.vstack(X_train)
    train_targets = np.vstack(y_train)

    val_inputs = np.vstack(X_val)
    val_targets = np.vstack(y_val)

elif P['model'] == 'global_mlp':
    for k, case in enumerate(grid_cases):
        if k == 0:
            X_train = np.load(DATA_PATH + case + data_format + 'X_train.npy')
            E_train = np.load(DATA_PATH + case + data_format + 'E_train.npy')
            y_train = np.load(DATA_PATH + case + data_format + 'y_train.npy')

            X_val = np.load(DATA_PATH + case + data_format + 'X_val.npy')
            E_val = np.load(DATA_PATH + case + data_format + 'E_val.npy')
            y_val = np.load(DATA_PATH + case + data_format + 'y_val.npy')

        else:
            X_train = np.vstack([X_train, np.load(DATA_PATH + case + data_format + 'X_train.npy')])
            E_train = np.vstack([E_train, np.load(DATA_PATH + case + data_format + 'E_train.npy')])
            y_train = np.vstack([y_train, np.load(DATA_PATH + case + data_format + 'y_train.npy')])

            X_val = np.vstack([X_val, np.load(DATA_PATH + case + data_format + 'X_val.npy')])
            E_val = np.vstack([E_val, np.load(DATA_PATH + case + data_format + 'E_val.npy')])
            y_val = np.vstack([y_val, np.load(DATA_PATH + case + data_format + 'y_val.npy')])

    # Shuffle the validation set
    shuffle_idx = np.arange(0, len(X_val), dtype = np.int32)
    rng.shuffle(shuffle_idx)
    X_val = X_val[shuffle_idx]
    E_val = E_val[shuffle_idx]
    y_val = y_val[shuffle_idx]

    # Construct masks
    train_mask = np.zeros_like(y_train, dtype=np.float32)
    train_mask[(y_train != 0.0)] = 1.0

    val_mask = np.zeros_like(y_val, dtype=np.float32)
    val_mask[(y_val != 0.0)] = 1.0

    train_inputs = [X_train, E_train, train_mask]
    val_inputs = [X_val, E_val, val_mask]

    train_targets = y_train
    val_targets = y_val

##########################################################################
# MODEL SETUP AND TRAINING
##########################################################################
optimizer = Adam(P['learning_rate'])

if P['model'] == 'global_mlp':
    loss_fn = custom_mse
else:
    loss_fn = MeanSquaredError()




if P['model'] in ['local_mlp', 'global_mlp']:
    stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         patience = P['patience'],
                                                         restore_best_weights = P['best_val_weights'])

    extra_model_args = {'local_mlp': None, 'global_mlp': [train_targets.shape[-1]]}

    batch_size = P['batch_size']
    if P['model'] == 'local_mlp':
        batch_size = int(mean_num_branches*batch_size)


    model, train_loss, val_loss = create_and_train_model_batch_mode(model_classes[P['model']],
                                                                    train_inputs,
                                                                    train_targets,
                                                                    (val_inputs, val_targets),
                                                                    loss_fn,
                                                                    optimizer,
                                                                    P['epochs'],
                                                                    batch_size,
                                                                    [stopping_callback],
                                                                    extra_model_args[P['model']])


elif P['model'] in ['arma_gnn']:

    model, train_loss, val_loss = create_and_train_model_disjoint_mode(model_classes[P['model']],
                                                                       train_inputs,
                                                                       train_targets,
                                                                       (val_inputs, val_targets),
                                                                       loss_fn,
                                                                       optimizer,
                                                                       P['epochs'],
                                                                       P['batch_size'],
                                                                       early_stopping = P['early_stopping'],
                                                                       patience = P['epochs'],
                                                                       restore_best_weights = P['best_val_weights'])

if P['save_model']:
    model.save_weights(SAVE_PATH + '/model_weights')


##########################################################################
# PLOT LOSS (OPTIONAL)
##########################################################################

if P['plot_loss'] == True:
    fig, ax = plt.subplots()
    ax.plot(train_loss, label = 'train loss')
    ax.plot(val_loss, label = 'val loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()

##########################################################################
# TESTING (OPTIONAL)
##########################################################################
P['test_model'] = True
if P['test_model']:

    if P['model'] in ['arma_gnn']:
        test_cases = ['case9', 'case14', 'case_ieee30', 'case39',
                      'case89pegase', 'case118', 'case57', 'case300'];

        X_test = []
        A_test = []
        y_test = []

        case_nrmse_vals = np.zeros(len(test_cases))
        for k, case in enumerate(test_cases):
            X_test = list(np.load(DATA_PATH + case + data_format + 'X_test.npy'))
            A_test = np.load(DATA_PATH + case + data_format + 'A_test.npy', allow_pickle=True).tolist()
            y_test = list(np.load(DATA_PATH + case + data_format + 'y_test.npy'))

            L_test = [normalized_adjacency(A_test[i]) for i in range(len(A_test))]

            preds = compute_predictions(model, X_test, L_test, batch_size=512)

            case_nrmse_vals[k] = compute_nrmse(y_test, preds)


    elif P['model'] == 'local_mlp':
        test_cases = ['case9', 'case14', 'case_ieee30', 'case39',
                      'case89pegase', 'case118', 'case57', 'case300'];

        X_test = []
        y_test = []

        case_nrmse_vals = np.zeros(len(test_cases))
        for k, case in enumerate(test_cases):
            X_test = np.vstack(np.load(DATA_PATH + case + data_format + 'X_test.npy'))
            y_test = np.vstack(np.load(DATA_PATH + case + data_format + 'y_test.npy'))

            preds = model.predict(X_test, batch_size=256)
            case_nrmse_vals[k] = compute_nrmse(y_test, preds)


    elif P['model'] == 'global_mlp':
        test_cases = ['case9', 'case14', 'case_ieee30', 'case39',
                      'case89pegase', 'case118', 'case57'];

        case_nrmse_vals = np.zeros(len(test_cases))
        for k, case in enumerate(test_cases):
            X_test = np.load(DATA_PATH + case + data_format + 'X_test.npy')
            E_test = np.load(DATA_PATH + case + data_format + 'E_test.npy')
            y_test = np.load(DATA_PATH + case + data_format + 'y_test.npy')

            test_mask = np.zeros_like(y_test, dtype=np.float32)
            test_mask[(y_test != 0.0)] = 1.0

            preds = model.predict([X_test, E_test, test_mask], batch_size = 256)

            case_nrmse_vals[k] = compute_nrmse(y_test, preds, global_mlp=True, padded = True)


    for i, label in enumerate(test_cases):
        print("{} NRMSE: {}".format(test_cases[i], case_nrmse_vals[i]))
