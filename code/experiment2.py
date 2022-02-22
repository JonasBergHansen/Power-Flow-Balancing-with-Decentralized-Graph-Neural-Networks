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
from collections import OrderedDict
import tensorflow.keras.backend as kb


def custom_mse(y_true, y_pred):
    return kb.mean(kb.square(y_true[y_true != 0] - y_pred[y_true != 0]))


# Tunables and other options
P = OrderedDict([
    ('model', 'arma_gnn'),  # 'local_mlp', 'global_mlp' or 'arma_gnn'
    ('epochs', 250),
    ('learning_rate', 1e-3),
    ('batch_size', 16),
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
    data_format = 'graph_structured/'

elif P['model'] == 'global_mlp':
    data_format = 'flattened/'

DATA_PATH = '../data/experiment2/' + data_format


if P['model'] == 'arma_gnn':
    X_train = np.load(DATA_PATH + 'X_train.npy', allow_pickle=True).tolist()
    A_train = np.load(DATA_PATH + 'A_train.npy', allow_pickle=True).tolist()
    y_train = np.load(DATA_PATH + 'y_train.npy', allow_pickle=True).tolist()

    X_val = np.load(DATA_PATH + 'X_val.npy', allow_pickle=True).tolist()
    A_val = np.load(DATA_PATH + 'A_val.npy', allow_pickle=True).tolist()
    y_val = np.load(DATA_PATH + 'y_val.npy', allow_pickle=True).tolist()

    L_train = [adj_normalizer[P['model']](A_train[i], symmetric = True) for i in range(len(A_train))]
    L_val = [adj_normalizer[P['model']](A_val[i], symmetric = True) for i in range(len(A_val))]

    train_inputs = [X_train, L_train]
    val_inputs = [X_val, L_val]

    train_targets = y_train
    val_targets = y_val

elif P['model'] == 'local_mlp':
    X_train = np.load(DATA_PATH + 'X_train.npy', allow_pickle=True).tolist()
    y_train = np.load(DATA_PATH + 'y_train.npy', allow_pickle=True).tolist()

    X_val = np.load(DATA_PATH + 'X_val.npy', allow_pickle=True).tolist()
    y_val = np.load(DATA_PATH + 'y_val.npy', allow_pickle=True).tolist()

    # The local MLP works on a single branch/vertex at a time, so the grids are combined
    # into one and the batch size is increased accordingly (multiplied by the number of branches in the full grid)
    train_inputs = np.vstack(X_train)
    train_targets = np.vstack(y_train)

    val_inputs = np.vstack(X_val)
    val_targets = np.vstack(y_val)

elif P['model'] == 'global_mlp':
    X_train = np.load(DATA_PATH + 'X_train.npy')
    E_train = np.load(DATA_PATH + 'E_train.npy')
    y_train = np.load(DATA_PATH + 'y_train.npy')

    X_val = np.load(DATA_PATH + 'X_val.npy')
    E_val = np.load(DATA_PATH + 'E_val.npy')
    y_val = np.load(DATA_PATH + 'y_val.npy')

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
        batch_size = int(411*batch_size) # 411 is the number of branches in the non-perturbated grid


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

    if P['model'] in ['arma_gnn', 'gcn']:
        X_test = np.load(DATA_PATH + 'X_test.npy', allow_pickle=True).tolist()
        test_targets = np.load(DATA_PATH + 'y_test.npy', allow_pickle=True).tolist()
        A_test = np.load(DATA_PATH + 'A_test.npy', allow_pickle=True).tolist()
        L_test = [adj_normalizer[P['model']](A_test[i], symmetric = True) for i in range(len(A_test))]

        preds = compute_predictions(model, X_test, L_test, batch_size = 256)
        nrmse_val = compute_nrmse(test_targets, preds)


    elif P['model'] == 'local_mlp':
        X_test = np.load(DATA_PATH + 'X_test.npy', allow_pickle=True).tolist()
        test_targets = np.load(DATA_PATH + 'y_test.npy', allow_pickle=True).tolist()
        X_test = np.vstack(X_test)
        test_targets = np.vstack(test_targets)

        preds = model.predict(X_test, batch_size = 256)
        nrmse_val = compute_nrmse(test_targets, preds)

    elif P['model'] == 'global_mlp':
        X_test = np.load(DATA_PATH + 'X_test.npy')
        test_targets = np.load(DATA_PATH + 'y_test.npy')
        E_test = np.load(DATA_PATH + 'E_test.npy')
        test_mask = np.zeros_like(test_targets, dtype=np.float32)
        test_mask[(test_targets != 0.0)] = 1.0
        test_inputs = [X_test, E_test, test_mask]

        preds = model.predict(test_inputs, batch_size = 256)
        nrmse_val = compute_nrmse(test_targets, preds, global_mlp = True, padded = True)


    print(f"Test NRMSE: {nrmse_val}")
