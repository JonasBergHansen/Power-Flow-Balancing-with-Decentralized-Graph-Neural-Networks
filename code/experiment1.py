import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils/'))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from training_utils import create_and_train_model_batch_mode, create_and_train_model_disjoint_mode
from testing_utils import compute_predictions, compute_nrmse
from models import Local_MLP, Global_MLP, GCN, ARMA_GNN
from spektral.utils.convolution import normalized_adjacency, gcn_filter
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from collections import OrderedDict


# Tunables and other options
P = OrderedDict([
    ('model', 'arma_gnn'), # Model: 'local_mlp', 'global_mlp', 'gcn' or 'arma_gnn'
    ('dataset', 'case30'), # case30, case118 or case300
    ('epochs', 250), # Number of epochs
    ('learning_rate', 1e-3),
    ('batch_size', 16),
    ('early_stopping', True),
    ('patience', 25), # Patience for early stopping
    ('best_val_weights', True), # Return best validation weights
    ('plot_loss', False),
    ('save_model', False),
    ('test_model', True)
])

SAVE_PATH = ''

model_classes = {'local_mlp': Local_MLP, 'global_mlp': Global_MLP,
                 'gcn': GCN, 'arma_gnn': ARMA_GNN}
adj_normalizer = {'gcn': gcn_filter, 'arma_gnn': normalized_adjacency}


##########################################################################
# LOAD TRAINING AND VALIDATION DATASETS
##########################################################################
if P['model'] in ['local_mlp', 'gcn', 'arma_gnn']:
    data_format = 'graph_structured/'

elif P['model'] == 'global_mlp':
    data_format = 'flattened/'

DATA_PATH = '../data/experiment1/' + P['dataset'] + '/' + data_format


X_train = np.load(DATA_PATH + 'X_train.npy')
y_train = np.load(DATA_PATH + 'y_train.npy')

X_val = np.load(DATA_PATH + 'X_val.npy')
y_val = np.load(DATA_PATH + 'y_val.npy')

train_targets = y_train
val_targets = y_val

if P['model'] in ['arma_gnn', 'gcn']:
    A_train = np.load(DATA_PATH + 'A_train.npy', allow_pickle=True).tolist()
    A_val = np.load(DATA_PATH + 'A_val.npy', allow_pickle=True).tolist()

    # Normalize adjacency matrices
    L_train = [adj_normalizer[P['model']](A_train[i], symmetric = True) for i in range(len(A_train))]
    L_val = [adj_normalizer[P['model']](A_val[i], symmetric = True) for i in range(len(A_val))]

    train_inputs = [X_train, L_train]
    val_inputs = [X_val, L_val]

elif P['model'] == 'local_mlp':
    train_inputs = X_train
    val_inputs = X_val

elif P['model'] == 'global_mlp':
    E_train= np.load(DATA_PATH + 'E_train.npy')
    E_val = np.load(DATA_PATH + 'E_val.npy')

    train_inputs = [X_train, E_train]
    val_inputs = [X_val, E_val]

##########################################################################
# MODEL SETUP AND TRAINING
##########################################################################
optimizer = Adam(P['learning_rate'])
loss_fn = MeanSquaredError()

# Train Local MLP or Global MLP
if P['model'] in ['local_mlp', 'global_mlp']:
    stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         patience = P['patience'],
                                                         restore_best_weights = P['best_val_weights'])

    extra_model_args = {'local_mlp': None, 'global_mlp': [train_targets.shape[-1]]}


    model, train_loss, val_loss = create_and_train_model_batch_mode(model_classes[P['model']],
                                                                    train_inputs,
                                                                    train_targets,
                                                                    (val_inputs, val_targets),
                                                                    loss_fn,
                                                                    optimizer,
                                                                    P['epochs'],
                                                                    P['batch_size'],
                                                                    [stopping_callback],
                                                                    extra_model_args[P['model']])


# Train GCN or ARMA GNN
elif P['model'] in ['gcn',  'arma_gnn']:

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

if P['save_model'] == True:
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

if P['test_model']:
    X_test = np.load(DATA_PATH + 'X_test.npy')
    test_targets = np.load(DATA_PATH + 'y_test.npy')

    if P['model'] in ['arma_gnn', 'gcn']:
        A_test = np.load(DATA_PATH + 'A_test.npy', allow_pickle=True).tolist()
        L_test = [adj_normalizer[P['model']](A_test[i], symmetric = True) for i in range(len(A_test))]

        preds = compute_predictions(model, X_test, L_test, batch_size = 256)
        nrmse_val = compute_nrmse(test_targets, preds)


    elif P['model'] == 'local_mlp':
        preds = model.predict(X_test, batch_size = 256)
        nrmse_val = compute_nrmse(test_targets, preds)

    elif P['model'] == 'global_mlp':
        E_test= np.load(DATA_PATH + 'E_test.npy')
        test_inputs = [X_test, E_test]

        preds = model.predict(test_inputs, batch_size = 256)
        nrmse_val = compute_nrmse(test_targets, preds, global_mlp = True)


    print(f"Test NRMSE: {nrmse_val}")
