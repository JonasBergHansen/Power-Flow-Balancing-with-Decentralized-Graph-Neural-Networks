import numpy as np
from spektral.layers import ops
from spektral.data.utils import batch_generator, to_disjoint

def compute_predictions(model, X, A, batch_size = 1):
    batches = batch_generator([X, A], batch_size = batch_size, epochs = 1, shuffle = False)
    output = []
    for b in batches:
            X_, A_, I_ = to_disjoint(b[0], b[1])
            A_ = ops.sp_matrix_to_sp_tensor(A_)
            
            preds = model([X_, A_], training = False).numpy()
            output.extend([preds[I_ == k,:] for k in range(len(b[0]))])
    return output 

def compute_nrmse(y_true, y_pred, global_mlp = False):
    if global_mlp:
        y_true = y_true.reshape([y_true.shape[0], int(y_true.shape[1]/8), 8], order = 'F')
        y_pred = y_pred.reshape([y_pred.shape[0], int(y_pred.shape[1]/8), 8], order = 'F')
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    true_var = np.nanvar(y_true, axis = 0)
    series_length = y_pred.shape[0]
    nrmse_vals = np.sqrt((np.sum((y_pred - y_true) ** 2, axis = 0) / float(series_length))/true_var)
    return np.nanmean(nrmse_vals)
