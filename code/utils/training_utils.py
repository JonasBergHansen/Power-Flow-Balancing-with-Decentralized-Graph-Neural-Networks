import numpy as np
import tensorflow as tf
from spektral.layers.ops import sp_matrix_to_sp_tensor
#from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.data.utils import batch_generator, to_disjoint
          
def create_and_train_model_disjoint_mode(model_class,
                                         inputs,
                                         targets,
                                         validation_set,
                                         loss_fn,
                                         optimizer,
                                         epochs,
                                         batch_size,
                                         early_stopping = False,
                                         patience = 10,
                                         restore_best_weights = False,
                                         ):
    
    """
    Function for training a graph neural network in disjoint mode.
    Only works for GNNs that do not use edge features or pooling operations.
    
    Params:
        - model_class: Class function for GNN model. Must belong to the tf.keras.Model class.
        - inputs: Tuple containing the input training data of the form (X, A). Here node features X and adj. matrices A
                  are either a list of samples or an array where the first axis is the sample dimension. 
        - targets: Training targets/labels. List or array.
        - tf_signature: Tensorflow input signature for the model. *Might not be needed*
        - validation_set: Tuple containing the validation data of the form (val_inputs, val_targets), or more explicitly
                          ((X_val, A_val), y_val).
        - loss_fn: Loss function for training model.
        - optimizer: Optimizer function (e.g. SGD, Adam...)
        - epochs: Number of training epochs. Integer.
        - batch_size: Batch size. Applies to both training and validation data. Integer.
        - early_stopping: If True, training terminates if validation loss has not decreased for a set
                          number of epochs given by patience. At termination, the weights that gave the
                          best validation score is restored. Boolean.
        - patience: A given number of epochs for early stopping. Integer.
    
    Returns:
        - Trained model object
        - Training and validation loss arrays.
    """
    
    # Set up model
    model = model_class()
    
    tf_signature = ((tf.TensorSpec((None, inputs[0][0].shape[-1]), dtype=tf.float32),
                     tf.SparseTensorSpec((None, None), dtype=tf.float32)),
                     tf.TensorSpec((None, targets[0].shape[-1]), dtype=tf.float32))
    
    @tf.function(input_signature=tf_signature,
                 experimental_relax_shapes=True)
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(targets, predictions)
            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    
    train_loss_vec = np.zeros(epochs)
    val_loss_vec = np.zeros(epochs)
    
    epoch = 1

    current_batch = 0
    model_loss = 0
    batches_in_epoch = np.ceil(len(inputs[0]) / batch_size)
    val_batches_in_epoch = np.ceil(len(validation_set[-1]) / batch_size)
    
    batches_train = batch_generator([*inputs, targets],
                                    batch_size=batch_size, 
                                    epochs=epochs)
    
    best_val_loss = np.Inf
    wait = 0
        
    for b in batches_train:
        # Convert batch data to disjoint mode
        X_, A_, _ = to_disjoint(b[0], b[1])
        A_ = sp_matrix_to_sp_tensor(A_)
        y_ = np.vstack(b[-1])
        
        outs = train_step([X_, A_], y_)
        model_loss += outs.numpy()
        current_batch += 1
        if current_batch == batches_in_epoch:
    
            loss_train = model_loss / batches_in_epoch
    
            batches_val = batch_generator([*validation_set[0], validation_set[1]],
                                          batch_size = batch_size, epochs = 1, shuffle = False)
            
            loss_val = 0
            for b_val in batches_val:
              X_, A_, _ = to_disjoint(b_val[0], b_val[1])
              A_ = sp_matrix_to_sp_tensor(A_)
              y_ = np.vstack(b_val[-1])
              
              outs = model([X_, A_], training = False)
              loss_val += loss_fn(y_, outs)
    
            loss_val = loss_val / val_batches_in_epoch
            
            
            print('Epoch: {} Train Loss: {:.6f} Valid Loss: {:.6f}'.format(epoch, 
                                                                   loss_train,
                                                                   loss_val))
            train_loss_vec[epoch-1] = loss_train
            val_loss_vec[epoch-1] = loss_val
            
            # Control early stopping procedure
            if early_stopping == True or restore_best_weights == True:
                if loss_val < best_val_loss:
                    wait = 0
                    best_weights = model.get_weights()
                    best_val_loss = loss_val
                elif wait >= patience and early_stopping == True:
                    break
                else:
                    wait += 1
            
            epoch += 1
            model_loss = 0
            current_batch = 0
    
    
    if restore_best_weights == True:
         model.set_weights(best_weights)
         # Fill missing loss values with NaN
         if epoch < epochs:
             train_loss_vec[epoch:] = np.nan
             val_loss_vec[epoch:] = np.nan
    
    
    return model, train_loss_vec, val_loss_vec



def create_and_train_model_batch_mode(model_class,
                                      inputs,
                                      targets,
                                      validation_set,
                                      loss_fn,
                                      optimizer,
                                      epochs,
                                      batch_size,
                                      callbacks = None,
                                      model_args = None):
    """
    Function for training a Keras model in batch mode.
    Params:
        - model_class: Class function for any model that can work in batch mode. Must belong to the tf.keras.Model class.
        - inputs: Tuple containing the input training data. Setup must match whatever the
                  model takes as input, where each tuple element is a list of samples or an array
                  where the first axis is the sample dimension.
        - targets: Training sample targets/labels. List or array.
        - validations set: Tuple containing the validation data of the form (inputs, targets).
        - loss_fn: Loss function.
        - optimizer: Optimizer function (e.g. SGD, Adam...)
        - epochs: Number of training epochs. Integer.
        - batch_size: Batch size. Applies to both training and validation data. Integer.
        - callbacks: Callbacks to be given to Keras's model.fit, e.g. early stopping. 
                     See Tensorflow doc for format/syntax.
        - model_args: Additional arguments that are used when instantiating an object of the model class.
                      E.g. number of output units. Must be an iterable (e.g. list).
    
    Returns:
        - Trained model object
        - Training and validation loss arrays.
    
    """
    if model_args:
        model = model_class(*model_args)
    else:
        model = model_class()
    model.compile(optimizer = optimizer, loss = loss_fn)
    history = model.fit(inputs, targets, batch_size=batch_size, epochs=epochs, 
                        validation_data = validation_set, callbacks = callbacks)
    
    return model, history.history['loss'], history.history['val_loss']
