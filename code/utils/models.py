import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from spektral.layers import ARMAConv, GCNConv

# =============================================================================
#                       Experiment models
# =============================================================================

# ARMA GNN class
class ARMA_GNN(Model):
    def __init__(self):
        super().__init__()    
        act_func = tf.keras.layers.LeakyReLU(alpha=0.2)
        
        # Pre-processing layers
        self.predense1 = Dense(64, kernel_regularizer = None, activation = act_func)
        self.predense2 = Dense(64, kernel_regularizer = None, activation = act_func)
               
        
        # ARMA layers
        self.arma1 = ARMAConv(64, order = 2, iterations = 8, share_weights = True,
                              dropout = 0.0, gcn_activation = act_func,
                              activation = None, kernel_regularizer = None)
        
        self.arma2 = ARMAConv(64, order = 2, iterations = 8, share_weights = True,
                              dropout = 0.0, gcn_activation = act_func,
                              activation = None, kernel_regularizer = None)
        
        self.arma3 = ARMAConv(64, order = 2, iterations = 8, share_weights = True,
                              dropout = 0.0, gcn_activation = act_func,
                              activation = None, kernel_regularizer = None)
        
        self.arma4 = ARMAConv(64, order = 2, iterations = 8, share_weights = True,
                              dropout = 0.0, gcn_activation = act_func,
                              activation = None, kernel_regularizer = None)
        
        self.arma5 = ARMAConv(64, order = 2, iterations = 8, share_weights = True,
                              dropout = 0.0, gcn_activation = act_func,
                              activation = None, kernel_regularizer = None)
        
        # Post-processing layer
        self.postdense1 = Dense(64, kernel_regularizer = None, activation = act_func)
        self.postdense2 = Dense(64, kernel_regularizer = None, activation = act_func)
        
        # Output layer
        self.readout = Dense(8, kernel_regularizer = None, activation = None)
    
    
    def call(self, inputs):

        X, A = inputs
        
        out = self.predense1(X)
        out = self.predense2(out)
              
        out = self.arma1([out, A])
        out = self.arma2([out, A])
        out = self.arma3([out, A])
        out = self.arma4([out, A])
        out = self.arma5([out, A])
        
        out = self.postdense1(out)        
        out = self.postdense2(out)
        
        out = self.readout(out)
        
        return out


# GCN class
class GCN(Model):
    def __init__(self):
        super().__init__()    
        act_func = tf.keras.layers.LeakyReLU(alpha=0.2)
        
        # Pre-processing layers
        self.predense1 = Dense(64, kernel_regularizer = None, activation = act_func)
        self.predense2 = Dense(64, kernel_regularizer = None, activation = act_func)
               
        
        # GCN layers
        self.gcn_layers = []
        
        for i in range(40):
            self.gcn_layers.append(GCNConv(64, activation = act_func, 
                                           kernel_regularizer=None))
        
        # Post-processing layer
        self.postdense1 = Dense(64, kernel_regularizer = None, activation = act_func)
        self.postdense2 = Dense(64, kernel_regularizer = None, activation = act_func)
        
        # Output layer
        self.readout = Dense(8, kernel_regularizer = None, activation = None)
    
    
    def call(self, inputs):

        X, A = inputs
        
        out = self.predense1(X)
        out = self.predense2(out)
        
        for i in range(40):
            out = self.gcn_layers[i]([out, A])
        
        out = self.postdense1(out)        
        out = self.postdense2(out)
        
        out = self.readout(out)
        
        return out
    


class Local_MLP(Model):
    def __init__(self):
        super().__init__()
        
        act_func = tf.keras.layers.LeakyReLU(alpha=0.2)
        
        self.dense1 = Dense(256, kernel_regularizer = None, activation = act_func)
        self.dense2 = Dense(256, kernel_regularizer = None, activation = act_func)
        self.dense3 = Dense(256, kernel_regularizer = None, activation = act_func)
        self.dense4 = Dense(128, kernel_regularizer = None, activation = act_func)
        self.dense5 = Dense(64, kernel_regularizer = None, activation = act_func)
        
        self.readout = Dense(8, kernel_regularizer = None, activation = None)
        
    def call(self, inputs):
        X = inputs
        
        out = self.dense1(X)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.dense4(out)
        out = self.dense5(out)
        
        out = self.readout(out)

        return out


class Global_MLP(Model):
    def __init__(self, F_out):
        super().__init__()
        
        act_func = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.bus_preprocess = Dense(64, kernel_regularizer=None, activation = act_func)
        self.edge_preprocess = Dense(64, kernel_regularizer=None, activation = act_func)
        
        self.dense1 = Dense(128, kernel_regularizer=None, activation = act_func)
        self.dense2 = Dense(128, kernel_regularizer=None, activation = act_func)
        self.readout = Dense(F_out, kernel_regularizer=None, activation = None)


    def call(self, inputs):
        if len(inputs) == 2:
            X, E = inputs
        else:
            X, E, I = inputs
        
        bus_processed = self.bus_preprocess(X)
        edge_processed = self.edge_preprocess(E)
        
        out = tf.concat([bus_processed, edge_processed], axis = 1)
        
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.readout(out)
        
        if len(inputs) == 2:
            return out
        else:
            return out*I
        
