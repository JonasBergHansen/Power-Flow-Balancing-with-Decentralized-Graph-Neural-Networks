# Data setup
Here, the structure of inputs and outputs when training and testing are explained with included illustrations.

## GNN and Local MLP
For the branch/line graph representation used by the GNNs (GCN and ARMA GNN), each grid is given by a feature matrix X, an adjacency matrix A and a target output matrix Y. An illustration of these three matrices is given in the below figure. For X and Y, each row corresponds to a specific branch in the power grid, and the columns correspond to the relevant features found on each branch. 

In case of the input matrix X, these features can be split into three main parts, where the first part contains features related to the branch in question, while the second and third are related to the buses found at the "from" and "to" ends of the branch, respectively. The features from the branch itself include (in order from left to right w.r.t. the columns): 
- resistance (p.u.), 
- reactance (p.u.), 
- charging susceptance (p.u.), 
- transformer tap ratio
- transformer shift angle (radians). 

From each bus at the ends, the included features are: 
- active power load (p.u.), 
- reactive power load (p.u.), 
- shunt conductance (p.u.), 
- shunt susceptance (p.u.), 
- voltage magnitude set point for connected generator (p.u.), 
- active power generation for connected generator (p.u.) 
- a two-dimensional one-hot vector indicating if the bus is the slack/reference bus or not. 

In total, X has 21 columns/features. If any of the features in some of the branches and connected buses, these features are set to zero. For instance, if the buses at the ends do not have a generator or a shunt, features relating to these components are zero.

The output target matrix Y is structured similarly to X, but the columns here hold the branch power flow output that are used to train the models and evaluate their performance. Specifically the columns contain: 
- active and reactive power injection (p.u.) at "from" end, 
- active and reactive current injection (p.u.) at "from" end, 
- active and reactive power injection (p.u.) at "to" end, 
- active and reactive current injection (p.u.) at "to" end, 
So, in total, there are 8 output quantities per branch.

The adjacency matrix A indicates which branches are considered to be neighbors of one another. Specifically, if branch number i shares a connected bus at either end with branch number j, then A_ij = A_ji = 1, otherwise A_ij = A_ji = 0. The branch order along the rows and columns of A follow the same order as the rows in X and Y. Also, A is given by a sparse representation which only contain the nonzero elements and their location (row, column).

For the Local MLP, only a single graph vertex is considered at a time, so for this model only X and Y are utilized during training and predictions are computed row-wise for these matrices. Therefore, the number of input and output units of the local MLP equals the number of columns in X and Y (21 and 8), respectively.

![gnn_data_structure](/figures/gnn_data_structure.png)

### Batch construction
To allow for input batches containing collections of differently sized grids, data is fed according to what in Spektral is referred to as *disjoint mode*. The interested reader can see more about this data mode in the documentation of Spektral found [here](https://graphneural.network/data-modes/#disjoint-mode). The core principle is that X and Y matrices for the batch elements are stacked vertically, while the A matrices form the blocks of a block-diagonal sparse matrix. This allows for highly parallizable and efficient sparse multiplications within the GNNs. 
Disjoint mode can accept graphs with either the same or different number of vertices/edges.

For the local MLP, the batch construction changes slightly the experiments. 
In the [first experiment](https://github.com/JonasBergHansen/Power-Flow-Balancing-with-Decentralized-Graph-Neural-Networks/blob/main/code/experiment1.py), all grids that are given as input have the same size, so a more standard batch mode is used where each input batch is a tensor with dimensions (N_batch, N_branches, N_features), where N_batch is the number of batch elements. 
For the [second](https://github.com/JonasBergHansen/Power-Flow-Balancing-with-Decentralized-Graph-Neural-Networks/blob/main/code/experiment2.py) and [third](https://github.com/JonasBergHansen/Power-Flow-Balancing-with-Decentralized-Graph-Neural-Networks/blob/main/code/experiment3.py) experiments, the grid size varies between samples, so to allow for processing of several grids simultaneously the X and Y matrices of the grids are stacked vertically and the batch size is increased such that the total number of branches  (rows) considered is roughly equal to the number processed in each batch by the GNN. For instance, if the GNN in the second experiment is trained with a batch size of 16, the local MLP is trained with a batch size of 16 x 411, where 411 is the number of branches in the unperturbed case300 grid. A similar setup to the GNN training can be used instead (batch elements chosen first and then stacked), but this will make training slower and does not improve the learning process since the local MLP only considers each row separately anyway.

## Global MLP
To achieve a global perspective for the Global MLP, the inputs and outputs are given as one-dimensional (flattened) arrays. An illustration of this data structure is given in the figure (a) below. Here, all relevant bus features are placed in an array x_bus, all relevant branch features are placed in an array x_branch, and the branch outputs are placed in y. In other words, these flattened arrays contain features from all buses/branches, which allows the MLP to utilize dependencies between them. The exact construction of these vectors, however, differ a bit between the experiments. 

In the first experiment, only "dynamic" features from the buses and branches are included in x_bus and x_branch, meaning only features that are affected by the sampling procedure used to create the dataset are included. So, if a bus does not have a shunt, shunt conductance and susceptance for this bus are not included. For the particular data used in this experiment, including these static variables would have led to an increase in model weight parameters that would not improve model performance because they would be multiplied with zero, and the increase in bias terms at the input layer would only make the model more susceptible to overfitting. In addition, slack bus indicators, which do not vary between samples of the same grid, are not included either. Any relevant information contained within such indicators can be inferred implicitly by the MLP by virtue of its global perspective and the fact that input and output units are component specific (always correspond to a specific feature for a specific bus/branch).  

In the second experiment, the data consists of perturbations of a specific grid. The data is thus structured similarly to that of the first experiment, although non-dynamic features are found from a non-perturbed reference grid. To represent the effects of the perturbations, features that are missing due to component disconnections are set to zero. So, for instance, if bus number i has been disconnected, all features related to this bus are set to zero. The same applies to the branches. During training, a masking operation is performed on the output to avoid taking into account non-present branches when computing the loss. 

![global_mlp_data_structure](/figures/global_mlp_data_structure.png)

In the third experiment, the datasets consist of samples based on a selection of different grids. To retain the topological perspective that the MLP has in the two previous experiments each input and output unit would have to correspond to a specific feature, on a specific component (bus/branch), in a specific grid (e.g. case30 or case118). This would lead to a large number of parameters that grow with the data samples and a large portion of parameters would be unutilized when processing each grid. In addition, since each parameters of the MLP is tied to a specific sample in the training set, the MLP would not be able to process grids with different topologies in the test set. 
Instead, the data in this experiment is structured such that each element of x_bus, x_branch and y does not correspond to specific components (bus/branch), but rather a certain type of feature (e.g. shunt susceptance) that can belong to a selection of buses/branches. A number of elements in the vectors are reserved to specific features, and this number is chosen such that any of the grids used for training can be used. For grids that are smaller than the maximum possible size, the vectors are zero-padded to match this size. An illustration of how this works is given in figure (b) below. Here, "input/output vector" may refer to x_bus, x_branch or y, and the uneven partition of the different feature types is meant to illustrate that these types might not be associated with the same number of maximum elements. Also, since features are filled in from the left to the right, instances of the same grid topology will occupy the same units (and thus the unit-component correspondence is in reality not completely arbitrary). Alternatively, one could scale the MLP to allow for very large grids by ramping up the number of input/output units. The issue with this approach is that if the training samples do not utilize all of the units then the spare units are never considered during training and thus serve no purpose. These extra units will therefore also not aid in the generalizability to larger, previously unseen grids.


<img src="https://github.com/JonasBergHansen/Power-Flow-Balancing-with-Decentralized-Graph-Neural-Networks/blob/master/figures/global_mlp_data_structure2.png" width="713" height="603">
