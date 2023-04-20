import numpy as np
import scipy.sparse as sp

# Construct line graph adjacency matrix
def construct_line_graph_adjacency(from_to_array):
    """
    Construct binary line/branch graph adjacency matrix.

    Parameters
    ----------
    from_to_array : numpy array
        Two-dimensional array containing node index pairs for branches as rows.

    Returns
    -------
    A : numpy array
        Binary branch graph adjacency matrix.

    """
    arr = from_to_array
    N_branches = from_to_array.shape[0]
    A = np.zeros([N_branches, N_branches], dtype = np.int32)
    
    for i in range(N_branches):
        A[i, np.where((arr[:, 0] == arr[i,0]) | (arr[:, 0] == arr[i,1]))] = 1
        A[i, np.where((arr[:, 1] == arr[i,0]) | (arr[:, 1] == arr[i,1]))] = 1
                        
    A = A - np.eye(N_branches, dtype = np.int32)
    
    return A


def compute_currents(p, q, vm, va):
    """
    Compute real and imaginary branch current injection at branch end.

    Parameters
    ----------
    p :  numpy array
        Active power injections.
    q :  numpy array
        Reactive power injections.
    vm : numpy array
        Voltage magnitudes.
    va : numpy array
        Voltage phase angles.

    Returns
    -------
    numpy array
        Real and imaginary current injections.

    """
    # Convert from degrees to radians
    va *= np.pi/180
    # Compute apparent power
    S = p + 1j*q
    # Compute complex voltage
    V = vm * np.exp(1j*va)
    # Compute complex current
    I = np.conj(S/V)
    
    return np.vstack([np.real(I), np.imag(I)]).T



def build_output_targets(n, f_out, branch_solns, bus_solns, idx_from,
                            idx_to):
    """
    Construct output target array for branch current and power injections

    Parameters
    ----------
    n : int
        Number of branches.
    f_out : int
        Number of output variables.
    branch_solns : numpy array
        Branch solutions given by MATPOWER (e.g. from ACPF or DCPF).
    bus_solns : numpy array
        Bus solutions given by MATPOWER.
    idx_from : numpy array
        Indices indicating which bus is located at the "from" end of each branch.
    idx_to : numpy array
        Indices indicating which bus is located at the "to" end of each branch.

    Returns
    -------
    y_i : numpy array
        Branch current and power injections

    """
    ## Construct label feature matrix for branches
    y_i = np.zeros([n, f_out])
    # Add real and reactive power injected at "from" end
    y_i[:, [0, 1]] = branch_solns[:, [13, 14]]
    # Add real and reactive current at "from" end
    y_i[:, [2, 3]] = compute_currents(y_i[:, 0], 
                                      y_i[:, 1], 
                                      bus_solns[:, 7][idx_from], 
                                      bus_solns[:, 8][idx_from])
    # Add real and reactive power injected at "to" end
    y_i[:, [4, 5]] = branch_solns[:, [15, 16]]
    # Add real and reactive current at "from" end
    y_i[:, [6, 7]] = compute_currents(y_i[:, 4], 
                                      y_i[:, 5], 
                                      bus_solns[:, 7][idx_to], 
                                      bus_solns[:, 8][idx_to])
    return y_i



def construct_gnn_dataset(mpc_data, acpf_data, dcpf_data):
    """
    Construct GNN branch graph dataset from MATPOWER case files.

    Parameters
    ----------
    mpc_data : numpy array
        Grid data pre ACPF or DCPF solver.
    acpf_data : numpy array
        Grid data post ACPF solver.
    dcpf_data : numpy array
        Grid data post DCPF solver

    Returns
    -------
    X : list
        List containing graph node input feature matrices.
    A : list
        List containing sparse binary adjacency matrices.
    y : list
        List containing branch label matrices given by ACPF.
    y_dcpf : list
        List containing branch label matrices given by DCPF.

    """
    # Number of branch features
    f = 21
    f_out = 8
    X = []
    A = []
    y = []
    y_dcpf = []
    
    N = mpc_data.shape[1]
        
    for i in range(N):
        # Extract relevant bus, generator and branch features
        bus_data = mpc_data[0, i]['bus'][0, 0]
        branch_data = mpc_data[0, i]['branch'][0, 0]
        
        branch_solns = acpf_data[0, i]['branch'][0, 0]
        bus_solns = acpf_data[0, i]['bus'][0, 0]
        gen_data = mpc_data[0, i]['gen'][0, 0]
        
        branch_dcpf_solns = dcpf_data[0, i]['branch'][0, 0]
        bus_dcpf_solns = dcpf_data[0, i]['bus'][0, 0]
        
        # Base power value
        base_mva = mpc_data[0, i]['baseMVA'][0, 0][0][0]
        
        # Convert all power quantities to per unit
        bus_data[:, [2, 3, 4, 5]] /= base_mva
        gen_data[:, 1] /= base_mva
        branch_solns[:, [13, 14, 15, 16]] /= base_mva
        branch_dcpf_solns[:, [13, 14, 15, 16]] /= base_mva
        
        
        # Number of branches
        n_branch = branch_data.shape[0]
        
        # Bus indices for branch connections
        idx_from = [np.where(bus_data[:, 0] == branch_data[k, 0])[0][0] for k in range(n_branch)]
        idx_to = [np.where(bus_data[:, 0] == branch_data[k, 1])[0][0] for k in range(n_branch)]
        # Bus indices for generators
        idx_gen = [np.where(bus_data[:, 0] == gen_data[k, 0])[0][0] for k in range(len(gen_data[:,0]))]
        # Index for slack bus
        idx_ref = np.where(bus_data[:, 1] == 3.0)
        
        # Generator active power injection for each bus
        P_g = np.zeros(bus_data.shape[0])
        P_g[idx_gen] = gen_data[:, 1]
        
        # Generator voltage magnitude set point for each bus
        V_g = np.zeros(bus_data.shape[0])
        V_g[idx_gen] = gen_data[:, 5]
        
        ## Construct graph node feature matrix
        X_i = np.zeros([n_branch, f])

        # Add resistance, reactance and charging susceptance
        # Also add transformer tap ratio and shift angle
        X_i[:, [0, 1, 2, 3, 4]] = branch_data[:, [2, 3, 4, 8, 9]]
        # Convert shift angle to radians
        X_i[:, 4] *= np.pi/180
        
        # Add values related to the buses at the "from" end of the branches
        # Real power demand(load)
        X_i[:, 5] = bus_data[:, 2][idx_from]
        # Reactive power demand
        X_i[:, 6] = bus_data[:, 3][idx_from]
        # Shunt conductance 
        X_i[:, 7] = bus_data[:, 4][idx_from]
        # Shunt susceptance
        X_i[:, 8] = bus_data[:, 5][idx_from]
        # Voltage magnitude set point
        X_i[:, 9] = V_g[idx_from]
        # Real power set point
        X_i[:, 10] = P_g[idx_from]
        # Reference bus indicator
        X_i[np.where(idx_from == idx_ref[0][0]), 11] = 1
        # Non-ref bus indicator
        X_i[np.where(idx_from != idx_ref[0][0]), 12] = 1
        
        # Add values related to the buses at the "to" end of the branches
        # Real power demand(load)
        X_i[:, 13] = bus_data[:, 2][idx_to]
        # Reactive power demand
        X_i[:, 14] = bus_data[:, 3][idx_to]
        # Shunt conductance 
        X_i[:, 15] = bus_data[:, 4][idx_to]
        # Shunt susceptance
        X_i[:, 16] = bus_data[:, 5][idx_to]
        # Voltage magnitude set point
        X_i[:, 17] = V_g[idx_to]
        # Real power set point
        X_i[:, 18] = P_g[idx_to]
        # Reference bus indicator
        X_i[np.where(idx_to == idx_ref[0][0]), 19] = 1
        # Non-ref bus indicator
        X_i[np.where(idx_to != idx_ref[0][0]), 20] = 1
        
    
        ## Construct line graph adjacency matrix
        A_i = construct_line_graph_adjacency(branch_data[:, [0,1]])
        
        # Output targets from ACPF
        y_i = build_output_targets(n_branch, f_out, branch_solns, 
                                      bus_solns, idx_from, idx_to)
        
        # Output targets from DCPF
        y_dcpf_i = build_output_targets(n_branch, f_out, branch_dcpf_solns, 
                                           bus_dcpf_solns, idx_from, idx_to)
        

        ## Add to lists
        X.append(X_i.astype(np.float32))
        A.append(sp.coo_matrix(A_i.astype(np.float32)))
        y.append(y_i.astype(np.float32))
        y_dcpf.append(y_dcpf_i.astype(np.float32))
        
        print("It: {}".format(i))
        
    return X, A, y, y_dcpf
