---
This repository provides code for solving PF with Graph Neural Networks. In particular, the code can be used to reproduce the experimental results presented in the paper [Power Flow Balancing with Decentralized Graph Neural Networks](https://arxiv.org/abs/2111.02169).

---
## Setup

The code is based on Python 3.9.7, TensorFlow 2.5.0, and [Spektral]() 1.0.6. 

The code can be executed by installing an Anaconda environment from the `environment_{os}.yml`:
```bash
conda env create -f environment_{os}.yml
``` 

The YAML file can be found [here](/environment_files/environment_windows.yml) for Windows 10 and [here](/environment_files/environment_linux.yml) for Linux (made with Ubuntu 20.04).



---
## Files
The datasets used in the experiments can be found [here](https://mega.nz/folder/LShW3AJL#UuTVSz-VdDmncdOVXc4NNQ).
The data is stored as .npy files and must therefore be collected using numpy.load. Moreover, some of these files contain arrays of different sizes or scipy.sparse matrices and are thus stored as numpy objects. To load these, set allow_pickle = True when loading. Examples for how this is done can be seen in the experiment scripts in the [code folder](/code). 

Note that folders named graph_structured contain the branch/line graph representations of the power grids. Here, files starting with X contain graph vertex features (meaning bus and branch features), files starting with A contain the binary adjacency matrices for the line graphs and files starting with y contain the branch PF solutions.

Folders named flattened contain the data representations used by the Global MLP, which come in form of flattened one-dimensional arrays. Here, files starting with X contain bus features, those starting with E contain bus features and those starting with y contain branch PF solutions. 

A more in-depth explanation of the strucure of the data is given [here](/data/data_setup.md).

---
## Execution
To execute the experiments, the data found in the above link must be placed inside the [data folder](/data). The experiments can then be executed by running the script files in the [code folder](/code). So, for instance, to perform the training and testing for the first experiment from the terminal, navigate to the code folder and execute the following

```bash
python experiment1.py
```

Parameters such as batch size, learning rate and number of training epochs can be adjusted within the scripts.

---
## Citation

Please, cite the original paper if you are using our PF solver in your research

    @inproceedings{hansen2021power,
    title={Power Flow Balancing with Decentralized Graph Neural Networks}, 
    author={Jonas Berg Hansen and Stian Normann Anfinsen and Filippo Maria Bianchi},
    year={2021},
    eprint={2111.02169},
    archivePrefix={arXiv},
    primaryClass={cs.LG}}
