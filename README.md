# pytorch_dlr

## Pytorch implementation of dynamical low-rank training (DLRT)

### Installation

1. create a python virtual environment (pyenv or conda) and install pip using  ``conda install pip``. If you are using no virtual environment, please be aware of
   version incompatibilities of tensorflow.
2. Install the project requirements (example for pip):
   ``pip install -r requirements.txt``
3. In this repository there are different datasets folders, inside each one of them there are the Python scripts to train on some neural networks. The files are named as ``netname_datasetname.py`` (for DLRT) and ``netname_datasetname_baseline.py`` for the full rank standard Pytorch baseline. Each script contains its parsers to modify the training parameters (the help for the parsers is available). Results are saved in the relative folder inside the current one.
4. All the other folders (e.g. accuracy_vs_cr,UV_vanilla,Lenet_experiment,timings) contain the experiment presented in the thesis to be run. 
5. Each folder contains a produce_results file, that is producing the table or the plots presented in the paper.

### Useful links

1. The Tensorflow implementation can be found at https://github.com/CSMMLab/DLRANet
2. DLRT paper can be found at https://arxiv.org/abs/2205.13571v2


