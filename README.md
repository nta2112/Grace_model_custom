# GRACE
The source code for "Graph Few-Shot Learning via Adaptive Spectrum Experts and Cross-Set Distribution Calibration"


conda create -n GPNFSL python=3.8

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install --no-cache-dir
torch-scatter
torch-sparse
torch-cluster
torch-spline-conv
pyg-lib
torch-geometric
-f https://data.pyg.org/whl/torch-2.4.1+cu118.html

pip install scikit-learn

pip install ogb

pip install matplotlib

# Usage
Modify the experimental settings in main.py by adjusting the following parameters in the argparse configuration:

`dataset`\
`n_way`\
`k_shot`

After setting the above parameters, run the following command in the terminal:

`python main.py --use_cor`
