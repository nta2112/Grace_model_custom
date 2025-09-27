# GRACE
The source code for "Graph Few-Shot Learning via Adaptive Spectrum Experts and Cross-Set Distribution Calibration"

# Requirement
conda create -n GRACE python=3.8

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

`python main.py`

# Cite
If you find our work can help your research, please cite our work! <br>

```
@inproceedings{liu2025simple,
  title={A Simple Graph Contrastive Learning Framework for Short Text Classification},
  author={Liu, Yonghao and Wang, Yajun and Guo, Chunli and Pang, wei and Li, Ximing and Giunchiglia, Fausto and Feng, Xiaoyue and Guan, Renchu},
  booktitle={NeurIPS},
  year={2025}
}
```

# Contact
If you have any question, feel free to contact via [email](mailto:yonghao20@mails.jlu.edu.cn).
