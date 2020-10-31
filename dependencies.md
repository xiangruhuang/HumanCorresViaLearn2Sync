# Dependencies.
This project is tested on Python `3.7.7`. It depends on PyTorch `1.4+cu100`, [Pytorch Geometric (torch_geometric)](https://github.com/rusty1s/pytorch_geometric) `1.4.3` , [Open3D (open3d)](http://www.open3d.org) `0.9.0.0`, torch_sparse `0.6.1`, torch_cluster `1.5.4`, torch_scatter `2.0.4`, torch_spline_conv `1.2.0`, [sklearn](https://scikit-learn.org/stable/).

Here are some lines of script for installing dependencies.
```
$ pip install torch-scatter==0.6.1+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
$ pip install torch-sparse==1.5.4+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
$ pip install torch-cluster==2.0.4+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
$ pip install torch-spline-conv==1.2.0+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
$ pip install torch-geometric
```
Multiple GPUs are preferred for best performance.

**If you would like our code to work on different configurations, feel free to file an issue. :)**
