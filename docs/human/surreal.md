# Prepare SURREAL dataset

Since SURREAL is the mainly used for training and is very space consuming, we split it into training dataset and test dataset.

### 1. SURREAL Training Split
If you only need to reproduce the result using pretrained models, we recommend skipping this step and go to step 2.

Download generated SURREAL Scans (training split) for 200K SMPL parameters indexed from in `[0, 100000) \union [115000, 215000)`
```
wget https://www.dropbox.com/s/erm8llhehqp9xs3/scans0_50000.tar
wget https://www.dropbox.com/s/q7rdlvkxecx9b5r/scans50000_99999.tar
wget https://www.dropbox.com/s/9wyrsjxbzh0fum4/11xxxx.tar.bz2
wget https://www.dropbox.com/s/0nzrncdoav9cov1/12xxxx.tar.bz2
wget https://www.dropbox.com/s/kbh6o1xfbe0e8g4/13xxxx.tar.bz2
wget https://www.dropbox.com/s/k7c6d0p481bcay3/14xxxx.tar.bz2
wget https://www.dropbox.com/s/r74w43y1neqtltq/15xxxx.tar.bz2
wget https://www.dropbox.com/s/bd1ajqbbko3sb6b/16xxxx.tar.bz2
wget https://www.dropbox.com/s/8pjrtxpx7oort3f/17xxxx.tar.bz2
wget https://www.dropbox.com/s/40nhzoe0n9mdg5e/18xxxx.tar.bz2
wget https://www.dropbox.com/s/30j97qxwwfwnzye/19xxxx.tar.bz2
wget https://www.dropbox.com/s/figof2ft8bnfhok/20xxxx.tar.bz2
wget https://www.dropbox.com/s/tijcbhhh9ilktum/21xxxx.tar.bz2
```
Note that each SMPL mesh model is rendered from 20 different view points.
This data should be placed like `'data/surreal/scans/{0:06d}/{1:03d}.mat'.format(mesh_index, view_index)`.
Please be aware that this is VERY LARGE.

### 2. SURREAL Test Split
Download SURREAL Scans (testing split) for 5K scans indexed in `[95000, 100000) \union [210000, 215000)`.

```
cd data/
wget https://www.dropbox.com/s/gzzn4q32v438eh5/surreal-test.tar.bz2
tar xjvf surreal-test.tar.bz2
rm surreal-test.tar.bz2
```
Note that each SMPL mesh model is rendered from 100 different view points.
This data should be placed like `'data/surreal-test/scans/{0:06d}/{1:03d}.mat'.format(mesh_index, view_index)`.

To download a test set of 1K test scans
```
wget https://www.dropbox.com/s/tb96zi86y7vvam3/test_scans.tar.bz2
tar xjvf test_scans.tar.bz2
```

### 3. Generating SURREAL from SMPL parameters
If you need to generate your own SURREAL mesh, please follow this step.
Download generated SURREAL SMPL parameters (Needed for loading mesh-to-mesh matching task)
```
wget https://www.dropbox.com/s/qg31f60pb4ezk3q/surreal_smpl_params.mat
mkdir -p data/surreal/
mv surreal_smpl_params.mat data/surreal/
```

SMPL model parameters are interpreted as follows
```
Each mesh is described in 83 numbers: sequentially gender (1) + betas (10) + poses (72). 
There are 230000 meshes generated using standard male and female model parameters. They are ordered sequentially as female normal (100000) + female bent (15000) + male (100000) + male bent (15000).
```
See `human_corres/smpl/smpl_np.py` for generating meshes from SURREAL SMPL parameters.

To render depth scans from SURREAL mesh, see `./docs/rendering.md`
