# Generate FAUST Rendered Depth Scans
1. Make sure FAUST Mesh models are stored in `../data/MPI-FAUST`.
2. Run the following script to render
```
python -m hybrid_corres.datasets.generate_faust_scans
```
Depth scans will be stored in `../data/faust/scans/`.

# Generate SURREAL Depth Scans 
1. Make sure generated SURREAL SMPL parameters are stored in `../data/surreal/surreal_smpl_params.mat`.
2. Run the following program to generate scans for mesh id `[10000, 20000)`.
```
python -m hybrid_corres.datasets.generate_surreal_scans --offset 10000 --length 10000
```
Depths scans will be stored in `../data/surreal/scans/`.

# Headless Rendering:

Create Conda Environment:
```
conda create -n headless python=3.7
conda activate headless
```
Run `which python` to get the path of python executable, for example
```
which python
/home/xxxxx/anaconda3/envs/headless/bin/python
```

Install OSMesa:
```
wget https://mesa.freedesktop.org/archive/mesa-19.0.8.tar.xz
tar xvf mesa-19.0.8.tar.xz
cd mesa-19.0.8
./configure --prefix=$HOME/osmesa \
  --disable-osmesa --disable-driglx-direct --disable-gbm --enable-dri \
  --with-gallium-drivers=swrast --enable-autotools --enable-gallium-osmesa
make -j9
make install
```
Add the following to `.bashrc`:
```
export LD_LIBRARY_PATH="$HOME/osmesa/lib:$LD_LIBRARY_PATH"
```

Then build Open3D from source, note that python executable path should be 
set to the result of `which python`
```
cd ~/Open3D
mkdir build&&cd build
cmake -DENABLE_HEADLESS_RENDERING=ON -DBUILD_GLEW=ON -DBUILD_GLFW=ON \
  -DOSMESA_INCLUDE_DIR=$HOME/osmesa/include \
  -DOSMESA_LIBRARY="$HOME/osmesa/lib/libOSMesa.so" \
  -DPYTHON_EXECUTABLE:FILEPATH=/home/xxxxx/anaconda3/envs/headless/bin/python \
  ..
make -j9
make install-pip-package
```

```
python -m training.train_reg --FEnet PointNet2 --checkpoints PointNet2Reg --val True --batch_size 1 --warmstart 8
```
