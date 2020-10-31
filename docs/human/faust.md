# Prepare FAUST Dataset

Download [FAUST](http://faust.is.tue.mpg.de/) dataset and placed it under `data/MPI-FAUST`.

Download Faust Intra and Inter pairs, placed under `data/faust/`.
```
wget https://www.dropbox.com/s/l1zwdjfahm49ljl/eval_pairs.mat
```

Download ground truth correspondence from FAUST meshes to SMPL templates.
```
wget https://www.dropbox.com/s/nvhqg5v7j1i1x6v/faust_mesh_smpl_correspondence.tar.bz2
tar xjvf faust_mesh_smpl_correspondence.tar.bz2
rm faust_mesh_smpl_correspondence.tar.bz2
```

Download depth scans rendered from FAUST dataset 
```
wget https://www.dropbox.com/s/qzx3iy8rexd2ebi/faust.zip
unzip faust.zip -d data
rm faust.zip
```
