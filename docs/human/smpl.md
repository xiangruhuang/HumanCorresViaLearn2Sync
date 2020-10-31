# Prepare SMPL Deformation model

Download [SMPL](https://smpl.is.tue.mpg.de/) deformation model.
Look for the `.pkl` files for the female and male human models, place these two `.pkl` files as `data/smpl/female_model.pkl` and `data/smpl/male_model.pkl`.

Download pre-computed descriptors and place it under `data/smpl/embedding/` by running the following command under `data/`. 
```
wget https://www.dropbox.com/s/morip0p5h0ti54p/smpl_embedding.tar.bz2
tar xjvf smpl_embedding.tar.bz2
rm smpl_embedding.tar.bz2
```
