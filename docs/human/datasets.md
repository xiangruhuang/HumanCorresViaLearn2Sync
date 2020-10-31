In this project, there are three datasets (SURREAL, FAUST, SHREC19), PyTorch loader code is located under `src/human_corres/datasets/`.
Below, we first illustrate the overall data layout in step 1.1, then we demonstrate the preparation of SMPL deformation model in step 1.2 and each dataset in step 1.3 to 1.5. Since SURREAL is the mainly used for training and is very space consuming, if you only need to reproduce the result using pretrained models, we recommend skipping step 1.4 (SURREAL Dataset) and step 2 (Training).

## Data Layout (not included in this repository)
Data should be stored into `data/` folder in the following layout (details below).

```
data/
|
|
|--surreal/
|  |
|  |--surreal_smpl_params.mat
|  |
|  |--scans/
|  |  |
|  |  |--0_0.mat
|  |
|  |--render_rotations/
|
|--smpl/
|  |
|  |--embedding/
|  |  |
|  |  |--descriptors.mat
|  |
|  |--model_intrinsics/
|  |  |
|  |  |--male_model.mat
|  |  |
|  |  |--female_model.mat
|  |
|  |--templates/
|
|--faust/
|  |
|  |--eval_pairs.mat
|  |
|  |--scans/
|
|--MPI-FAUST/
|  |
|  |--training/
|  |
|  |--testing/
|
|--SHREC19/
|  |
|  |--mesh/
|  |
|  |--scans/
```

## [Prepare SMPL deformation model](./smpl.md)
