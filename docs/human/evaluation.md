# 1. Prepare Evaluation Dataset (SURREAL, FAUST, SHREC19-Human)

## 1.1 [Prepare SURREAL Dataset](./surreal.md)

## 1.2 [Prepare FAUST Dataset](./faust.md)

## 1.3 [SHREC19-Human Dataset](./shrec19.md)

# 2. Testing the model.
We assume a checkpoint is located in `data/ckpt/HumanMultiGPU/phase2/13.pt` (this is the pretrained model checkpoint). To test the performance of the model, run
```
python -m training.multi_train --checkpoints HumanMultiGPU --warmstart latest --transf_reg --testing --dump_result
```
This will report average correspondence in euclidean distance in meters and 10cm-recall rate.
Correspondence results will be saved under `data/result/$Dataset/`, for all test datasets.
You can optionally select the dataset you would like to test, simply comment/uncomment lines in function `construct_datasets` in `training/utils.py`.

## 2.1 Test Scan to Mesh performance

### 2.1.1 Test scan to mesh performance
This error will be reported by standard output of the testing program.

## 2.2 Test Mesh to Mesh performance.
### 3.2.1 Synchronize scan-to-template correspondences and Refine.

We first synchronize scan-to-template correspondences into mesh-to-template correspondences. 
Then we run non-rigid ICP between SMPL deformation model and each mesh to refined the results.
This is achieved by running
```
python -m eval.align_mesh_corres --dataset SHREC19 --offset 0 --length 44
python -m eval.align_mesh_corres --dataset FAUST --offset 0 --length 100
```
This will dump synchronized and ICP-refined correspondences in `data/result/$DATASET/{}_{}.mesh_corres` and `data/result/$DATASET/{}_{}.mesh_corres_icp`, respectively.
Check [here](./stats.md) for a detailed explanation.

### 2.1.2 Evaluate by pairs.
We then used the refined correspondences to evaluate correspondences between pairs of mesh.
```
python -m eval.eval_corres --dataset SHREC19 --refined
python -m eval.eval_corres --dataset FAUST-inter --refined
python -m eval.eval_corres --dataset FAUST-intra --refined
```
This will dump errors and mesh-to-mesh correspondences under `data/result/$DATASET/{}_{}.errors.refined` and `data/result/$DATASET/{}_{}.refined.npy`, respectively.
To compute correspondences without ICP refinement, simply remove `--refined` from the command line, the results will then be stored in `data/result/$DATASET/{}_{}.errors` and `data/result/$DATASET/{}_{}.npy`, respectively. See [here](.stats.md) for a detailed explanation.

We then simply aggregate the statistics. For example, by
```
python -m eval.calc_stats ../data/result/SHREC19/*_*.errors.refined
```
