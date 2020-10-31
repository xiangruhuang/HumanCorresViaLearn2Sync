Here we illustrate how to reproduce our experiments for human shapes.

# 1. [Prepare Datasets](./animal_datasets.md)

# 2. Training.
### 2.1 Training phase1 using multi-GPU (recommended)
Run this command on multiple gpus via `nn.DataParallel`
```
python -m training.multi_train --embed_dim 100 --checkpoints TrainOnSMAL --warmstart latest --desc laplacian --epochs 10 --batch_size 9 --animals
```
Run on single-gpu.
```
python -m training.train --embed_dim 100 --checkpoints TrainOnSMAL --warmstart latest --desc laplacian --epochs 10 --batch_size 9 --animals
```
Run on CPU only.
```
python -m training.train --embed_dim 100 --checkpoints TrainOnSMAL --warmstart latest --desc laplacian --epochs 10 --batch_size 9 --animals --cpu
```
Output will look like
```
ETA:  1 day, 20:10:18 training, loss: 0.00221, l2: (0.047981, 0.000000), recall: ((0.663879, 0.918080), (0.000000, 0.000000)), count:  78500,
```
which indicates the `loss`, the `l2` correspondence distance, the `recall` within `0.05` and `0.1` before regularization, and the recall after regularization. Checkpoints will be stored in `TrainOnSMAL/phase1` in this example.

# 3. Testing.
### 3.1 Test Partial Scan Correspondence.
This is similar to the experiment for human shapes except adding `--animals` in the command line.

### 3.2 Test Mesh to Mesh performance.
This is similar to the experiment for human shapes except adding `--animals` in the command line.
