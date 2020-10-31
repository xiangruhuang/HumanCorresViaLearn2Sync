# 1. [Prepare Training Data (SURREAL dataset)](./surreal.md)

# 2. Training the model
After data is placed correctly, go to `src/`.
The training is split into two phases. In the first phase, we only train the feature extraction network with the following command.
```
python -m training.multi_train --checkpoints HumanMultiGPU --embed_dim 100 --desc Laplacian_n --warmstart latest
```
where checkpoints will be saved under `HumanMultiGPU/phase1/*.pt`.  
**Note** that `training.multi_train` will take multiple gpus (if available), if training is restricted to single gpu, please use its equivalent `training.train`.

In the second phase, we train our entire pipeline from the warmstart checkpoint obtained in the first phase.
```
python -m training.multi_train --checkpoints HumanMultiGPU --embed_dim 100 --desc Laplacian_n --warmstart latest --transf_reg
```
where checkpoints will be saved under `HumanMultiGPU/phase2/*.pt`.

During training, errors information will be dumped from python `progressbar` like the following lines:
```
ETA:   0:00:06 training, loss: 0.00104, l2: (0.028626, 0.022934), recall: ((0.944320, 0.990652), (0.962435, 0.993666)), count:   3334,
```
where the two numbers after `l2` represent the average correspondence error before and after transformation regularization,
similarly for the four numbers after `recall`, the first two numbers represent the 5cm-recall rate and 10cm-recall rate **before** transformation regularization, and
the last two numbers represent the 5cm-recall and 10cm-recall **after** transformation regularization.
 
