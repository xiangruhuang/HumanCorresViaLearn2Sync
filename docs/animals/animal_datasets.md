# 0. Overview
  We generated `20K` animal shapes for `horse` and `cat` class using SMAL deformation model. Each animal shape is rendered from 100 random viewpoints, resulting in `2,000K` depth scans. We evaluated our results on TOSCA dataset.
  To reproduce our evaluation results, we need to download three things:
    pretrained.model
  
# 1. Download Generated Animal Shapes
 
## 1.1 Download generated animal meshes.
Run `./initialize_smal.sh` under folder `data/` to download and unpack the generated meshes.

## 1.2 Download or Generate Rendered Partial Scans for Animal Shapes

To generate the partial scans, run under folder `src` to initialize the rendering configuration, stored in `data/json/smal_rendering.json`.
```
python -m hybrid_corres.datasets.smal_generate_json
mkdir -p ../data/smal_scans
```

Then run the following script (preferably in parallel with non-overlapping interval) to render all meshes (40K in total).
```
python -m geop.rendering.render_mesh --json_file ../data/json/smal_rendering.json --offset 0 --length 10
```
