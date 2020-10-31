# Rendering Depth Scans from Meshes

Here we illustrate how to render depth scans from generated mesh. In fact, this is a slow yet generalizable renderer. You can also use your own rendering engine for this.

## Step 1. Generating JSON files for each dataset.
We first generate a JSON file for the rendering machine. For example, for SHREC19-Human dataset, this can be done via
```
python -m human_corres.datasets.shrec19_generate_json
```
This will generate a JSON file under `data/json/shrec19_rendering.json`.
Please also checkout `human_corres/datasets/smal_generate_json.py`.

## Step 2. Rendering.
Run the following command under `src/` to render depth scans under `../data/SHREC19/scans`.
```
python -m geop.rendering.render_mesh --json_file ../data/json/shrec19_rendering.json --offset 0 --length 44
```
