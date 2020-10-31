
# Pre-computed correspondences and error statistics (also data explanation)

Here we provide links to download the correspondence results and error statistics computed using our methods as well as explanation of data generated by our methods.

## Download packed result folder
Run the following command under `data/` folder to unpack the pre-computed `result/` folder
```
wget https://www.dropbox.com/s/i7gempm7ra4bsko/result.tar.bz2
tar xjvf result.tar.bz2
```
The unpacked folder should contains two sub-folders: `SHREC19` and `FAUST` corresponding to the SHREC19-Human and FAUST datasets. We provided detailed explanation in the following.

## 1. SHREC19
<table>
  <tr>
    <td> Files</td>
    <td> Example </td>
    <td> Descriptions</td>
  </tr>
  <tr>
    <td> result/SHREC19/*.mesh_corres </td>
    <td> result/SHREC19/44.mesh_corres </td>
    <td> Mesh to SMPL Template Correspondence before ICP Refinement </td>
  </tr>
  <tr>
    <td> result/SHREC19/*.mesh_corres_icp </td>
    <td> result/SHREC19/44.mesh_corres_icp </td>
    <td> Mesh to SMPL Template Correspondence after ICP Refinement </td>
  </tr>
  <tr>
    <td> result/SHREC19/*_*.npy </td>
    <td> result/SHREC19/35_2.npy </td>
    <td> Mesh to Mesh correspondence before ICP Refinement </td>
  </tr>
  <tr>
    <td> result/SHREC19/*_*.refined.npy </td>
    <td> result/SHREC19/35_2.refined.npy </td>
    <td> Mesh to Mesh correspondence after ICP Refinement </td>
  </tr>
  <tr>
    <td> result/SHREC19/*_*.errors </td>
    <td> result/SHREC19/35_2.errors </td>
    <td> Mesh to Mesh correspondence before ICP Refinement </td>
  </tr>
  <tr>
    <td> result/SHREC19/*_*.errors.refined </td>
    <td> result/SHREC19/35_2.errors.refined </td>
    <td> Mesh to Mesh correspondence after ICP Refinement </td>
  </tr>
</table>
We provide the computed mesh to template correspondence from each mesh in the SHREC19-Human dataset to the SMPL model, as well as mesh-to-mesh correspondences and error statistics for the official 430 pairs.

### Mesh-to-template Correspondence
We provide computed correspondences for all 44 meshes in SHREC19-Human dataset, stored in `*.mesh_corres` and `*.mesh_corres_icp`. We use standard SMPL deformation model, which contains 6890 mesh vertices.

Each `*.mesh_corres` file contains two columns of integers, the first (or second) column specifies the correspondence without (or with) transformation regularization. The `i`-th row specifies the correspondences from the `i`-th source mesh point to the SMPL template.

Each `*.mesh_corres_icp` file contains two columns of integers, the first (or second) column specifies the correspondence without (or with) transformation regularization after ICP refinement. The `i`-th row specifies the correspondences from the `i`-th source mesh point to the SMPL template.

### Mesh-to-Mesh Correspondence
We provide the computed mesh to mesh correspondence between the 430 pairs of meshes from SHREC19-Human dataset. 

Each `*_*.npy` files contains two columns of integers, the first (or second) column specifies the correspondence without (or with) transformation regularization. The `i`-th row specifies the correspondences from the `i`-th source mesh point to the target mesh.
Each `*_*.refined.npy` file contains two columns of integers, the first (or second) column specifies the correspondence without (or with) transformation regularization after ICP refinement. The `i`-th row specifies the correspondences from the `i`-th source mesh point to the target mesh.

We also provide computed error statistics for all 430 evaluation pairs.
Each `*_*.errors` files contains two columns of floats, the first (or second) column specifies the error without (or with) transformation regularization. The `i`-th row specifies the correspondence error from the `i`-th source mesh point to the target mesh.
Each `*_*.errors.refined` file contains two columns of integers, the first (or second) column specifies the correspondence without (or with) transformation regularization after ICP refinement. The `i`-th row specifies the correspondence error from the `i`-th source mesh point to the target mesh.

## 2. FAUST
<table>
  <tr>
    <td> Files</td>
    <td> Example </td>
    <td> Descriptions</td>
  </tr>
  <tr>
    <td> result/FAUST/*.mesh_corres </td>
    <td> result/FAUST/44.mesh_corres </td>
    <td> Mesh to SMPL Template Correspondence before ICP Refinement </td>
  </tr>
  <tr>
    <td> result/FAUST/*.mesh_corres_icp </td>
    <td> result/FAUST/44.mesh_corres_icp </td>
    <td> Mesh to SMPL Template Correspondence after ICP Refinement </td>
  </tr>
  <tr>
    <td> result/FAUST/*_*_inter.npy </td>
    <td> result/FAUST/70_31_inter.npy </td>
    <td> Mesh to Mesh correspondence before ICP Refinement for inter-subject pairs </td>
  </tr>
  <tr>
    <td> result/FAUST/*_*_inter.refined.npy </td>
    <td> result/FAUST/35_2_inter.refined.npy </td>
    <td> Mesh to Mesh correspondence after ICP Refinement for inter-subject pairs </td>
  </tr>
  <tr>
    <td> result/FAUST/*_*_intra.npy </td>
    <td> result/FAUST/70_31_intra.npy </td>
    <td> Mesh to Mesh correspondence before ICP Refinement for intra-subject pairs </td>
  </tr>
  <tr>
    <td> result/FAUST/*_*_intra.refined.npy </td>
    <td> result/FAUST/35_2_intra.refined.npy </td>
    <td> Mesh to Mesh correspondence after ICP Refinement for intra-subject pairs </td>
  </tr>
  <tr>
    <td> result/FAUST/*_*_inter.errors </td>
    <td> result/FAUST/35_2_inter.errors </td>
    <td> Mesh to Mesh correspondence before ICP Refinement for inter-subject pairs </td>
  </tr>
  <tr>
    <td> result/FAUST/*_*_inter.errors.refined </td>
    <td> result/FAUST/35_2_inter.errors.refined </td>
    <td> Mesh to Mesh correspondence after ICP Refinement for inter-subject pairs </td>
  </tr>
  <tr>
    <td> result/FAUST/*_*_intra.errors </td>
    <td> result/FAUST/35_2_intra.errors </td>
    <td> Mesh to Mesh correspondence before ICP Refinement for intra-subject pairs </td>
  </tr>
  <tr>
    <td> result/FAUST/*_*_intra.errors.refined </td>
    <td> result/FAUST/35_2_intra.errors.refined </td>
    <td> Mesh to Mesh correspondence after ICP Refinement for intra-subject pairs </td>
  </tr>
</table>
We provide the computed mesh to template correspondence from each mesh in the MPI-FAUST dataset to the SMPL model, as well as mesh-to-mesh correspondences and error statistics for all inter-subject and intra-subject pairs.

### Mesh-to-template Correspondence
We provide computed correspondences for all 100 meshes in MPI-FAUST dataset (training split), stored in `*.mesh_corres` and `*.mesh_corres_icp`. We use standard SMPL deformation model, which contains 6890 mesh vertices.

Each `*.mesh_corres` file contains two columns of integers, the first (or second) column specifies the correspondence not using (or using) transformation regularization. The `i`-th row specifies the correspondences from the `i`-th source mesh point to the SMPL template.
Each `*.mesh_corres_icp` file contains two columns of integers, the first (or second) column specifies the correspondence not using (or using) transformation regularization after ICP refinement. The `i`-th row specifies the correspondences from the `i`-th source mesh point to the SMPL template.

### Mesh-to-Mesh Correspondence
We provide the computed mesh to mesh correspondence between the 50 intra-subject pairs and 50 inter-subject pairs of meshes among the MPI-FAUST dataset. 
Each `*_*_inter.npy` files contains two columns of integers, the first (or second) column specifies the correspondence without (or with) transformation regularization. The `i`-th row specifies the correspondences from the `i`-th source mesh point to the target mesh.
Each `*_*_inter.refined.npy` file contains two columns of integers, the first (or second) column specifies the correspondence without (or with) transformation regularization after ICP refinement. The `i`-th row specifies the correspondences from the `i`-th source mesh point to the target mesh.

We also provide computed error statistics for all evaluation pairs.
Each `*_*_inter.errors` files contains two columns of floats, the first (or second) column specifies the error without (or with) transformation regularization. The `i`-th row specifies the correspondence error from the `i`-th source mesh point to the target mesh.
Each `*_*_inter.errors.refined` file contains two columns of integers, the first (or second) column specifies the correspondence without (or with) transformation regularization after ICP refinement. The `i`-th row specifies the correspondence error from the `i`-th source mesh point to the target mesh.

The data for intra-subject pairs follow the similar naming fashion, one just needs to replace 'inter' with 'intra'.