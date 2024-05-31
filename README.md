# Codebase for Training Object Deformations using PokeFlex Dataset

## Environment
Environment setup is done using `conda`. The environment file is provided in the repository.

## Data
Get the data from: https://nas-srl.ethz.ch/#/signin

## Running a Training Session

The `main.py` script accepts the following command line arguments:

- `--log`: A boolean value indicating whether to log the training process or not. Default is `True`.
- `--batch_size`: An integer specifying the size of the batch to be used in training. Default is `8`.
- `--epochs`: An integer specifying the number of epochs for training. Default is `500`.
- `--lr`: A float specifying the learning rate for the training. Default is `1e-4`.
- `--datasets`: A list of datasets to be used for training. Default is `["Paper"]`.
- `--cameras`: A list of cameras to be used for training. Default is `["0068"]`.
- `--force_features`: A boolean value indicating whether to force features or not. Default is `False`.
- `--chamfer_weight`: A float specifying the weight of the chamfer in the loss function. Default is `1.0`.
- `--roi_chamfer_weight`: A float specifying the weight of the ROI chamfer in the loss function. Default is `0.0`.
- `--normals_weight`: A float specifying the weight of the normals in the loss function. Default is `0.0`.
- `--roi_normals_weight`: A float specifying the weight of the ROI normals in the loss function. Default is `0.0`.

```bash
python3 main.py
```






