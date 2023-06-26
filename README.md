## Environment

conda activate cadex</b></b>
the environment of cadex has been exported into this very directory in the file: environment.yml</b></b>

## running validations

for the deformation(change config number inside the file)</b></b>

```bash
python deformValid.py --load
```

for the reconstruction(change config number inside the file)</b></b>

```bash
python evaluation_ae.py 
```



## running the trainings
example for deformation training</b></b>

```bash
python deformTrainCos5ycbEnd2End.py
```

example for reconstruction training</b></b>

```bash
python train_car_donut.py
```


## Folders
~/Desktop/CaDeX</b></b>

~/Desktop/occupancy_flow</b></b>

~/Desktop/GRNet</b></b>

~/Desktop/imagesFinal</b></b>

~/Desktop/imagesProposal</b></b>

~/Desktop/makeDataset</b></b>

~/Desktop/visulaize</b></b>

~/Desktop/ycb</b></b>

~/Desktop/detectron2</b></b>

~/Desktop/zed_camera</b></b>


## running cadex
python run.py --config ./configs/dfaust/testing/dfaust_w_pf_test_seen.yaml -f</b></b>

Cadex timing on my computer:</b></b>

During testing the batch size is 1 and the number of frames is 17:</b></b>

Model_base.py runs the val_batch function for just one batch: ([1, 17, 100000, 3])</b></b>

This contains 5 steps:</b></b>

Prediction: 0.056</b></b>

Post_process:1.5974e-5</b></b>

Dataparallel_postprocess:5.483e-6</b></b>

Post_process_after_optim:2.474→this contains mesh generation</b></b>

First mesh</b></b>

For the rest of the frames :0.365069</b></b>

The mapping to first frame for all in parallel takes 0.02</b></b>

There is also some clamping(clamp all vtx to unit cube)</b></b>

Detach_before_optim:4.4345</b></b>

So all in all for rest frames it is 0.056+0.3 which is too long(note that there is one for for which its runtime is divided by T so the runtime calculated here is for just one rest frame)</b></b>

And the reported time for the all of the rest frames in the paper is 0.68(this runtime should not be divided by the number of rest frames since some of the operations on the frames are done in parallel)</b></b>

## running Neural_Diffeomorphic_Flow--NDF
conda activate nmf</b></b>

CUDA_VISIBLE_DEVICES=0 python generate_training_meshes.py -e '/home/elham/Desktop/Neural_Diffeomorphic_Flow--NDF/pretrained/pancreas_experiments/'  --debug --start_id 0 --end_id 10 --octree --keep_normalization</b></b>

My code timing:</b></b>

item reading:  7.152557373046875e-07</b></b>

creating meshes:  0.004199981689453125</b></b>

sample points:  0.007259368896484375</b></b>

encode:  0.017279624938964844</b></b>

decode:  0.05488896369934082</b></b>

loop time:  5.7220458984375e-05</b></b>

backpass time:  0.31566452980041504</b></b>

how long did it take in all:  0.46349287033081055</b></b>


## running the occupancy flow code
conda activate see</b></b>

python generate.py configs/demo.yaml</b></b>

cd ~/hdd/occflow/occupancy_flow</b></b>

timing for first mesh:  0.5334651470184326</b></b>

rest time:  0.21221256256103516</b></b>


in order to train on the 6 ycb items with 1000 deforming sequences for each
python train.py ./configs/ycbTrain2.yml</b></b>


## Datasets
all 6 ycb objects with one deforming sequence generated for each</b></b> 

/home/elham/srl-nas/elham/watertight/ycb/ycb_mult_5_one_seq</b></b> 


just scissors with a thousand sequences of deformed objects in it</b></b>

/home/elham/srl-nas/elham/watertight/ycb/ycb_mult_1_thousand_seq</b></b>

all 6 ycb objects with a 1000 sequences for each
/home/elham/hdd/data/ycb/ycb_mult_5_thousand_seq </b></b>


## Environments
Cadex : for this repo(deformTemplate) and the cadex repo</b></b>

See : for the occflow repo</b></b>

Nmf : for the Neural_Diffeomorphic_Flow--NDF repo</b></b>

zed2: for the detectron2 repo and also the zed camera repo</b></b>




## mounting the drives
sshfs -o allow_other eli@129.132.57.251:/hdd/eli ~/hdd </b></b>

sudo mount -t cifs -o username=eaminmans,domain=D,vers=2.0  </b></b>

