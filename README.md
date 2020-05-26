**Update: Detectron2 version of matrixnets coming out soon**

## MatrixNets

MatrixNetis a scale and aspect ratio aware deep learning architecture for object detection. We implemented matrixnets anchors (centers) and corners. For more details, please refer to the papers linked below.

<p align="center">
<img src="https://github.com/arashwan/matrixnet/blob/master/images/figure5.png" height="250px">
  </p>
  
 We have two implementations based on Corners and Anchor (Centers):
<p align="center">      <img src="https://github.com/arashwan/matrixnet/blob/master/images/figure6.png" height="200px" caption=
                             "Corners"> 
<img src="https://github.com/arashwan/matrixnet/blob/master/images/figure_centers.png" height="200px"> 

  </p>

### Training and Evaluation Code

Code for reproducing the results in the following paper:

[**Matrix Nets (ICCV'19) (short paper)**](https://arxiv.org/abs/1908.04646)  
[**Matrix Nets (long paper)**](https://arxiv.org/abs/2001.03194)


### Selecting Layers in MatrixNets

One of the capabilities offered by MatrixNets is to be able to choose which layers to use for training and inference. Although we used 19 layers matrixnet in the paper, we implemented matrixnet here such that any matrixnet design can be specified by setting the `layer_range` variable in the config file. The `layer_range` is defined as a 3D matrix were the outer matrix is 5x5, and each entry of this matrix is either a 1D matrix of [y_min, y_max, x_min, x_max] or -1 if we do not want to include this layer.

Example 1:

In the paper, we use a 19-layer MatrixNet by ignoring the left top and bottom right corners of the 5x5 matrix. The range for the base layer (top left) is [24,48,24,48].

The corresonding layer range would look like:

[[[0,48,0,48],[48,96,0,48],[96,192,0,48], -1, -1],
[[0,48,48,96],[48,96,48,96],[96,192,48,96],[192,384,0,96], -1],
[[0,48,96,192],[48,96,96,192],[96,192,96,192],[192,384,96,192],[384,2000,96,192]],
[-1, [0,96,192,384],[96,192,192,384],[192,384,192,384],[384,2000,192,384]],
[-1, -1, [0,192,384,2000],[192,384,384,2000],[384,2000,384,2000]]]

Note that we extended the range for the layers on the boundary to include any objects that are out of range.

### Performance

Following table gives the AP for Corners and Anchors with different backbones (from the paper):

|Backbone |Centers |Corners|
| -------| ---------| ------- |
|Resnet-50-X   |41.0 | 41.3 |
|Resnet-101-X  |42.3 | 42.3 |
|Resnet-152-X | 43.6 | 44.7|

Note that these numbers are reported from the validation set, whereas the final numbers in the paper are reported on the test set. 

### Sample Images with Detections for both Architectures

![alt text](https://github.com/arashwan/matrixnet/blob/master/images//figure7.png)


## Getting Started

### Installing Packages
#### Using Conda
Please first install Anaconda and create an Anaconda environment using the provided package list.
```
conda create --name matrixnets --file packagelist_conda.txt
```

After one creates the environment, activate it.
```
source activate matrixnets
```
#### Using Pip
Alternatively, one can use pip and install all packages from the requirements file. Note we are using python 3.6+. Torch 1.2.0 and torchvision 0.4.0

```
pip install -r requirements.txt
```

Our current implementation only supports GPU, so one needs a GPU and need to have CUDA(9+)  installed on your machine.

### Compiling NMS
You also need to compile the NMS code (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/cpu_nms.pyx) and [Soft-NMS](https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx)).
```
cd <Matrixnet dir>/external
make
```
### Downloading MS COCO Data
- Download the training/validation split we use in our paper from [here](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) 
- Unzip the file and place `annotations` under `<MatrixNets dir>/data/coco`
- Download the images (2017 Train and 2017 Val) from [here](http://cocodataset.org/#download)
- Create 2 directories, `train2017` and `val2017`, under `<MatrixNets dir>/data/coco/images/`
- Copy the training/validation/testing images to the corresponding directories according to the annotation files

## Training and Evaluation
To train and evaluate a network, one will need to create a configuration file, which defines the hyperparameters, and a model file, which defines the network architecture. The configuration file should be in JSON format and placed in `config/`. Each configuration file should have a corresponding model file in `models/` (specified by `model_name` in the config file). i.e. 

To train a model:
```
python train.py <config_file>
```

We provided four different configuration files under config directory for training both MatrixNetCorners and MatrixNetCenters. 

To train MatrixNets:
```
python train.py MatrixNetsCornersResnet50
```

To evaluate the trained model:
```
python test.py <config_file> --testiter <iter> --split validation
```

`--debug` flag can be used to save the first 200 images with detections under results directory.

### Pretrained Models

We provide pre-trained models for Resnet-50 and Resnet-152 for both Anchors and Corners.

[Resnet-50+Corners](https://dl.dropboxusercontent.com/s/8u1e7ctq7ppaaa5/MatrixNetCorners_Resnet50.pkl)  
[Resnet-152+Corners](https://dl.dropboxusercontent.com/s/0vubld80o9zxb7w/MatrixNetCorners_Resnet152.pkl)  
[Resnet-50+Anchors](https://dl.dropboxusercontent.com/s/zmhhvtm2wvf78qv/MatrixNetAnchors_Resnet50.pkl)  
[Resnet-152+Anchors](https://dl.dropboxusercontent.com/s/aa19s8orl5a72uk/MatrixNetAnchors_Resnet152.pkl)  


Please copy the pre-trained models into the following directory under matrixnets.

'matrixnets/<cache_dir>/nnet/<model_name>/<name>'

Here `cache_name` is the name of the directory specified in `config.json` and `name` should be in the format `<model_iters.pkl>`

Note that the results might be slightly different from the paper (+/- 0.2 MAP) since we reproduced all experiments using only 4 GPUs. We could not fit the batch size of 23 for the anchors' experiments, so we ran the experiments for longer iterations to compensate for the smaller batch size.

List of avialble configuration options:

| Option  |  Meaning | Allowed Values |
| -------| ---------| ------------- |
| dataset  | Specify standard data   | MSCOCO | 
| batch_size  |Specify batch size| At least 1|
|chunk_sizes| Size of chunk as a array of dim #GPU that sums to batch_size| |
| model_name | specifying model (also picks the sampling function with the same name) | MatrixNetsCorners, MatrixNetAnchors|
| train_split |Spcify train set|  |
| val_split | Specify Validation Set||
|opt_algo| Specify Optimization Algorithm|adam|
|learning_rate|Specify learning rate| |
|decay_rate| Specify learning rate decay| |
|max_iter| Maximum number of Iterations| |
|stepsize| Number of iterations for each learning rate decay | |
|snapshot| Snapshot interval| |
|cache_dir|directory to store snapshots| |
|data_dir| directory data is stored| | 
|rand_scale_min| Random Scaling Minimum Limt | |
|rand_scale_max| Random Scaling Maximum Limt| |
|rand_scale_step| Random Scaling Steps | |
|rand_scales| Random Scaling | |
|rand_crop| Random Cropping| |
|rand_color| Random Colouring| |
|gaussian_bump| Gaussian Bump| |
|gaussian_iou| IOU | |
|input_size| Training image size| 1d list that looks like [width, height]|
|output_kernel_size| This helps smoothing the heatmaps to get the max detections | |
|base_layer_rangge| the input size for images in matrixnetanchors. |[y_min, y_max, x_min, x_max] |
|layers_range| 3D matrix of Layer Range -1 inbdicating which layer to ignore| |
|test_image_max_dim| max dim of input image | |
|test_scales| test scales (if you want to test doing multiscale)| List of scales  |
|test_flip_images| flip flag | True, False |
|cutout| cutout flag| True, False| 
|top_k| Number of top k detections per layer| Integer | 
|categories| number of classes| |
|matching_threshold| Matching threshold | |
|nms_threshold| NMS threshold | |
|max_per_image| Max detections per image  | |
|merge_bbox| Merge bbox flag| True, False|
|weight_exp| exponential weighting specification  | |
|backbone|  Backbone for Matrix Nets| resnet50, resnet100, resnet152, resnext101  

## Contributions

Contributions to this project are welcome. Please make a pull request and we will attend to it as soon as possible. 

Also if you extend this model to other datasets or build cool projects using it we'd love to hear from you. 

## Acknowledgements

Our code is based on [CornetNets](https://github.com/princeton-vl/CornerNet)
