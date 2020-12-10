# NLOSFeatureEmbeddings Code & Datasets

This repository contains code for the paper _Learned Feature Embeddings for Non-Line-of-Sight Imaging and Recognition_ by Wenzheng Chen, Fangyin Wei, Kyros Kutulakos, Szymon Rusinkiewicz, and Felix Heide ([project webpage](https://light.cs.princeton.edu/publication/nlos-learnedfeatures/)).

## Results on Real Scenes

### Bike 
|<img src="scenes/bike_1.png" width="200" height="200" style="padding-right:20px;" />|<img src="scenes/bike_2.png" height="200" style="padding-top:333px;"/>|
|---|---| 

- Description: A white stone statue captured at approximately 1 m distance from the wall.
- Resolution: 512 x 512
- Scanned Area: 2 m x 2 m planar wall
- Integration times: 10 min., 30 min., 60 min., 180 min.

### Discoball 
|<img src="scenes/discoball_1.png" width="200" height="200" style="padding-right:20px;" />|<img src="scenes/discoball_2.png"  height="200" />|
|---|---|

- Description: A specular disoball captured at approximately 1 m distance from the wall.
- Resolution: 512 x 512
- Scanned Area: 2 m x 2 m planar wall
- Integration times: 10 min., 30 min., 60 min., 180 min.


### Dragon 
|<img src="scenes/dragon_1.png" width="200" height="200" style="padding-right:20px;" />|<img src="scenes/dragon_2.png" height="200" />|
|---|---|

- Description: A glossy dragon captured at approximately 1 m distance from the wall.
- Resolution: 512 x 512
- Scanned Area: 2 m x 2 m planar wall
- Integration times: 15 sec., 1 min., 2 min., 10 min., 30 min., 60 min., 180 min.

### Resolution 
|<img src="scenes/resolution_1.png" height="200" style="padding-right:20px;" />|<img src="scenes/resolution_2.png" height="200" />|
|---|---|

- Description: A resolution chart captured at approximately 1 m distance from the wall.
- Resolution: 512 x 512
- Scanned Area: 2 m x 2 m planar wall
- Integration times: 10 min., 30 min., 60 min., 180 min.

### Statue 
|<img src="scenes/statue_1.png" height="200" style="padding-right:20px;" />|<img src="scenes/statue_2.png" height="200" />|
|---|---|

- Description: A white stone statue captured at approximately 1 m distance from the wall.
- Resolution: 512 x 512
- Scanned Area: 2 m x 2 m planar wall
- Integration times: 10 min., 30 min., 60 min., 180 min.

### Teaser 
|<img src="scenes/teaser_1.png" height="200" style="padding-right:20px;" />|<img src="scenes/teaser_2.png" height="200" style="padding-right:20px;" />|
|---|---|

- Description: The teaser scene used in the paper which includes a number of objects, including a bookshelf, statue, dragon, and discoball.
- Resolution: 512 x 51
- Scanned Area: 2 m x 2 m planar wall
- Integration times: 10 min., 30 min., 60 min., 180 min.


The realistic scenes above are captured by [this work](https://github.com/computational-imaging/nlos-fk).

## Description of Files

The code/dataset should be organized as in the following directory tree

    ./cuda-render
        conversion/
        render/
	./DL_inference
        inference
        network7_256
        re
        utils
        utils_pytorch
	./data
        bunny-model/
        img/
        LICENSE
        README.md

## Usage

The code base contains two parts. The first part is how to render data and the second is how to train and test the neural network models.
 
### Rendering

Please check the cuda-render folder. We recommand to open it in Nsight (tested). Other IDE should also work. To compile the code, please install cuda (tested for cuda 9.0), libglm, glew, glfw and opencv (tested for opencv 3.4).

```
sudo apt-get install libglm-dev
sudo apt-get install libglew-dev
sudo apt-get install libglfw3-dev
sudo apt-get install libopencv-dev
```

To render the 3D model, first create a cuda proejct in Nsight and put everything in cuda-render/render folder to the created project and compile. To successfully run the code, modify the folder path and data saving path in [main.cpp](https://github.com/princeton-computational-imaging/NLOSFeatureEmbeddings/blob/6274ff26c31748c760414664c9f3655d7874de1a/cuda-render/render/src/main.cpp#L32). We provide a bunny model for test.

### Rendering Settings

1) Change 3D model location and scale.  We change the model size in two places. When we load a 3D model, we normalize it by moving it to the origin and load with a specific scale. The code can be modified [here](https://github.com/princeton-computational-imaging/NLOSFeatureEmbeddings/blob/6274ff26c31748c760414664c9f3655d7874de1a/cuda-render/render/src/display_4_loaddata.cpp#L337). Next, when we render the model, we may change the model location and rotation [here](https://github.com/princeton-computational-imaging/NLOSFeatureEmbeddings/blob/6274ff26c31748c760414664c9f3655d7874de1a/cuda-render/render/src/display_6_render.cpp#L361).

2) 3D model normal. For the bunny model, we use point normals. We emperically find tht it is better to use face normals for ShapeNet data set. You can change it [here](https://github.com/princeton-computational-imaging/NLOSFeatureEmbeddings/blob/6274ff26c31748c760414664c9f3655d7874de1a/cuda-render/render/src/display_4_loaddata.cpp#L464).

3) Confocal/Non-confocal renderings. Our rendering algorithm supports both confocal and non-confocal settings. One can change it [here](https://github.com/princeton-computational-imaging/NLOSFeatureEmbeddings/blob/6274ff26c31748c760414664c9f3655d7874de1a/cuda-render/render/src/display_6_render.cpp#L613), where conf=0 means non-confocal and conf=1 means confocal.

4) Specular rendering. Our rendering algorithm supports both diffuse and specular materials. To render a specular object (metal material), change the switch [here](https://github.com/princeton-computational-imaging/NLOSFeatureEmbeddings/blob/6274ff26c31748c760414664c9f3655d7874de1a/cuda-render/render/src/display_6_render.cpp#L693).

5) Video conversion. To convert a rendered hdrfile to a video, we provide a script in cuda-render/conversion. Please change the render folder [here](https://github.com/wenzhengchen/Learned-Feature-Embeddings-for-Non-Line-of-Sight-Imaging-and-Recognition/blob/dc12a8c907c7cd6392b7d3a0717ce650b07930fb/cuda-render/conversion/preprocess_hdr2video.py#L284) then run the python script. It will generate a video which is of much smaller size and easier to load to train the deep leaning model.

6) SPAD simulation. The rendered hdr file does not have any noise simulation. One can add simple Gaussian noise in datalaoder, but we recommand to employ a computational method for spad simulation to synthesize noise. We adopt the method from [here](https://graphics.unizar.es/data/spad/).

7) Rendered dataset. We provide a motorbike dataset with 3000 motorbike exmaples [here](https://drive.google.com/file/d/183VAD_wuVtwkyvfaBoguUHZgHu065BNW/view?usp=sharing).

### Rendering Examples

Non-confocal Rendering
<table>
 <tr>
  <td> t=1.2m</td> 
  <td> t=1.4m</td> 
  <td> t=1.6m</td> 
  <td> t=1.8m</td> 
  </tr>
  <tr>
   <td> <img src="./data/img/confocal/video-confocal-gray-full_120.png" width = "200px" /></td>
<td><img src="./data/img/confocal/video-confocal-gray-full_140.png" width = "200px" /></td>
<td><img src="./data/img/confocal/video-confocal-gray-full_160.png" width = "200px" /></td>
<td><img src="./data/img/confocal/video-confocal-gray-full_180.png" width = "200px" /></td>
   </tr> 
 </table>
 Confocal Rendering
 <table>
 <tr>
  <td> t=1.2m</td> 
  <td> t=1.4m</td> 
  <td> t=1.6m</td> 
  <td> t=1.8m</td> 
  </tr>
   <tr>
    <td><img src="./data/img/non-confocal/video-gray-full_120.png" width = "200px" /></td>
<td><img src="./data/img/non-confocal/video-gray-full_140.png" width = "200px" /></td>
<td><img src="./data/img/non-confocal/video-gray-full_160.png" width = "200px" /></td>
<td><img src="./data/img/non-confocal/video-gray-full_180.png" width = "200px" /></td>
  </tr>
  </table>
Specular Confocal Rendering
 <table>
  <tr>
  <td> t=1.2m</td> 
  <td> t=1.4m</td> 
  <td> t=1.6m</td> 
  <td> t=1.8m</td> 
  </tr>
  <tr>
  <td><img src="./data/img/specular/video-confocal-gray-full_120.png" width = "200px" /></td>
<td><img src="./data/img/specular/video-confocal-gray-full_140.png" width = "200px" /></td>
<td><img src="./data/img/specular/video-confocal-gray-full_160.png" width = "200px" /></td>
<td><img src="./data/img/specular/video-confocal-gray-full_180.png" width = "200px" /></td>
  </tr>
</table>


### Deep Learning Model

To run the inference model, please first download the data and pretrained model [here](https://drive.google.com/drive/folders/17KlddkUmEav-2DeDNYRD013-COqgZc0T?usp=sharing). Next, go to DL_inference/inference folder and run:

```
python eval2.py --datafolder YOUR_DATA_FOLDER --mode fk --netfolder network7_256 --netsvfolder model10_bike --datanum 800 --dim 3 --frame 128 --grid 128 --tres 2 --h 256 --w 256
```

### Deep Learning Settings

We provide our reimplementions of different NLOS methods in python and PyTorch. The python implementations are in DL_inference/utils, and the PyTorch implementations are in DL_inference/utils_pytorch. The file name starts with tf. You may directly check tflct.py, tffk.py and tfphasor.py for NLOS methods LCT (back-projection included), F-K, and Phasor, respectively.


## License  
The code and dataset are licensed under the following license:

> MIT License
> 
> Copyright (c) 2020 wenzhengchen
> 
> Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
> 
> The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
> 
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact  
Questions can be addressed to [Wenzheng Chen](mailto:chen1474147@gmail.com) and [Fangyin Wei](mailto:fwei@princeton.edu).

## Citation
If you find it is useful, please cite

```
@article{Chen:NLOS:2020,
title = {Learned Feature Embeddings for Non-Line-of-Sight Imaging and Recognition},
author = {Wenzheng Chen and Fangyin Wei and Kiriakos N. Kutulakos and Szymon Rusinkiewicz and Felix Heide},
year = {2020},
issue_date = {December 2020}, 
publisher = {Association for Computing Machinery}, 
volume = {39}, 
number = {6}, 
journal = {ACM Transactions on Graphics (Proc. SIGGRAPH Asia)}, 
}
```
