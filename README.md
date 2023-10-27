# specularflow_highlight_deepNet


This repository provides data and codes used in the paper *Specular Surface Detection with Deep Static Specular Flow and Highlight* submitted to a journal and *Combining Static Specular Flow and Highlight with Deep Features for Specular Surface Detection* presented at MVA2023.

## Requirement
- Python 3
- PyTorch>=1.7.1

## Dataset
- Spherical mirror dataset: [Google Drive](https://drive.google.com/file/d/1ihubgbLd-IT8EQgjJ4EwJiYP-MSMA3IT/view?usp=sharing)  
consists of 131 pairs of an image of a spherical mirror and its corresponding mask.
Images were captured in various indoor (living and bedrooms) and outdoor (garage and garden) environments by changing the distance and angle between the camera and the spherical mirror so that the mirror reflects diverse textures.
Images captured in the indoor and garage were used for training, and the ones in the garden were used for testing.
The following table lists the breakdown numbers of images for training and testing.  

|  type  |  breakdown  |
|:---- |:----:|
|  train  |  104  |
|  test  |  27  |

- Plastic mold dataset: [Google Drive (restricted to registered users only)](https://drive.google.com/file/d/1rBLSxiyrWLV4eOzFQwJCbrcHaUaghVDy/view?usp=drive_link)  
consists of 189 pairs of an image of mirror-polished molds and their corresponding mask.
There are eight different types of molds, and images are shot in real factory production lines.
The following table lists the breakdown number of images for each type of mold.
Images of each type of mold are used for testing, and the remaining ones are used for training, i.e., 8-fold cross-validation.  

|  type  |  breakdown  |
|:----|:----:|
|  movable small case  |  18  |
|  small case  |  13  |
|  coin case  |  21  |
|  nameplate  |45|
|  movable hard case | 12|
|  hard case  |21|
|  number tag | 41|
|  coin dish  |18|


## Data structure
-  Unzipped folders (spherical_mirror_dataset and plastic_mold_dataset) downloaded from google drives are placed under ./data folder. 
- In each subfoloder, e.g., coin_case and test,  there are *image* and *mask* foldders containing images and corresponding mirror mask.

```
  data/
    ├── plastic_mold_dataset/
    │   ├── coin_case/
    │   │   ├── image/
    │   │   │   ├── img1.jpg
    │   │   │   ├── img2.jpg
    │   │   │   └── ...
    │   │   └── mask/  
    │   │       ├── img1.png
    │   │       ├── img2.png
    │   │       └── ...
    │   ├── coin_tray/
    │   │   └── ...
    │   ├── hard_case/
    │   │   └── ...
    │   ├── hard_case_movable/
    │   │   └── ...
    │   ├── nameplate/
    │   │   └── ...
    │   ├── number_tag/
    │   │   └── ...
    │   ├── small_case/
    │   │   └── ...
    │   └── small_case_movable/
    │       └── ...
    │ 
    └── spherical_mirror_dataset/
        ├── test/
        │   ├── image/
        │   |    ├── img1.jpg
        │   |    ├── img2.jpg
        │   |    └── ...
        |   └── mask/  
        │       ├── img1.png
        │       ├── img2.png
        │       └── ...
        └── train/
            └── ...
```


## Installation
To get started with the project, follow these steps:
1. Clone the repository.
```bash
$ git clone https://github.com/yossi-yuto/specularflow_highlight_deepNet.git
```
2. Install the required packages from the ```requirements.txt``` file using pip:
```bash
$ pip install -r requirements.txt
```

## How to run
To train and test the model on the plastic mold dataset, run:
```bash 
$ source pipeline_mold.sh {GPU_NUM} {date}
```
To train and test the model on the spherical mirror dataset, run:
```bash
$ source pipeline_spherical.sh {GPU_NUM} {date}
```
where `{GPU_NUM}` is the GPU device number and `{date}` is the execution date. 

### Example
To conduct an experiment on the Spherical mirror dataset using GPU device 0 on September 23, execute the following command:
```bash
$ source pipeline_spherical.sh 0 0923
```


## Citation
```
@inproceedings{mva:hachiya:2023,
  title={Combining Static Specular Flow and Highlight with Deep Features for Specular Surface Detection},
  author={Hirotaka Hachiya and Yuto Yoshimura},
  booktitle={Proceedings of 18th International Conference on Machine Vision Applications (MVA)},
  year={2023}
}
