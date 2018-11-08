# Air-writing recognition from an image classification viewpoint

This repository contains an example code used for hand-gesture recognition from an image classification viewpoint. That means there is no previous hand-localization stage, but the class of the performed gesture is predicted from the whole image.


## Installation on Windows 10

1) Install Anaconda for Python 3.6: [Anaconda](https://www.anaconda.com/download/#download).
2) Download this repository.
3) (If not installed) Install CUDA Toolkit, Nvidia drivers, and library cuDNN for GPU support in Tensorflow. More instructions in section _Requirements to run TensorFlow with GPU support_ from [Tensorflow Installation Guide](https://www.tensorflow.org/install/install_windows?hl=es).
3) Install the conda environment and required packages: "conda env create -f tensorflow.yml".
4) Download and install "graphviz-2.38.msi" from https://graphviz.gitlab.io/_pages/Download/Download_windows.html.
5) Add the graphviz bin folder to the PATH system environment variable (Example: "C:/Program Files (x86)/Graphviz2.38/bin/")
6) Download and copy models from the link [Models](https://lima.gti.ssr.upm.es/index.php/s/1qkoHVfcnDSWaWL) to the subfolder "models".
7) Capture dataset with the project [Leap acquisition](https://github.com/cda-gti-upm/Video-Acquisition-by-mouse-events).
8) Copy dataset to the subfolder "input". The Leap air-writing dataset must have three subfolders: training, validation, and test, with the following folder structure inside each one:

```
gesture_0/
  rep_000/
    frame_000000.png
    frame_000001.png
    ...
    frame_000999.png
    
  rep_001/
  ...
  rep_NNN/
  
gesture_1/
  ...
  
gesture_3/
```
where `gesture_N` is a hand gesture type and `rep_N` is a repetition of the corresponding gesture.


## Running the code on Windows 10
### Prediction from a trained model
Execute `./windows/testme.bat`
Alternatively:
1) Run a Anaconda prompt.
2) Activate the conda environment with the command "activate tensorflow".
3) Execute:
```
python test.py --experiment_rootdir=./models ^
--weights_fname=./models/weights_064.h5 ^
--test_dir=./input/Air-writing-dataset/testing/ ^
--img_mode=grayscale
```
Note1 : Depending on your installation, you will need to write ```python3``` or just ```python``` to run the code.

### Training from scratch
Execute `./windows/trainme.bat`
Alternatively:
1) Run a Anaconda prompt.
2) Activate the conda environment with the command "activate tensorflow".
3) Execute:
```
python train.py --experiment_rootdir=./models/test_1 ^
--train_dir=./input/Air-writing-dataset/training/ ^
--val_dir=./input/Air-writing-dataset/validation/ ^
--img_mode=grayscale 
```

See more flas in `common_flags.py` to set batch size, number of epochs, dataset directories, etc. 

### Training from an existing model (fine-tuning)
1) Run a Anaconda prompt.
2) Activate the conda environment with the command "activate tensorflow".
3) Execute:
```
python train.py --restore_model=True --experiment_rootdir=./models/test_1 ^
--weights_fname=model_weights.h5 ^
--train_dir=./input/Air-writing-dataset/training/ ^
--val_dir=./input/Air-writing-dataset/validation/ ^
--img_mode=rgb 
```
where the pre-trained model called `m ./models/test_1` must be in the directory you indicate in `--experiment_rootdir`.

## FAQ
* [How to solve error: _dataset_ops.so not found](https://github.com/tensorflow/tensorflow/issues/20320)
  * Follow thes path `D:\programfiles\Anaconda3\envs\tensorflow\Lib\site-packages\tensorflow\contrib\data
` and locate `_dataset_ops.so`, then move _dataset_ops.so file out of that folder to another location.
* [ValueError: If printing histograms, validation_data must be provided, and cannot be a generator](https://github.com/aurora95/Keras-FCN/issues/50)
  * Change histogram_freq=10 to histogram_freq=0 you can still use tensor board with this fix rather than removing it all together.

