# Air-writing recognition from an image stream input

This repository contains an example code used for air-writing recognition from an image input viewpoint. That means there is no previous hand-localization stage, but the class of the performed gesture is predicted from the whole video sequence.


## Installation on Windows 10

1) Install Anaconda for Python 3.6: [Anaconda](https://www.anaconda.com/download/#download).
2) Download this repository.
3) (If not installed) Install CUDA Toolkit, Nvidia drivers, and library cuDNN for GPU support in Tensorflow. More instructions in section _Requirements to run TensorFlow with GPU support_ from [Tensorflow Installation Guide](https://www.tensorflow.org/install/install_windows?hl=es).
3) Install the conda environment and required packages: "conda env create -f tensorflow.yml".
4) Download and install "graphviz-2.38.msi" from https://graphviz.gitlab.io/_pages/Download/Download_windows.html.
5) Add the graphviz bin folder to the PATH system environment variable (Example: "C:/Program Files (x86)/Graphviz2.38/bin/")
6) Create the subfolder "models".
7) Develope dataset from the link [Leap Motion writing acquisition](https://github.com/cda-gti-upm/Video-Acquisition-by-mouse-events) to the subfolder "input". The final dataset will have with the following folder structure:

```
gesture_1/
  repetition_1/
    frame_000000.png
    frame_000001.png
    ...
    frame_000999.png
    
  repetition_2/
  ...
  repetition_7/
  
gesture_2/
  ...
  
gesture_N/
```
where `repetition_N` is a sample folder and `class_N` is a writing gesture type.


## Running the code on Windows 10
### Prediction from a trained model
Execute `./windows/testme.bat`
Alternatively:
1) Run a Anaconda prompt.
2) Activate the conda environment with the command "activate tensorflow".
3) Execute:
```
python ../test.py --experiment_rootdir=../models ^
--weights_fname=../models/test_4/weights_015.h5 ^
--img_mode=rgb
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
--img_mode=rgb 
```

See more flas in `common_flags.py` to set batch size, number of epochs, dataset directories, etc. 

### Training from an existing model (fine-tuning)
1) Run a Anaconda prompt.
2) Activate the conda environment with the command "activate tensorflow".
3) Execute:
```
python train.py --restore_model=True --experiment_rootdir=./models/test_1 ^
--weights_fname=models/weights_015.h5 ^
--img_mode=rgb 
```
where the pre-trained model called `m ./models/test_1` must be in the directory you indicate in `--experiment_rootdir`.
