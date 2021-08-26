# SOTR: Segmenting Objects with Transformers 

[SOTR](https://github.com/easton-cau/SOTR) fork with [SOTR_inference.py](https://github.com/deepdrivepl/SOTR/blob/main/SOTR_inference.py) showing how to run inference on video input. 

![SOTR](images/sotr.jpeg)

More details can be found at [deepdrive.pl](https://deepdrive.pl/sotr-czyli-segmentacja-instancji-z-wykorzystaniem-transformera/)

## Install packages (cu10.2 + torch 1.9)

```
conda create -n SOTR python=3.6
conda activate SOTR
pip install torch torchvision
python -m pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
pip install opencv-python gdown
```

## Clone

```
git clone https://github.com/deepdrivepl/SOTR.git
cd SOTR
```

## Download models & video

```
mkdir models
gdown 'https://drive.google.com/u/0/uc?export=download&confirm=taWO&id=1CzQTsvn9vxLnFkDJpIlitFXu1X_vw1dZ' -O models/SOTR_R101.pth
gdown 'https://drive.google.com/u/0/uc?id=19Dy6sXrwaNwGwNvuQyv5pZMWGM_at0ym&export=download' -O models/SOTR_R101_DCN.pth
mkdir data && cd data
wget https://archive.org/download/0002201705192/0002-20170519-2.mp4
```

## Inference

```
python SOTR_inference.py --model_cfg configs/SOTR/R101.yaml --model_path models/SOTR_R101.pth --video_path data/0002-20170519-2.mp4 --out_dir data/results/SOTR_R101
```

## Run ffmpeg

```
ffmpeg -i data/results/SOTR_R101/img%08d.jpg data/results/SOTR_R101.mp4
```

## Results

- [SOTR-R101](https://youtu.be/dcp7lSJ6dzk)
- [SOTR-R101-DCN](https://youtu.be/xUMUvIBe9fE)

