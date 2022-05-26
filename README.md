# S-Prompts
Evaluation code for S-Prompts
"S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning"

# Enviroment setup
Create a conda env:
```
conda env create -f environments.yaml
```
# Getting pretrained models

Pretrained Model for DomainNet:
```angular2html
https://drive.google.com/file/d/1qDWMnxyVXNCRCmNls1lrB3k4UleON_w4/view?usp=sharing
```
Pretrained Model for CORe50:
```angular2html
https://drive.google.com/file/d/1HD7auESA89zOxdDUN0hFeAVPvOPWUAgV/view?usp=sharing
```
Pretrained Model for CDDB:
```angular2html
https://drive.google.com/file/d/1MXoBpVnGe_1aHQRLL9-VcdkDoTfeXOiM/view?usp=sharing
```



# Preparing data
Please refer to the following links to download and prepare data. 
```
DomainNet:
http://ai.bu.edu/M3SDA/
CORe50:
https://vlomonaco.github.io/core50/index.html#dataset
CDDB (need to ask the authors for the deepfake data):
https://arxiv.org/abs/2205.05467
```

After unzipping downloaded files, the file structure should be as shown below.


```angular2html
domainnet
├── clipart
│   ├── aircraft_carrier
│   ├── airplane
│   ├── alarm_clock
│   ├── ambulance
│   ├── angel
│   ├── animal_migration
│   ... ...
├── clipart_test.txt
├── clipart_train.txt
├── infograph
│   ├── aircraft_carrier
│   ├── airplane
│   ├── alarm_clock
│   ├── ambulance
│   ... ...
├── infograph_test.txt
├── infograph_train.txt
├── painting
│   ├── aircraft_carrier
│   ├── airplane
│   ├── alarm_clock
│   ├── ambulance
│   ├── angel
│   ... ...
... ...
```

```
core50
└── core50_128x128
    ├── labels.pkl
    ├── LUP.pkl
    ├── paths.pkl
    ├── s1
    ├── s10
    ├── s11
    ├── s2
    ├── s3
    ├── s4
    ├── s5
    ├── s6
    ├── s7
    ├── s8
    └── s9

```
```
DeepFake_Data
├── biggan
│   ├── test
│   ├── train
│   └── val
├── gaugan
│   ├── test
│   ├── train
│   └── val
├── san
│   ├── test
│   ├── train
│   └── val
├── whichfaceisreal
│   ├── test
│   ├── train
│   └── val
├── wild
│   ├── test
│   ├── train
│   └── val
... ...
```


# Launching experiments

[//]: # (```)

[//]: # (python eval.py --resume ./domainnet.pth --dataroot ../datasets/domainnet --datatype domainnet )

[//]: # (python eval.py --resume ./core50.pth --dataroot ../core50/data/core50_128x128 --datatype core50 )

[//]: # (python eval.py --resume ./deepfake.pth --dataroot ../DeepFake_Data/CL_data/ --datatype deepfake )

[//]: # (```)
```
python eval.py --resume ./domainnet.pth --dataroot [YOUR PATH]/domainnet --datatype domainnet 
python eval.py --resume ./core50.pth --dataroot [YOUR PATH]/core50_128x128 --datatype core50 
python eval.py --resume ./deepfake.pth --dataroot [YOUR PATH]/DeepFake_Data/ --datatype deepfake 

```
