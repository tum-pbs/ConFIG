## About

This experiment is from the repository of [FAMO:Official PyTorch Implementation for Fast Adaptive Multitask Optimization (FAMO)](https://github.com/Cranial-XIX/FAMO), Please refer to the original repository for more details.

Compared to the original repository, we mainly implement the `ConFIG` method in ` methods/weight_methods.py` and `experiments/utils.py`  You can also move these files to the original repository for more experiments of MTL.

## Install requirements

Before running the experiments, you need to install all the required packages into your environments:

```bash
pip install -r requirements.txt
```



## Download dataset
To download the CelebA dataset, please refer to this [link](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg). Note that in the `Img` folder,  you only need to download `img_align_celeba.zip`.

The downloaded dataset should be moved into the dataset folder of current folder, where the hierarchy should be

```
└─MTL
    └─dataset
         └─ Anno
         └─ Eval
         └─ Img 
             └─ img_align_celeba 
```

## Run training

Use the following command to run the celeba experiments.

```bash
python trainer.py --method=config --seed=42 --gamma=0.001
```

`method` option can be replaced with other available method in the `METHODS` dict of`methods/weight_methods.py`

The f1 score for each task of each training epoch will be saved as a single `stats` file, which can be read through `torch.load()` function,  in the `save` folder.
