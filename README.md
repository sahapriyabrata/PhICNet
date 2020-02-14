# PhICNet

Repository for our work **PhICNet: Physics-Incorporated Convolutional Recurrent Neural Networks for Modeling Dynamical Systems**

## Installation

Compatible with Python 3.5 and Pytorch 1.1.0

1. Create a virtual environment by `python3 -m venv env`
2. Source the virtual environment by `source env/bin/activate`
3. Install requirements by `pip install -r ./requirements.txt`

## Usage

Dataset and pre-trained models can be downloaded from these two links: [dataset](http://bit.ly/31TPu2I) and [models](http://bit.ly/2uAov0g).

#### Training
To train heat system, run `python scripts/train_heat_system.py --dataset ./dataset/train_heat_maps.npy`

To train wave system, run `python scripts/train_wave_system.py --dataset ./dataset/train_wave_maps.npy`


#### Evaluation
To evaluate heat system, run `python scripts/test_heat_system.py --dataset ./dataset/test_heat_maps.npy --model_path ./saved_models/heat_system/model.ckpt  --param_path saved_models/heat_system/parameters.ckpt`

To evaluate wave system, run `python scripts/test_wave_system.py --dataset ./dataset/test_wave_maps.npy --model_path ./saved_models/wave_system/model.ckpt  --param_path ./saved_models/wave_system/parameters.ckpt`

