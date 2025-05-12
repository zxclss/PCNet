# Pivot-Centric Trajectory Prediction
> This is a PCTP implementation based on QCNet Baseline.

## Getting Started

**Step 1**: clone this repository:

```
git clone ****
# **** will be replaced with the real github link after the paper is accepted.
# For review stage, reviewer can download the repo in tar type and then 
```

**Step 2**: change the dir:

```
cd PCNet
```

**Step 3**: create a conda environment and install the dependencies:
```
conda env create -f environment.yml
conda activate PCTP
```

**Step 4**: install the [Argoverse 2 API](https://github.com/argoverse/av2-api) and download the [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html) following the [Argoverse 2 User Guide](https://argoverse.github.io/user-guide/getting_started.html).

## Training

The training process consumes ~160G GPU memory.
```
python train.py --root /path/to/dataset_root/ --train_batch_size 4 --val_batch_size 4 --test_batch_size 4 --devices 8 --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150
```
