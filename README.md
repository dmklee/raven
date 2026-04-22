# RAVEN
Official code for "[RAVEN: End-to-end Equivariant Robot Learning with RGB Images](https://openreview.net/pdf?id=z8BN7KyaPl)" paper accepted to ICLR'26.

## Setup
1. Install the following dependencies:
```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

2. Create virtual environment with [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
```bash
micromamba create -f env.yml 
```

3. Activate virtual environment and install other dependencies:
```bash
micromamba activate mimicgen_env
micromamba update ffmpeg
pip install -e .

git clone https://github.com/NVlabs/mimicgen_environments.git
cd mimicgen_environments
git checkout 081f7dbbe5fff17b28c67ce8ec87c371f32526a9
pip install -e .

git clone https://github.com/dmklee/robomimic.git
cd robomimic
pip install -e .
```

## Replicating MimicGen Experiments
1. Download and prepare demo datasets.  This will download the hdf5 files, render image observations for all states, and convert to absolute actions.  It may take 10 minutes per task.
```bash
bash data/mimicgen_prep.sh
```

2. Train models (results logged in WandB). Change the `task_name`to generate results for all mimicgen tasks.
```
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python -m uferl.train \
  --config-name=train_raven_fm_rel \
  task=mimicgen_abs
  obs=agent_and_inhand
  training.seed=0 \
  n_demo=100 \
  task_name=stack_three_d1
```

## Replicating DexMimicGen Experiments
1. Create a different virtual environment.  Above, we used an older version of robomimic to be compatible with the baselines.  For dexmimicgen, we use a different version so we provode a separate environment.
```bash
micromamba create -f dex_env.yml 
micromamba activate dexmimicgen_env
micromamba update ffmpeg
pip install -e .

git clone https://github.com/ARISE-Initiative/robosuite.git
pip install robosuite/

git clone https://github.com/NVlabs/dexmimicgen.git
pip install -e dexmimicgen
```

2. Download and prepare demo datasets.  This will download the hdf5 files, render image observations for all states, and convert to absolute actions.  It may take 10 minutes per task.
```bash
bash data/dexmimicgen_prep.sh
```

3. Train and eval models. For dexterous tasks like `task_name=two_arm_box_cleanup`, set `obs=agent_and_two_hands_dexterous` (which expands the action space to include finger joints). 
```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python -m uferl.train \
  --config-name=train_raven_fm_rel \
  task=mimicgen_abs
  obs=agent_and_two_inhands
  training.seed=0 \
  n_demo=50 \
  task_name=two_arm_threading
```

## Citation
If you would like to cite our work, please use the following bibtex:
```bibtex
@inproceedings{kleeraven,
  title={RAVEN: End-to-end Equivariant Robot Learning with RGB Cameras},
  author={Klee, David and Hu, Boce and Cole, Andrew and Tian, Heng and Wang, Dian and Platt, Robert and Walters, Robin},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```

## Acknowledgements
The codebase was built based on the [DiffPo](https://github.com/real-stanford/diffusion_policy) and [Equi. DiffPo](https://github.com/pointW/equidiff) repos.  The implementation for Geometric Transform Attention was based on the [GTA repo](https://github.com/autonomousvision/gta).
