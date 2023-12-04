# Domain Knowledge PhaseNet picker

Deep-Learning picker based on the U-net architecture of PhaseNet. 
Final version to be fully embedded in SeisBench.

VERSION: _0.4.12_

DATE: _11.2023_

This version is the first fully fledged, stable release to be used
in combination with `SeisBench (v0.4.0)`.
Soon it will be loaded to PyPI as well (and an easy-install through `pip` provided as well)


## Installation

In next releases, the framework will be available through PyPI.
At the moment, though, the installation is only possible from this repository.
Additionally, it is recommended to create a separate virtual environment with `conda` or `venv`
(e.g., `$ conda create -n dkpn python=3.9`). Though, this is taken care already by an
environment-file.

You can follow these simple commands.

```bash
$ git clone https://gitlab.rm.ingv.it/some/dkpn  # use the master branch for stable releases
$ cd dkpn
$ conda env create -f dkpn_env.yml
$ conda activate dkpn
```

If problem installing (or unwanted package listed in the env), you could install the bare minimu as:

```bash
conda create -n dkpn python=3.9.12
conda activate dkpn
pip install numpy==1.24.3
pip install scipy==1.8.0
pip install torch==1.11.0 torchvision torchaudio
pip install seisbench==0.4.0
pip install pandas==1.3.2
pip install matplotlib==3.5.1
pip install seaborn
pip install obspy==1.4.0
pip install jupyterlab==3.3.3
```
