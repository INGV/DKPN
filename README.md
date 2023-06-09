# Domain Knowledge PhaseNet picker

Deep-Learning picker based on the U-net architecture of PhaseNet. 
Final version to be fully embedded in SeisBench.

Following this issue `https://github.com/seisbench/seisbench/issues/151#issuecomment-1356723017`,
we may now switch to the original TensorFlow implemetnation of PN

VERSION: _0.3.1_

DATE: _06.2023_

This version is the first fully fledged, stable release to be used
in combination with `SeisBench (v0.3.0)`. When you install the package (with pip)
it will automatically load all the weights in the seisbench cache/model dir.


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

## Developing

From v0.0.2 the code is open to contributions, just clone the `DEVELOP` branch
on your local machine and continue from there:

```bash
$ git clone --branch DEVELOP --single-branch git@gitlab.rm.ingv.it:some/dkpn.git ./LOCALFOLDERNAME
```
