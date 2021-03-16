# neurosynth-mfc
This repository contains code, data, and results for De La Vega, Chang, Banich, Wager & Yarkoni. Journal of Neuroscience (2016).

Final parcellation images are available under images/

Follow along the notebooks to recreate analyses, results and visualizations from the article.
These notebooks are intended to allow researchers to easily perform similar analyses on other brain areas of interest.

Interact with these notebooks live, with all dependencies preinstalled here:

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/adelavega/neurosynth-mfc)

### Requirements
- Scientific Python environment. A recommended way to install is to use Anaconda: https://www.continuum.io/
- Install library requirements using pip:

`pip install -r requirements.txt`

Remember to unzip pre-generated Neurosynth dataset prior to analysis

## Docker
To facilitate environment set up, you can use Docker.
Once Docker is installed, you can clone this repo, build the Docker image,
and run the Jupyter notebook

```
git clone https://github.com/adelavega/neurosynth-mfc.git
cd neurosynth-mfc
docker build . -t neurosynth-mfc
docker run --rm -v "$PWD":/usr/src/work -p 8888:8888 neurosynth-mfc
```
This will set up a Jupyter notebook server, with the current directory
mounted so any changes will be saved. Follow the instructions in the terminal
to launch the notebook.
