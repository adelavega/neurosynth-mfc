# neurosynth-mfc
This repository contains code, data, and results for De La Vega, Chang, Banich, Wager & Yarkoni. Journal of Neuroscience (2016). 

Final parcellation images are available under images/

Follow along the [Clustering](Clustering.ipynb), [Coactivation](Coactivation.ipynb), and [Functional preference profiles](Functional preference profiles.ipynb) notebooks to recreate analyses, results and visualizations from the article. These notebooks are intended to allow researchers to easily perform similar analyses on other brain areas of interest.

### Requirements
Python 2.7.x

For analysis:
- Neurosynth tools (github.com/neurosynth/neurosynth)

    Note: PyPI is acting strange so install directly from github: `pip install git+https://github.com/neurosynth/neurosynth.git`
- Scipy/Numpy (Easiest way is using miniconda distribution)
- Scikit-learn
- joblib
- nibabel 1.x

For visualization:
- Pandas
- nilearn
- seaborn

Unzip pre-generated Neurosynth dataset prior to analysis


![alt text](/preprint/Figure 2 - Labeled_clusters.png)


