# Analysis of the Ganglion Cells in the Retina

---

**Organization:** [Fundación Ciencia & Vida](https://cienciavida.org/)  
**Supervisor:** Cesar Ravello  
**Experiment:** Universidad de Valparaíso  
**Code and computacional analysis:** [Javier Cruz](https://github.com/sisyphvs) & [David Felipe](https://github.com/davidnfu0)  
**Date:** January 2024  

___
## Table of contents
- [Libraries](#libraries)
- [Description](#description)
- [Experiment](#experiment)
- [Requirements](#requirements)
- [Cloning the repository](#cloning-the-repository)
- [Running the notebooks](#running-the-notebooks)
- [Contributing](#contributing)
___

## Libraries
- [h5py](https://www.h5py.org/)
- [ipykernel](https://ipykernel.readthedocs.io/en/stable/)
- [lmfit](https://lmfit.github.io/lmfit-py/)
- [matplotlib](https://matplotlib.org/)
- [neo](https://neo.readthedocs.io/en/latest/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [pyret](https://pyret.readthedocs.io/en/master/index.html)
- [PyYAML](https://pyyaml.org/)
- [requests](https://pypi.org/project/requests/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [scipy](https://www.scipy.org/)
- [seaborn](https://seaborn.pydata.org/)
- [umap](https://umap-learn.readthedocs.io/en/latest/)


## Description

This repository contains a computational analysis of an experiment investigating ganglion cells in the retinas of mice. The analysis includes mathematical modeling of temporal contrast curves and clustering to identify different types of cells. 

The analysis attempts to replicate and expand upon the findings of two different scientific papers:
 
1. Chichilnisky EJ, Kalmar RS. Functional asymmetries in ON and OFF ganglion cells of primate retina. J Neurosci. 2002 Apr 1;22(7):2737-47. doi: 10.1523/JNEUROSCI.22-07-02737.2002. Erratum in: J Neurosci 2002 Oct 1;22(19):1a. PMID: 11923439; PMCID: PMC6758315.
2. Baden, T., Berens, P., Franke, K. et al. The functional diversity of retinal ganglion cells in the mouse. Nature 529, 345–350 (2016). https://doi.org/10.1038/nature16468

## Experiment

The experiment was carried out by the Universidad de Valparaíso, so we do not have the right to upload the results.

## Requirements

- `Python 3.10 or higher`
- `pip`
- `git`

## Cloning the repository

1. Open a terminal and go to the directory where you want to clone the repository. 

2. Clone the repository using the following command:
```bash
git clone git@github.com:davidnfu0/Retinal-Ganglion-Cell-Analysis.git
```

3. Go to the directory of the repository:
```bash
cd Retinal-Ganglion-Cell-Analysis
```

4. Create a virtual environment:
- **Using python env:**

    Creating a virtual environment:

    ```bash
    python3 -m venv ganglion-venv
    ```
    Activate the virtual environment:

    - **Linux/Mac OS:**

        ```bash
        source ganglion-venv/bin/activate
        ```
    - **Windows:**

        ```bash
        .\ganglion-venv\Scripts\activate
        ```
    Install the requirements:

    ```bash
    pip install -r requirements.txt
    ```
    If you want to deactivate the virtual environment:

    ```bash
    deactivate
    ```
- **Using conda env:**

    Create a virtual environment:

    ```bash
    conda create --name ganglion-venv python=3.10
    ```
    Activate the virtual environment:

    ```bash
    conda activate ganglion-venv
    ```
    Install the requirements:

    ```bash
    pip install -r requirements.txt
    ```
    If you want to deactivate the virtual environment:

    ```bash
    conda deactivate
    ```

## Running the notebooks

1. Activate the virtual environment.
2. Run the notebooks in numbered order.
Some notebooks may take a long time to run.

## Contributing

If you want to contribute to the project by fixing the code or contributing new code, feel free to do so and create a pull request explaining the changes. Remember to update the authorship and modification date of the documents.

If you find an error in any text or in any of the codes, please write an issue. Thank you!
