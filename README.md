# Machine Learning and Quantum Chemistry
This work explores the integration of machine learning (ML) techniques to address limitations in quantum algorithms such as Variational Quantum Eigensolver (VQE), Hartree-Fock (HF), and Quantum Phase Estimation (QPE). We focus on the development of datasets tailored to enhance quantum simulations and present a synergistic framework that leverages ML to improve accuracy, scalability, and computational efficiency. This repository aims to make the results obtained in the paper "Leveraging Machine Learning to Overcome Limitations in Quantum Algorithms" easy to reproduce and transparent.

Note that this code is intend to reproduce the results obtained in the paper but it can be also used to predict energies for your own molecules.

## Framework

![image](https://github.com/user-attachments/assets/84b7c200-1b9d-46c1-b234-61729f82eb6a)

## Setup

### Install Pyscf

To run some of the quantum simulations you will need to install the Pyscf library (https://pyscf.org/). 
Please take into account that PySCF is not supported natively on Windows. You must use the Windows Subsystem for Linux.
Install it using the following command:

```python
pip install --prefer-binary pyscf
```

### Prepare the environment

All the other libraries that we use can be normally installed in your enviroment using **pip**. For convinience, we recommend running the following commands before starting:
```python
git clone https://github.com/laiacoronas/ML-vs-quantum-algorithms.git
cd ML-vs-quantum-algorithms
pip install -r requirements.txt
```

### Folder structure
The contents of this repository are organized as follows:

- data: original data used to train/test models
- preprocessing: some codes to preprocess data and atomic positions from json files or csv files downloaded from pubchem
- ml: machine learning models trained and tested to predict ground state energies
- quantum: quantum models used to obtained the ground state energies of the molecules
- results: final plots obtained by running the codes on the data folder

### Example usage

Look at the notebook ```example_usage.ipynb``` for more details. Please note that there are two different paths to follow: the ML approach and the quantum approach. Depending on your preferences you may want to compute the energies using one approach or the other.
There is also the ```plots.ipynb``` file where you can input your data or results and obtain the same plots as we did.

## Acknowledgements
I want to express my deepest gratitute to Parfait Atchade, who helped me every step of the way during the development of this project.

## Citation
The work presented in this repository is published in the following paper:

```python
Laia Coronas Sala, Parfait Atchade-Adelemou. Leveraging Machine Learning to Overcome Limitations in Quantum Algorithms. arXiv:2412.11405 [physics.chem-ph], 16 Dec 2024. Disponible en: https://doi.org/10.48550/arXiv.2412.11405
```


