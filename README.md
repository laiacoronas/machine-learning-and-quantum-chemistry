# ML-vs-quantum-algorithms
This work explores the integration of machine learning (ML) techniques to address limitations in quantum algorithms such as Variational Quantum Eigensolver (VQE), Hartree-Fock (HF), and Quantum Phase Estimation (QPE). We focus on the development of datasets tailored to enhance quantum simulations and present a synergistic framework that leverages ML to improve accuracy, scalability, and computational efficiency. This repository aims to make the results obtained in the paper "" easy to reproduce and transparent.

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

- baseline: baseline model used to compare with proposed method
- data: original data used to train/test models
- model: main entrypoint of the proposed model
- notebook: step by step guidance of how to build this model

### Example usage

Look at the notebook ```example_usage.ipynb``` for more details. Please note that there are two different paths to follow: the ML approach and the quantum approach. Depending on your preferences you may want to compute the energies using one approach or the other.

## Acknowledgements
I want to express my deepest gratitute to Parfait Atchade, who helped me every step of the way during the development of this project.


