import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def fit_linear_model(file):
    data = pd.read_csv(file)
    energies = data['energy'].values
    qubits = data['electrones'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(qubits, energies)
    slope = model.coef_[0]
    intercept = model.intercept_

    plt.figure(figsize=(10, 6))
    plt.scatter(qubits, energies, alpha=0.7, label='Data points', color='blue')
    plt.plot(qubits, model.predict(qubits), color='red', label='Regression line', linewidth=2)
    plt.title('Ground State Energy as a Function of the Number of Electrons', fontsize=14)
    plt.xlabel('Electrons', fontsize=12)
    plt.ylabel('Ground State Energy [Ha]', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print(f'Slope: {slope}')
    print(f'Intercept: {intercept}')

    e_pred = model.predict(qubits)
    re_error = np.abs((e_pred - energies) / energies) * 100
    mean_error = np.mean(re_error)

    print("Mean RE%:", mean_error)

    return slope, intercept, mean_error

def predict(file, new_qubits):
    data = pd.read_csv(file)
    energies = data['energy'].values
    qubits = data['electrones'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(qubits, energies)
    slope = model.coef_[0]
    intercept = model.intercept_
    new_energy = slope * new_qubits + intercept

    print(f"The predicted energy of the molecule with {new_qubits} electrons is: {new_energy} Ha")
    return new_energy
