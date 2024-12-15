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
    plt.scatter(qubits, energies, alpha=0.7, label='Data points')
    plt.plot(qubits, model.predict(qubits), color='red', label='Regression line')
    plt.title('Ground state energy as function of the number of electrones')
    plt.xlabel('Electrones')
    plt.ylabel('Ground State Energy [Ha]')
    plt.legend()
    plt.show()

    print(f'Slope: {slope}')
    print(f'Intercept: {intercept}')

    e_pred = model.predict(qubits)
    re_error = np.abs((e_pred - energies) / energies) * 100

    mean_error = np.mean(re_error)
    
    print("RE%:", mean_error)

    return slope, intercept, mean_error

def predict(file, new_qubits):
    data = pd.read_csv(file)
    energies = data['energy'].values
    qubits = data['electrones'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(qubits, energies)
    slope = model.coef_[0]
    intercept = model.intercept_
    new_energy = new_qubits*slope + intercept
    
    print("The energy of the molecule is:",new_energy)
    return new_energy
    
    