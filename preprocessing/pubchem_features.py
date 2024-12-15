import pandas as pd
import re

def clean_pubchem(file):
    data = pd.read_csv(file)
    data = data[["id","mf","mw","polararea","complexity","xlogp","heavycnt","hbonddonor","hbondacc","rotbonds","covalentunitcnt","isotopeatomcnt","totalatomstereocnt","definedatomstereocnt"]]
    data['electrones'] = data['mf'].apply(num_qubits)
    data['S'] = data['mf'].apply(one_hot_encoding, args=('S',))
    data.to_csv(file)
    return data

def num_qubits(formula):
    num_qubits = 0
    for atom, count in re.findall(r'([A-Z][a-z]?)([0-9]*)', formula):
        if atom == 'C':
            multiplier = 6
        elif atom == 'O':
            multiplier = 8
        elif atom == 'H':
            multiplier = 1
        elif atom == 'N':
            multiplier = 7  
        elif atom == 'S':
            multiplier = 16
    count = int(count) if count else 1
    num_qubits += count * multiplier
    return num_qubits

def one_hot_encoding(molecula, elemento):
    return molecula.count(elemento)