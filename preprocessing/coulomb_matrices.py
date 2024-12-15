import json
import numpy as np

def read_json_and_write_csv(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    compounds = data["PC_Compounds"]

    for compound in compounds:
        atoms = compound["coords"][0]
        x = atoms["conformers"][0]["x"]
        y = atoms["conformers"][0]["y"]
        z = atoms["conformers"][0]["z"]

        coordinates = np.array([x, y, z]).T

        num_atoms = len(coordinates)
        coulomb_matrix = np.zeros((num_atoms, num_atoms))

        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    distance = np.linalg.norm(coordinates[i] - coordinates[j])
                    coulomb_matrix[i][j] = 1.0 / distance

        # Flatten upper triangular part of the symmetric matrix
        upper_triangular = coulomb_matrix[np.triu_indices(num_atoms, k=1)]
        np.savetxt('matrix.csv', upper_triangular, delimiter=",", fmt="%s")
        return upper_triangular