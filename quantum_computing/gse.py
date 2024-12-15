
def compute_ground_state_energy(symbols,coordinates,spin,charge,method):
    
    if method == "HF":
            import numpy as np
            from pyscf import gto, scf
            mol = gto.M(
                atom=[(symbols[i], coordinates[i]) for i in range(len(symbols))],
                basis='sto-3g',
                spin=spin,
                charge=charge
            )
            mf = scf.RHF(mol)
            energy = mf.kernel()
            print(f"Ground state energy: {energy} Hartrees")
   
    if method == "VQE":
            import numpy as np
            import pennylane as qml
            from pennylane import qchem

            H, qubits = qchem.molecular_hamiltonian(
                    symbols,
                    coordinates,
                    charge=charge,
                    mult=1,
                    basis="sto-3g",
                    active_electrons=4,
                    active_orbitals=4
                )
            dev = qml.device("default.qubit", wires=qubits)
            from pennylane import numpy as np
            theta = np.array(0.0, requires_grad = True)
            electrons = 2
            hf = qml.qchem.hf_state(electrons,qubits)
            def circuit(param,wires):
                qml.BasisState(hf, wires = wires)
                qml.DoubleExcitation(param,wires = [0,1,2,3])
            opt = qml.GradientDescentOptimizer(stepsize=0.2)
            @qml.qnode(dev, interface = "autograd")
            def cost_fn(param):
                circuit(param,wires = range(qubits))
                return qml.expval(H)
            energy = [cost_fn(theta)]
            angle = [theta]
            max_iterations = 10
            conv_tol = 1e-06
            for n in range(max_iterations):
                theta, prev_energy = opt.step_and_cost(cost_fn, theta)
                energy.append(cost_fn(theta))
                angle.append(theta)
                conv = np.abs(energy[-1] - prev_energy)
                if n % 2 == 0:
                    print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

                #if conv <= conv_tol:
                # break
            energy = energy[-1]
            print(f"Ground state energy: {energy} Hartrees")
                
    if method == "QPE":
            print("This code will be released shortly.")
    
    
    return energy
        