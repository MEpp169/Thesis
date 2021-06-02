""" Test File """


from generate_data import *

"""
#generate 10000 samples for a density matrix of a 4-qubit system
dmat = random_density_matrix(2)

dmat.matrix = np.array([[1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1]])
#print(dmat.generate_samples(20))

tot_trafo = total_basis_transform([local_basis_transform(pauli_matrices[0],
            pauli_matrices[2]),local_basis_transform(pauli_matrices[0], pauli_matrices[2])])

print(tot_trafo)
dmat.change_basis(tot_trafo)
print(dmat.matrix)
"""

print(np.log(-1+0j))
