""" Test File """


from generate_data import *
from RBM_purification import *

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

"""


a =  0
b = 0
c = 32
t = exp_unitary(2, 0.5j, 0.5j, 1 + 3/4j*np.pi)
test = exp_unitary(1, a, b, c)

#print(test@np.conj(test).T) #actually encodes a unitary_operation

a1, b1, c1, A = exp2spin_unitary(a, b, c)
U1 = exp_operator(a1, b1, c1, A)
print(U1)
#print(U1@np.conj(U1.T))
"""
a, b, c = 1, -1, 10
a1, b1, c1, d = exp2spin_unitary(a, b, c)
def exp_operator(alpha, beta, omega, A):
    U = np.zeros((2, 2), dtype=complex)
    U[0, 0] = 1
    U[0, 1] = np.exp(beta)
    U[1, 0] = np.exp(alpha)
    U[1, 1] = np.exp(alpha + beta + omega)
    U *= A
    return(U)

print(exp_operator(a1, b1, c1, d))
print(exp_unitary(1, 1, -1, 10))
print(a1, b1, c1)
