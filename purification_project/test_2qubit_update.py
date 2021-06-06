import numpy as np
from generate_data import *
from RBM_purification import *

np.set_printoptions(precision = 2, linewidth = 150) # for nice format

# 2-body unitary: matrix rep
U2 = two_body_entangling(np.pi/4)
#print(U2)
#U_tot = total_unitary([identity, U2, identity])  #U2 acts on 2nd and 3rd qubit
# 2-body unitary: RBM params
alpha_1 = 0
alpha_2 = 0
beta_1 = 0
beta_2 = 0
Gamma =  np.zeros((2, 2))
lam = np.pi/4
Lambda = 1j*lam*np.array([[0, 1], [1, 0]])
n = 2
Omega = 1j*(2 * n + 1) * np.pi / 4 * np.array([[1, 0], [0, 1]]) # actually (2 * n + 1)


#specify RBM size
n_visible_units = 2
n_hidden_units = 2
n_auxiliary_units = 2

#initialize Boltzmann Machine
boltz_machine = RBM(n_visible_units, n_hidden_units, n_auxiliary_units, 0)
boltz_machine.nodeType = "-11"
boltz_machine.calc_rho_NN()
boltz_machine.check_rho_valid()


#store and update density matrix
test_rho = random_density_matrix(n_visible_units)

test_rho.matrix = np.copy(boltz_machine.rho_encoded) #copy RBM matrix to new class
print("The density matrix before the unitary operation:")
print(test_rho.matrix)
a = np.copy(test_rho.matrix )
# trace check: print(np.trace(test_rho.matrix @ test_rho.matrix))
print("\n ")
print("Squared Trace before gate")
print(np.trace(test_rho.matrix @ test_rho.matrix))
print(" Trace before gate")
print(np.trace(test_rho.matrix))


test_rho.unitary_operation(U2)

print("The density matrix after the unitary operation:")
print(test_rho.matrix)
print("\n ")
print("Squared Trace after gate")
print(np.trace(test_rho.matrix @ test_rho.matrix))

#update RBM
boltz_machine.UBM_update_double(0, 1, alpha_1, alpha_2, beta_1, beta_2, Gamma, Lambda, Omega)
boltz_machine.UBM_rho()
density_matrix_UBM_prime = np.copy(boltz_machine.rho_encoded_UBM)
print("The density matrix after the unitary evolution, according to the UBM update rule: ")
print(density_matrix_UBM_prime)
#print("Fidelity with analytically calculated density matrix: ")
#print(fidelity(density_matrix_UBM_prime, test_rho.matrix))
print("Fidelity with initial RBM density matrix: ")
#print(fidelity(np.flipud(np.fliplr(density_matrix_UBM_prime)), test_rho.matrix))
print(fidelity(test_rho.matrix, density_matrix_UBM_prime))
