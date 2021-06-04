import numpy as np
from generate_data import *
from RBM_purification import *

np.set_printoptions(precision = 2, linewidth = 90) # for nice format


# specify settings
n_spins = 2
n_samples = 1000
n_bases = 3**n_spins #Pauli-operator specific!

n_visible_units = 2
n_hidden_units = 4
n_auxiliary_units = 4

training_steps = 10
learning_rate = 1.0e-2
subset_size = 20

boltz_machine = RBM(n_visible_units, n_hidden_units, n_auxiliary_units, 1)
boltz_machine.calc_rho_NN()
boltz_machine.check_rho_valid()

print(np.trace(boltz_machine.rho_encoded @ boltz_machine.rho_encoded))
# apply simple unitary

#alpha = 0
#beta = 0
#omega = 30

alpha = 0.7
beta = 0.2
omega = 0.9

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
"""



U_test = exp_unitary(1, alpha, beta, omega)
local_unitaries = [U_test, identity]
alpha_RBM, beta_RBM, omega_RBM, A_RBM = exp2spin_unitary(alpha, beta, omega)
# the unitary is now correctly encoded in two reps: RBM exponential with complex
# parameters, simple exponential with 3 real parameters

"initialize random density matrix"
boltz_machine = RBM(n_visible_units, n_hidden_units, n_auxiliary_units, 1)
boltz_machine.calc_rho_NN() #calculate the density matrix
boltz_machine.check_rho_valid()
print("trace of rho^2")
print(np.trace(boltz_machine.rho_encoded @ boltz_machine.rho_encoded))


"update the density matrix"
test_rho = random_density_matrix(n_spins)
test_rho.matrix = np.copy(boltz_machine.rho_encoded) #copy RBM matrix to new class
print("The density matrix before the unitary operation:")
print(test_rho.matrix)
print("\n ")
test_rho.unitary_operation(total_unitary(local_unitaries))
print("The density matrix after the unitary operation:")
print(test_rho.matrix)

print("Fidelity of density matrices before and after unitary:")
print(fidelity(boltz_machine.rho_encoded, test_rho.matrix))

"perform the upates on RBM"
boltz_machine.UBM_update_single(alpha_RBM, beta_RBM, omega_RBM, A_RBM, 0)
boltz_machine.UBM_rho()
print("The density matrix encoded by the UBM after the unitary operation:")
print(boltz_machine.rho_encoded_UBM)
print("Fidelity of density matrices exact vs UBM")
print(fidelity(boltz_machine.rho_encoded_UBM, test_rho.matrix))


"""
local_unitaries = [U_test]

print(U_test@np.conj(U_test.T))
test_rho = random_density_matrix(n_spins)
test_rho.matrix = np.copy(boltz_machine.rho_encoded)

# define new RBM to check if density matrix is valid
test_RBM = RBM(n_visible_units, n_hidden_units, n_auxiliary_units)
test_RBM.rho_encoded = test_rho.matrix
#test_RBM.check_rho_valid()
print("The density matrix after the unitary operation:")
print(test_rho.matrix)

# update the RBM
alpha_RBM, beta_RBM, omega_RBM, A_RBM = exp2spin_unitary(alpha, beta, omega)
print(alpha_RBM, beta_RBM, omega_RBM)
print(simple_exp_unitary(1, alpha_RBM, beta_RBM, omega_RBM, A_RBM ))


boltz_machine.UBM_update_single(alpha_RBM, beta_RBM, omega_RBM, 0)
boltz_machine.UBM_rho()
print(boltz_machine.rho_encoded_UBM)
#test_RBM.rho_encoded = boltz_machine.rho_encoded_UBM
#test_RBM.check_rho_valid()
"""
