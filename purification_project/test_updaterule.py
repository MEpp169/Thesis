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

boltz_machine = RBM(n_visible_units, n_hidden_units, n_auxiliary_units)
boltz_machine.calc_rho_NN()
boltz_machine.check_rho_valid()

# apply simple unitary

alpha = 1
beta = 0.5
omega = 0.75


U_test = exp_unitary(n_spins, alpha, beta, omega)
local_unitaries = [identity, U_test]

test_rho = random_density_matrix(n_spins)
test_rho.matrix = np.copy(boltz_machine.rho_encoded)
print("The density matrix before the unitary operation:")
print(test_rho.matrix)
print("\n ")
test_rho.unitary_operation(total_unitary(local_unitaries))
# define new RBM to check if density matrix is valid
test_RBM = RBM(n_visible_units, n_hidden_units, n_auxiliary_units)
test_RBM.rho_encoded = test_rho.matrix
test_RBM.check_rho_valid()
print("The density matrix after the unitary operation:")
print(test_rho.matrix)

# update the RBM
alpha_RBM, beta_RBM, omega_RBM = exp2spin_unitary(alpha, beta, omega)
print(alpha_RBM, beta_RBM, omega_RBM)
boltz_machine.UBM_update_single(alpha_RBM, beta_RBM, omega_RBM, 1)
boltz_machine.UBM_rho()
print(boltz_machine.rho_encoded_UBM)
test_RBM.rho_encoded = boltz_machine.rho_encoded_UBM
test_RBM.check_rho_valid()
