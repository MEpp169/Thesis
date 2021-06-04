"File to test how the a-weight-matrix influences infidelity after unitary gates"

import numpy as np
import matplotlib.pyplot as plt
from generate_data import *
from RBM_purification import *

n_spins = 2

n_visible_units = 2
n_hidden_units = 4
n_auxiliary_units = 4
n_average = 10
a_coeffs = [0, 0.1, 0.2, 0.5, 1, 2, 3, 4]

fidelity_list = []
purity_list = []


#unitary parameters
alpha = 0.5
beta = 0.4
omega = 0.6

#unitaries
U_test = exp_unitary(1, alpha, beta, omega)
local_unitaries = [U_test, identity]
alpha_RBM, beta_RBM, omega_RBM, A_RBM = exp2spin_unitary(alpha, beta, omega)

for coeff in a_coeffs:
    #initialize Boltzmann Machine with specific "impurity coefficient"
    boltz_machine = RBM(n_visible_units, n_hidden_units, n_auxiliary_units, coeff)
    boltz_machine.calc_rho_NN()
    purity = np.trace(boltz_machine.rho_encoded @ boltz_machine.rho_encoded)
    purity_list.append(purity)


    #copy state of Boltzmann Machine to other matrix
    test_rho = random_density_matrix(n_spins)
    test_rho.matrix = np.copy(boltz_machine.rho_encoded)
    #make unitary evolution on that matrix exact
    test_rho.unitary_operation(total_unitary(local_unitaries))


    #update Boltzmann Machine
    boltz_machine.UBM_update_single(alpha_RBM, beta_RBM, omega_RBM, A_RBM, 0)
    boltz_machine.UBM_rho() #calculate the newly encoded density matrix

    #evaluate fidelity between matrices
    fid = fidelity(boltz_machine.rho_encoded_UBM, test_rho.matrix)
    fidelity_list.append(fid)


print(purity_list)
print(fidelity_list)


plt.scatter(a_coeffs, fidelity_list)
plt.xlabel("Magnitude Scale of Weight Matrix a")
plt.ylabel("Fidelity of UBM Evolution to Exact ")
plt.title("UBM Performance depending on Purity of Rho")
plt.show()
plt.savefig("fidelities_purity.pdf")
