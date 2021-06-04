"File to test how the a-weight-matrix influences purity"

import numpy as np
import matplotlib.pyplot as plt
from generate_data import *
from RBM_purification import *

n_visible_units = 2
n_hidden_units = 4
n_auxiliary_units = 4
n_average = 10
a_coeffs = [0, 0.1, 0.2, 0.5, 1, 2, 3, 4]

purity_list = []

for coeff in a_coeffs:
    this_av = 0
    for i in range(n_average):
        boltz_machine = RBM(n_visible_units, n_hidden_units, n_auxiliary_units, coeff)
        boltz_machine.calc_rho_NN()
        this_av += np.trace(boltz_machine.rho_encoded @ boltz_machine.rho_encoded)
    purity = this_av/n_average
    purity_list.append(this_av/n_average)
    print(purity)
plt.scatter(a_coeffs, purity_list)
plt.xlabel("Magnitude Scale of Weight Matrix a")
plt.ylabel("Purity of random density matrix")
plt.title("Parameter Scaling and Purity of Purification RBMs")
plt.show()
plt.savefig("params_purity.pdf")
