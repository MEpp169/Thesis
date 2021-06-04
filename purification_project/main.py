"""
Project: "Simulating the action of quantum gates on density matrices using RBM
purifications"

created: 28.5.2021
author: Moritz Epping
"""

# import other files
import numpy as np
from generate_data import *
from RBM_purification import *

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

np.set_printoptions(precision = 2) # for nice format

"""
# generate a random density matrix to be learnt
rho = random_density_matrix(n_spins)

print("The density matrix to be learnt is: \n")
print(rho.matrix)
print("\n")
print("It has trace " + str(np.around(np.real(np.trace(rho.matrix)))))
print("\n")
"""

#initialize random density matrix
rho = random_density_matrix(n_spins)
"""
print(rho.matrix)
local_unitaries = [pauli_x, identity]
rho.unitary_operation(total_unitary(local_unitaries))
print(rho.matrix)

#generate data: for each row (=measurement basis) a list of samples
rho_samples = rho.generate_samples_all_bases(n_samples)
print("Generated data: {} samples for each of the {} bases".format(
      n_samples, 3**n_spins))


#initialize an RBM
boltz_machine = RBM(n_visible_units, n_hidden_units, n_auxiliary_units)

#train the RBM
#boltz_machine.stochastic_gradient_descent(çtraining_steps, learning_rate,
                                          #n_bases, rho_samples, subset_size,
                                          #rho.matrix)

print(np.trace(boltz_machine.calc_rho_NN()))
print(boltz_machine.biases_a)
print(boltz_machine.calc_rho_NN())
#plot the results
"""
