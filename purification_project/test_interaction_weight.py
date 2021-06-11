import numpy as np
from generate_data import *
from RBM_purification import *


# code to test whether my interaction weight updates were correct

"""
idea:

- initialize some state (visible-hidden auxiliary; hiddens integrated out)
- calculate the visible-version of this state (integrate out hiddens as well)

- compare if same state is encoded
    - if yes: update worked!
    - if not: missed something!
"""

def weight_new(type, W):
    "returns the new weight (either v or a) given the old weight of a direct interaction"

    if type == "v":
        return(1/2 * (np.arccosh(1/2 * np.exp(W)) + np.arccosh(1/2 * np.exp(-W))))

    elif type == "a":
        return(1/2 * (np.arccosh(1/2 * np.exp(W)) - np.arccosh(1/2 * np.exp(-W))))



#specify RBM size
n_visible_units = 2
n_hidden_units = 2
n_auxiliary_units = 2


boltz_machine = RBM(n_visible_units, n_hidden_units, n_auxiliary_units, 1)
boltz_machine.nodeType = "-11"


n_hid_add = n_visible_units * n_auxiliary_units #new mediation units

test_machine = RBM(n_visible_units + n_auxiliary_units, n_hidden_units + n_hid_add, 0, 0)

test_machine.biases_v[:n_visible_units] = boltz_machine.biases_v
test_machine.biases_v[n_visible_units:] = boltz_machine.biases_a
test_machine.biases_h[:n_hid_add] = np.zeros(n_hid_add, dtype = complex)
test_machine.biases_h[n_hid_add:] = boltz_machine.biases_h

test_machine.weights_h = np.zeros((test_machine.n_h, test_machine.n_v), dtype=complex)
test_machine.weights_h[n_hid_add:, :n_visible_units] = boltz_machine.weights_h

for i in range(n_visible_units):
    for j in range(n_auxiliary_units):
        # look at each interaction pair v-a
        test_machine.weights_h[n_auxiliary_units*i + j, i] = weight_new("v", boltz_machine.weights_a[j, i])
        test_machine.weights_h[n_auxiliary_units*i + j, n_visible_units + j] = weight_new("a", boltz_machine.weights_a[j, i])


# initialized two Boltzmann machines

# now look if they behave identical
test_machine.nodeType = "-11"
boltz_machine.nodeType = "-11"

v = np.array([[1, 1]]).T
a =  np.array([[1, 1]]).T
vprime = np.concatenate((v, a), axis=0)

print(test_machine.calc_psi_vversion(vprime))

print(boltz_machine.calc_psi(boltz_machine.biases_v, boltz_machine.biases_h, boltz_machine.biases_a, boltz_machine.weights_h, boltz_machine.weights_a, v, a))
