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
    # code may return an error if W is a real number and smaller than 1
    if type == "v":
        return(1/2 * (np.arccosh(1/2 * np.exp(W)) + np.arccosh(1/2 * np.exp(-W))))

    elif type == "a":
        return(1/2 * (np.arccosh(1/2 * np.exp(W)) - np.arccosh(1/2 * np.exp(-W))))



#specify RBM size
n_visible_units = 2
n_hidden_units = 2
n_auxiliary_units = 2

pres = 3
np.set_printoptions(precision = pres, linewidth = 150)


boltz_machine = RBM(n_visible_units, n_hidden_units, n_auxiliary_units, 1)
boltz_machine.nodeType = "-11"


n_hid_add = n_visible_units * n_auxiliary_units #new mediation units

test_machine = RBM(n_visible_units + n_auxiliary_units, n_hid_add + n_hidden_units, 0, 0)

test_machine.biases_v[:n_visible_units] = boltz_machine.biases_v
test_machine.biases_v[n_visible_units:] = boltz_machine.biases_a
test_machine.biases_h[:n_hid_add] = np.zeros(n_hid_add, dtype = complex)
test_machine.biases_h[n_hid_add:] = boltz_machine.biases_h


test_machine.weights_h = np.zeros((test_machine.n_h, test_machine.n_v), dtype=complex)
test_machine.weights_h[n_hid_add:, :n_visible_units] = boltz_machine.weights_h

# this just includes the old interactions
#print(test_machine.weights_h)

#now add the hidden-mediation interactions
for i in range(n_visible_units):
    for j in range(n_auxiliary_units):
        # look at each interaction pair (ij) -->

        #A in notebook sketch
        test_machine.weights_h[n_auxiliary_units*i + j, i] = weight_new("v", boltz_machine.weights_a[j, i])

        #B in notebook sketch
        test_machine.weights_h[n_auxiliary_units*i + j, n_visible_units + j] = weight_new("a", boltz_machine.weights_a[j, i])
#print(test_machine.weights_h)


# initialized two Boltzmann machines

# now look if they behave identical
test_machine.nodeType = "-11"
boltz_machine.nodeType = "-11"

v = np.array([1, 1])
a =  np.array([1, 1])
vprime = np.concatenate((v, a), axis=0)

psi_test = 0
for hconf in range(2**test_machine.n_h):
    hconf_spins = 2*sample2conf(index2state(hconf, test_machine.n_h))-1
    psi_test += np.exp(test_machine.RBM_energy(vprime, hconf_spins))

norm_test = 0
for vconf in range(2**test_machine.n_v):
    vconf_spins = 2*sample2conf(index2state(vconf, test_machine.n_v))-1
    cf = 0
    for hconf in range(2**test_machine.n_h):
        hconf_spins = 2*sample2conf(index2state(hconf, test_machine.n_h))-1
        cf += np.exp(test_machine.RBM_energy(vconf_spins, hconf_spins))
    norm_test += np.abs(cf)**2

print("new machine encodes: ")
encoded = psi_test/np.sqrt(norm_test)
print(encoded)

#print(test_machine.calc_psi_vversion(vprime))
psi_true = 0
for hconf in range(2**boltz_machine.n_h):
    hconf_spins = 2*sample2conf(index2state(hconf, boltz_machine.n_h))-1
    psi_true += np.exp(boltz_machine.RBM_energy_ahid(v, hconf_spins, a))

norm_true = 0
for aconf in range(2**boltz_machine.n_a):
    aconf_spins = 2*sample2conf(index2state(aconf, boltz_machine.n_a))-1
    for vconf in range(2**boltz_machine.n_v):
        vconf_spins = 2*sample2conf(index2state(vconf, boltz_machine.n_v))-1
        cf = 0
        for hconf in range(2**boltz_machine.n_h):
            hconf_spins = 2*sample2conf(index2state(hconf, boltz_machine.n_h))-1
            cf += np.exp(boltz_machine.RBM_energy_ahid(vconf_spins, hconf_spins, aconf_spins))
        norm_true += np.abs(cf)**2

print("true value: ")
true_encoded = psi_true/np.sqrt(norm_true)
print(true_encoded)

print("the difference is: ")
print(np.abs(true_encoded - encoded))
