""" Test File """


from RBM_purification import *


#generate 10000 samples for a density matrix of a 4-qubit system
dmat = random_density_matrix(4).generate_samples(10000)
print(dmat[0]+3)
