"""
This file includes all the functions necessary for data generation to train the
neural network.
"""

#imports
import numpy as np
import itertools

# Classes


class pure_state:
    """ Initialize a state vector for some pure state """

    # some typical pure states

    up = np.array([[1, 0]]).T
    down = np.array([[0, 1]]).T

    plus = 1/np.sqrt(2) * np.array([[1, 1]]).T
    minus = 1/np.sqrt(2) * np.array([[1, -1]]).T



    def __init__(self, product_state, single_particle_states, coeffs,
                 states_list):
        """Initialize the state vector
            product_state: Boolean
            single_particle_states: list of states to build product states
            coeffs: coefficients for superposition
            states_list: list of product states """

        if product_state == True:
            psi = single_particle_states[0]
            for state in single_particle_states[1:]:
                psi = np.kron(psi, state)

        else:
            psi = coeffs[0] * states_list[0]
            for (c, vec) in zip(coeffs[1:], states_list[1:]):
                psi += c * vec

        self.state_vector = psi


class density_matrix:
    """initialize a density matrix, retrieve information about it, generate
    samples from it"""

    def __init__(self, probabilities, states):
        """ initialize the density matrix from a mixture of states"""

        dim = np.shape(states[0])[0] #length of the state vector
        matrix_representation = np.zeros((dim, dim)) #build empty matrix

        for (prob, state) in zip(probabilities, states):
            matrix_representation += prob * state @ np.conjugate(state).T

        self.matrix_rep = matrix_representation


    def generate_samples(self, n_samples):
        """generate a set of samples from the density matrix to be used for RBM
        training"""


class random_density_matrix:
    "initialize a random density matrix"

    def __init__(self, n_spins):
        'n_spins: Integer that specifies the size of the density matrix'
        rand_mat = (np.random.rand(2**n_spins, 2**n_spins) +
                    1j * np.random.rand(2**n_spins, 2**n_spins))
        rho = np.dot(np.conj(rand_mat.T), rand_mat)

        initial_trace = np.trace(rho)
        for i in range(2**n_spins):
            rho[i, i] /= initial_trace

        self.n_spins = n_spins
        self.matrix = rho

    def generate_samples(self, n_samples):
        cumulative_sums = np.cumsum(np.diagonal(self.matrix))
        samples = []

        for n in range(n_samples):
            sample_prob = np.random.uniform()
            sample_index = np.min(np.where(cumulative_sums > sample_prob))
            samples.append(index2state(sample_index, self.n_spins))

        return(samples)

    def change_basis(self, U_global):
        """transform the density matrix to another basis, where the
            transformation is given by some unitary U"""
        self.matrix =  U_global @ self.matrix @ np.conj(U_global).T

    def generate_samples_all_bases(self, n_samples):

        n_bases = 3**self.n_spins # for pauli operators --> not general!

        samples = np.empty((n_bases, n_samples), dtype="U2") # rows: bases, columns: n_samples

        all_transformations = generate_basis_transformations(pauli_matrices,
                                                             self.n_spins)

        for (i, trafo) in zip(range(len(all_transformations)),
                              all_transformations):
            new_rho = random_density_matrix(self.n_spins)
            new_rho.matrix = np.copy(self.matrix) #have to be identical!
            new_rho.change_basis(trafo)
            samples[i, :] = new_rho.generate_samples(n_samples)
        return(samples)

#functions

def local_basis_transform(basis1, basis0):
    """transform basis_0 into basis_1. Each basis is given by a matrix with
    basis vectors as columns."""

    U_local = np.conj(basis1).T @ basis0
    return(U_local)

    """what bases? For each qubit sig_x sig_y sig_z"""

def total_basis_transform(local_transforms):
    'build a basis transformation from local transformations of bases'
    U = local_transforms[0]
    for U_b in local_transforms[1: ]:
        U = np.kron(U, U_b)

    return(U)

def index2state(index, n_qubits):
    'transforms an index into a n-qubit state'
    return(bin(index)[2:].zfill(n_qubits))
    #return( np.binary_repr(index, n_qubits))



pauli_x = np.array([ [0, 1],
                     [1, 0] ])

pauli_y = np.array([ [0, -1j],
                     [1j, 0] ])

pauli_z = np.array([ [1,0],
                     [0,-1] ])

pauli_matrices = [pauli_x, pauli_y, pauli_z]


def generate_basis_transformations(single_basis, n_qubits):
    'returns transformation matrices to all possible new measurement bases'

    #build all possible measurement bases from a single qubit measurement basis
    all_bases = list(itertools.product(single_basis, repeat=n_qubits))
    single_transforms = ([[[] for j in range(len(all_bases[0]))] for i in
                         range(len(all_bases))])
    total_transforms = []
    for i in range(len(all_bases)):
        for j in range(len(all_bases[i])):
            single_transforms[i][j] = local_basis_transform(all_bases[i][j],
                                                            pauli_z)

        #append the relevant basis transformation to transform the density
        #matrix from its original value to the one in the new basis

        total_transforms.append(total_basis_transform(single_transforms[i]))

    return(total_transforms)
