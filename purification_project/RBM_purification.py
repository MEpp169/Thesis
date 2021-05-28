"""
subfile: Finding an RBM representation of a density matrix using machine
learning techniques



procedure: "Find optimal network parameters for an RBM to represent some density
matrix"
goal: "Minimize Kullback-Leibler divergence between RBM density matrix and true
density matrix"

    1. input: Data sets D_b that contain || D_b || snapshots sigma^b in various
    bases b.
    2. Update network parameters using Stochastic Gradient Descent (repeat the
    following procedure until satisfactory fidelity is achieved)
        2.1 Calculate the negative log-likelihood averaged over the data for a
        random subset of training samples
        2.2 Differentiate neg. log-likelihood w.r.t network parameters
        2.3 update network parameters
"""

# imports
import numpy as np


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
            sample_index = np.max(np.concatenate((np.array([0]),
                                  np.where(cumulative_sums < sample_prob)[0])))
            samples.append(index2state(sample_index, self.n_spins))

        return(samples)



def index2state(index, n_qubits):
    'transforms an index into a n-qubit state'

    return( np.binary_repr(index, n_qubits))
