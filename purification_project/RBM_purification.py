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
from scipy.linalg import sqrtm

'Classes'


class RBM():
    """Restricted Boltzmann Machine class"""

    def __init__(self, n_v, n_h, n_a):
        """ Initialize Restricted Boltzmann Machine graph structure:

        Geometry
        n_v: nr of visible nodes
        n_h: nr of hidden nodes
        n_a: nr of auxiliary (hidden) nodes
        """

        'Initialize size of RBM'
        self.n_v = n_v
        self.n_h = n_h
        self.n_a = n_a

        'Initialize parameters of RBM'
        self.biases_v = np.random.rand(self.n_v) + 1j * np.random.rand(self.n_v)
        self.biases_h = np.random.rand(self.n_h) + 1j * np.random.rand(self.n_h)
        self.biases_a = np.random.rand(self.n_a) + 1j * np.random.rand(self.n_a)

        self.weights_h = (np.random.rand(self.n_h, self.n_v) +
                            1j * np.random.rand(self.n_h, self.n_v))
        self.weights_a = (np.random.rand(self.n_a, self.n_v) +
                            1j * np.random.rand(self.n_a, self.n_v))


    def derivative_bias_v(self, v1, v2, part):
        """returns the derivative of log(rho_unnormalized) w.r.t. the visible
        biases. part = "r" for derivative of real part, part = "i" for
        derivative of imaginary part"""

        d_bias_v = np.zeros(self.n_v)

        if part == "r":
            for i in range(self.n_v):
                d_bias_v[i] = v1[i] + v2[i]
        elif part == "i":
            for i in range(self.n_v):
                d_bias_v[i] = 1.0j*(v1[i] - v2[i])

        return(d_bias_v)


    def derivative_bias_h(self, v1, v2, bias_h, weight_h, part):
        """returns the derivative of log(rho_unnormalized) w.r.t. the hidden
        biases h. part = "r" for derivative of real part, part = "i" for
        derivative of imaginary part"""

        d_bias_h = np.zeros(self.n_h)

        if part == "r":
            for i in range(self.n_h):
                up1 = 1/(np.exp(-(np.dot(weight_h[i, :], v1) + bias_h[i])) + 1)
                up2 = 1/(np.exp(-(np.dot(np.conj(weight_h[i, :]), v2) +
                                np.conj(bias_h[i]))) + 1)
                d_bias_h[i] = up1 + up2

        elif part == "i":
            for i in range(self.n_h):
                up1 = 1/(np.exp(-(np.dot(weight_h[i, :], v1) + bias_h[i])) + 1)
                up2 = 1/(np.exp(-(np.dot(np.conj(weight_h[i, :]), v2) +
                                np.conj(bias_h[i]))) + 1)
                d_bias_h[i] = 1.0j*(up1 - up2)

        return(d_bias_h)


    def derivative_bias_a(self, v1, v2, bias_a, weight_a, part):
        """returns the derivative of log(rho_unnormalized) w.r.t. the auxiliary
        biases a. part = "r" for derivative of real part, part = "i" for
        derivative of imaginary part"""

        d_bias_a = np.zeros(self.n_a)

        if part == "r": #only real part changes
            for i in range(self.n_a):
                d_bias_a[i] = 2/(np.exp(-(np.dot(U[i, :], v1) + np.dot(np.conj(
                              U[i,:]), v2) + bias_a[i] + np.conj(bias_a[i])))+1)
        return(d_bias_a)


    def derivative_weight_h(self, v1, v2, bias_h, weight_h, part):
        """returns the derivative matrix for log(rho_unnormalized) w.r.t. the
        weights corresponding to the hidden units h. 'part' specifies whether
        the derivative should be calculated w.r.t. the real ("r") or imaginary
        ("i") part of the weight"""

        d_weight_h = np.zeros((self.n_h, self.n_v))

        if part =="r":
            for i in range(self.n_h):
                for j in range(self.n_v):
                    up1 = v1[j] / (np.exp(np.dot(weight_h[i, :], v1) +
                                   bias_h[i]) + 1)
                    up2 = v2[j] / (np.exp(np.dot(np.conj(weight_h[i, :]), v2) +
                                   np.conj(bias_h[i])) + 1)
                    d_weight_h[i, j] = up1 + up2


        else:
            for i in range(self.n_h):
                for j in range(self.n_v):
                    up1 = v1[j] / (np.exp(np.dot(weight_h[i, :], v1) +
                                   bias_h[i]) + 1)
                    up2 = v2[j] / (np.exp(np.dot(np.conj(weight_h[i, :]), v2) +
                                   np.conj(bias_h[i])) + 1)
                    d_weight_h[i, j] = 1.0j*(up1 - up2)

        return(d_weight_w)


    def derivative_weight_a(self, v1, v2, bias_a, weight_a, part):
        """returns the derivative matrix for log(rho_unnormalized) w.r.t. the
        weights corresponding to the auxiliary units a. 'part' specifies whether
        the derivative should be calculated w.r.t. the real ("r") or imaginary
        ("i") part of the weight"""

        d_weight_a = np.zeros((self.n_a, self.n_v))

        if part == "r":
            for i in range(self.n_a):
                for j in range(self.n_v):
                    d_weight_a[i, j] = (v1[j] + v2[j]) / ( np.exp(np.dot(
                                        weight_a[i, :], v1) + np.dot(np.conj(
                                        weight_a[i, :]), v2) + bias_a[i] +
                                        np.conj(bias_a[i])))


        else:
            for i in range(self.n_a):
                for j in range(self.n_v):
                    d_weight_a[i, j] = 1.0j*(v1[j] - v2[j]) / ( np.exp(np.dot(
                                        weight_a[i, :], v1) + np.dot(np.conj(
                                        weight_a[i, :]), v2) + bias_a[i] +
                                        np.conj(bias_a[i])))

        return(d_weight_a)

    def derivative_sample_mean(self, samples, network_params, deriv_params):
        """Calculates the mean of the gradient over some set of samples:
        evaluate the gradient at each sample and compute mean
        doesn't matter if MCMC samples or true samples. Returns derivative
        w.r.t complex and real part seperately"""

        bias_v = network_params[0]
        bias_h = network_params[1]
        bias_a = network_params[2]
        weight_h = network_params[3]
        weight_a = network_params[4]


        if deriv_params == "bias_v":
            total_grad_r = np.zeros(self.n_v)
            total_grad_i = np.zeros(self.n_v)

            for sample in samples:
                conf  = sample2conf(sample)
                total_grad_r += derivative_bias_v(conf, conf, "r")
                total_grad_i += derivative_bias_v(conf, conf, "i")

            total_grad_r /= len(samples)
            total_grad_i /= len(samples)





        elif deriv_params == "bias_h":
            total_grad_r = np.zeros(self.n_h)
            total_grad_i = np.zeros(self.n_h)

            for sample in samples:
                conf  = sample2conf(sample)
                total_grad_r += derivative_bias_h(conf, conf, bias_h,
                                                  weight_h, "r")
                total_grad_i += derivative_bias_h(conf, conf, bias_h,
                                                  weight_h, "i")


            total_grad_r /= len(samples)
            total_grad_i /= len(samples)

        elif deriv_params == "bias_a":
            total_grad_r = np.zeros(self.n_a)
            total_grad_i = np.zeros(self.n_a)

            for sample in samples:
                conf  = sample2conf(sample)
                total_grad_r += derivative_bias_a(conf, conf, bias_a,
                                                  weight_a, "r")
                total_grad_i += derivative_bias_a(conf, conf, bias_a,
                                                  weight_a, "i")

            total_grad_r /= len(samples)
            total_grad_i /= len(samples)

        elif deriv_params == "weight_h":
            total_grad_r = np.zeros(self.n_h, self.n_v)
            total_grad_i = np.zeros(self.n_h, self.n_v)

            for sample in samples:
                conf  = sample2conf(sample)
                total_grad_r += derivative_weight_h(conf, conf, bias_h,
                                                    weight_h, "r")
                total_grad_i += derivative_weight_h(conf, conf, bias_h,
                                                    weight_h, "i")


            total_grad_r /= len(samples)
            total_grad_i /= len(samples)

        elif deriv_params == "weight_a":
            total_grad_r = np.zeros(self.n_a, self.n_v)
            total_grad_i = np.zeros(self.n_a, self.n_v)

            for sample in samples:
                total_grad_r += derivative_weight_a(conf, conf, bias_a,
                                                    weight_a, "r")
                total_grad_i += derivative_weight_a(conf, conf, bias_a,
                                                    weight_a, "i")

            total_grad_r /= len(samples)
            total_grad_i /= len(samples)



        return(total_grad_r, total_grad_i)

    def fidelity(self, learning_step, rho_true):
        """Calculates the fidelity of the NN-density matrix and the true density
        matrix"""

        bias_v = self.biases_v_training[learning_step, :]
        bias_h = self.biases_h_training[learning_step, :]
        bias_a = self.biases_a_training[learning_step, :]

        weight_h = self.weights_h_training[learning_step, :, :]
        weight_a = self.weights_a_training[learning_step, :, :]

        rho_NN = np.zeros((2**self.n_v, 2**self.n_v), dtype=complex)
        for i in range(2**self.n_v):
            for j in range(2**self.n_v):
                rho_NN[i, j] = (self.calc_rho_ij(bias_v, bias_h, bias_a,
                                weight_h, weight_a, i, j))

        print(rho_NN)
        #print(rho_true)
        fidelity = np.trace( sqrtm( sqrtm(rho_NN) @ rho_true @ sqrtm(rho_NN)))

    def calc_rho_ij(self, bias_v, bias_h, bias_a, weight_h, weight_a, i, j):
        'calculates one density matrix element specified by (v1, v2)'

        rho_ij = 0
        for a in range(2**self.n_a):
            rho_ij += (calc_psi(bias_v, bias_h, bias_a, weight_h, weight_a, i, a)
                        * np.conj(calc_psi(bias_v, bias_h, bias_a, weight_h,
                                  weight_a, j, a)))
        return(rho_ij)


    def stochastic_gradient_descent(self, n_iterations, learning_rate, n_bases,
                                    samples, subset_size, rho_true):

        # add attributes to RBM:
        self.biases_v_training = np.zeros((n_iterations, self.n_v),
                                            dtype=complex)
        self.biases_h_training = np.zeros((n_iterations, self.n_h),
                                            dtype=complex)
        self.biases_a_training = np.zeros((n_iterations, self.n_a),
                                            dtype=complex)

        self.weights_h_training = np.zeros((n_iterations, self.n_h, self.n_v),
                                            dtype=complex)
        self.weights_a_training = np.zeros((n_iterations, self.n_a, self.n_v),
                                            dtype=complex)


        # assign randomly initialized parameters of RBM to first training step
        self.biases_v_training[0, :] = self.biases_v
        self.biases_h_training[0, :] = self.biases_h
        self.biases_a_training[0, :] = self.biases_a

        self.weights_h_training[0, :, :] = self.weights_h
        self.weights_a_training[0, :, :] = self.weights_a


        self.fidelities_training = np.zeros(n_iterations)

        for i in range(1, n_iterations):

            # set current parameters
            biases_v_step = self.biases_v_training[i-1, :]
            biases_h_step = self.biases_h_training[i-1, :]
            biases_a_step = self.biases_a_training[i-1, :]

            weights_h_step = self.weights_h_training[i-1, :, :]
            weights_a_step = self.weights_a_training[i-1, :, :]

            #wrap parameters as list to pass to functions later
            params_step = ([biases_v_step, biases_h_step, biases_a_step,
                            weights_h_step, weights_a_step])

            #calculate the fidelity
            #print(params_step)
            fidelities[i] = self.fidelity(i, rho_true)

            # output for current step
            print("Step " + str(i))
            print("Current Fidelity: " + str(fidelities[i]))

            #stochastic_gradient_descent: build random subsets for each step
            rand_subsets = np.zeros((n_bases, subset_size))
            for j in range(n_bases):
                rand_subsets[j, :] = random_subset(samples[j, :], subset_size)

            #calculate the updates for the parameters

            #initialize gradients
            gradient_bias_v_r = np.zeros(self.n_v, dtype=float)
            gradient_bias_v_c = np.zeros(self.n_v, dtype=complex)

            gradient_bias_h_r = np.zeros(self.n_h, dtype=float)
            gradient_bias_h_c = np.zeros(self.n_h, dtype=complex)

            gradient_bias_a_r = np.zeros(self.n_a, dtype=float)
            gradient_bias_a_c = np.zeros(self.n_a, dtype=complex)

            gradient_weight_h_r = np.zeros((self.n_h, self.n_v), dtype=float)
            gradient_weight_h_c = np.zeros((self.n_h, self.n_v), dtype=complex)

            gradient_weight_a_r = np.zeros((self.n_a, self.n_v), dtype=float)
            gradient_weight_a_c = np.zeros((self.n_a, self.n_v), dtype=complex)

            #calculate derivatives for each basis
            """
            for k in range(n_bases):

                # first part
                n_basis_samples = np.shape(rand_subsets[k, :])[1]
                gradient_bias_v += (derivative_sample_mean(self, rand_subsets[k,
                                    :], params_step, "bias_v")/n_basis_samples)
                gradient_bias_v += (derivative_sample_mean(self, rand_subsets[k,
                                    :], params_step, "bias_v")/n_basis_samples)
                gradient_bias_v += (derivative_sample_mean(self, rand_subsets[k,
                                    :], params_step, "bias_v")/n_basis_samples)

                gradient_bias_v += (derivative_sample_mean(self, rand_subsets[k,
                                    :], params_step, "bias_v")/n_basis_samples)
                gradient_bias_v += (derivative_sample_mean(self, rand_subsets[k,
                                    :], params_step, "bias_v")/n_basis_samples)

                #MCMC partt
            """





            #update the perameters, combine real and complex gradients
            self.biases_v_training[i, :] = biases_v_step - lr * (gradient_bias_v_r + 1j * gradient_bias_v_c)
            self.biases_h_training[i, :] = biases_h_step - lr * (gradient_bias_h_r + 1j * gradient_bias_h_c)
            self.biases_a_training[i, :] = biases_a_step - lr * (gradient_bias_a_r + 1j * gradient_bias_a_c)

            self.weights_h_training[i, :, :] = weights_h_step - lr * (gradient_weight_h_r + 1j*gradient_weight_h_c)
            self.weights_a_training[i, :, :] = weights_a_step - lr * (gradient_weight_a_r + 1j*gradient_weight_a_c)

'Functions'

def state2index(state, n_qubits):
    'transforms a state to its associated index'

def calc_psi(bias_v, bias_h, bias_a, weight_h, weight_a, v, a):
    'calculate the probability p(v, a) encoded by an RBM'

    P = np.exp(np.sum(np.log(1 + np.exp(np.dot(weight_h, v) + bias_h)), axis=0)
               + np.dot(a.T, np.dot(weight_a, v)) + np.dot(bias_v.T, v)
               + np.dot(bias_a.T, a))
    return(P)



def random_subset(set, subset_size):
    """ determines a random subset with size=subset_size from set
        set: list of elements
        subset_size: int"""

    randints = np.random.randint(0, len(set), size=subset_size)
    subset = [set[i] for i in randints]

    return(subset)




def metropolis_hastings(n_samples):
    """finds samples that are disributed according to a probability distribution
    whose normalization constant we don't know."""
    samples = []

    for i in range(1, n_samples):
        previous_sample = samples[i - 1]

        proposed_sample = previous_sample.copy()
        r = np.random.randint(0, len(samples[i]))
        proposed_sample[r] = np.abs(proposed_sample[r] - 1) #flips 0<->1

        #replace p with calc_rho
        p_a = np.min(np.array([1, (calc_rho_ij(proposed_sample)
                                    /calc_rho_ij(previous_sample))]))

        rand_p = np.random.uniform()

        if p_a > rand_p:
            samples[i] = proposed_sample
        else:
            samples[i] = previous_sample


    return(samples)


def sample2conf(sample):
    """takes as input a measurent string and returns the corresponding values of
    the visible units as a np.array"""

    sample_list = list(sample)
    conf = np.array(list(map(int, sample_list)))

    return(conf)


def basis_sum(bases_samples):
    """Calculates the derivative over several basis states"""
