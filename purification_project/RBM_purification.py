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

    def __init__(self, n_v, n_h, n_a, i=1):
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
        self.biases_v = (np.random.uniform(-1, 1, (self.n_v)) + 1j * np.random.uniform(-1, 1, (self.n_v)))
        self.biases_h = (np.random.uniform(-1, 1, (self.n_h)) + 1j * np.random.uniform(-1, 1, (self.n_h)))
        self.biases_a = (np.random.uniform(-1, 1, (self.n_a)) + 1j * np.random.uniform(-1, 1, (self.n_a)))
        #self.biases_a = np.zeros(self.n_a)

        self.weights_h = ((np.random.uniform(-1, 1, (self.n_h, self.n_v)) +
                            1j * np.random.uniform(-1, 1, (self.n_h, self.n_v))))
        self.weights_a = i*((np.random.uniform(-1, 1, (self.n_a, self.n_v)) +
                            1j * np.random.uniform(-1, 1, (self.n_a, self.n_v))))
        #self.weights_a = np.zeros((self.n_a, self.n_v))

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
                up1 = 1/self(np.exp(-(np.dot(weight_h[i, :], v1) + bias_h[i])) + 1)
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

    def calc_rho_NN(self):
        rho_NN = np.zeros((2**self.n_v, 2**self.n_v), dtype=complex)
        for i in range(2**self.n_v):
            v1 = np.array([sample2conf(index2state(i, self.n_v))]).T
            for j in range(2**self.n_v):
                v2 = np.array([sample2conf(index2state(j, self.n_v))]).T
                rho_NN[i, j] = (self.calc_rho_ij(self.biases_v, self.biases_h,
                                self.biases_a, self.weights_h, self.weights_a,
                                v1, v2))
        self.rho_encoded = rho_NN


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
            v1 = np.array([sample2conf(index2state(i, self.n_v))]).T
            for j in range(2**self.n_v):
                v2 = np.array([sample2conf(index2state(j, self.n_v))]).T
                rho_NN[i, j] = (self.calc_rho_ij(bias_v, bias_h, bias_a,
                                weight_h, weight_a, v1, v2))

        #print(rho_true)
        fidelity = np.trace( sqrtm( sqrtm(rho_NN) @ rho_true @ sqrtm(rho_NN)))

    def calc_rho_ij(self, bias_v, bias_h, bias_a, weight_h, weight_a, i, j):
        'calculates one density matrix element specified by (v1, v2)'

        rho_ij = 0

        for a in range(2**self.n_a):
            a_conf = np.array([sample2conf(index2state(a, self.n_a))]).T
            rho_ij += (self.calc_psi(bias_v, bias_h, bias_a, weight_h, weight_a, i, a_conf)
                        * np.conj(self.calc_psi(bias_v, bias_h, bias_a, weight_h,
                                    weight_a, j, a_conf)))
        return(rho_ij)

    def calc_psi_vversion(self, v):
        P = 0
        for h_prime in range(2**self.n_v):
            h_prime_conf = np.array([sample2conf(index2state(h_prime, self.n_h))]).T
            h_prime_conf = 2*h_prime_conf - 1
            E =(np.dot(self.biases_v, v) + np.dot(self.biases_h, h_prime_conf) +
                 np.dot(h_prime_conf, self.weights_h @ v))
            P += np.exp(E)
        return(P)


    def calc_psi(self, bias_v, bias_h, bias_a, weight_h, weight_a, v, a):
        if self.nodeType == "-11":
            v = 2*v-1
            a = 2*a-1
            P = np.exp(np.sum(np.log(np.exp(-weight_h @ v - np.array([bias_h]).T) + np.exp(weight_h @ v + np.array([bias_h]).T)), axis=0)
                       + np.dot(a.T, weight_a @ v ) + np.dot(bias_v.T, v) + np.dot(bias_a.T, a))

            norm = 0
            for v_prime in range(2**self.n_v):
                v_prime_conf = np.array([sample2conf(index2state(v_prime, self.n_v))]).T
                v_prime_conf = 2*v_prime_conf-1
                for a_prime in range(2**self.n_a):
                    a_prime_conf = np.array([sample2conf(index2state(a_prime, self.n_a))]).T
                    a_prime_conf = 2*a_prime_conf-1
                    norm += np.abs(np.exp(np.sum(np.log(np.exp(-weight_h @  v_prime_conf  -np.array([bias_h]).T) + np.exp(weight_h @ v_prime_conf +
                            np.array([bias_h]).T)), axis=0) + np.dot(a_prime_conf.T,
                            weight_a @ v_prime_conf ) + np.dot(bias_v.T, v_prime_conf) +
                            np.dot(bias_a.T, a_prime_conf)))**2


        elif self.nodeType == "01":
            P = np.exp(np.sum(np.log(1 + np.exp(weight_h @ v + np.array([bias_h]).T)), axis=0)
                       + np.dot(a.T, weight_a @ v ) + np.dot(bias_v.T, v)
                       + np.dot(bias_a.T, a))

            norm = 0

            for v_prime in range(2**self.n_v):
                v_prime_conf = np.array([sample2conf(index2state(v_prime, self.n_v))]).T
                #v_prime_conf = 2*v_prime_conf-1
                for a_prime in range(2**self.n_a):
                    a_prime_conf = np.array([sample2conf(index2state(a_prime, self.n_a))]).T
                    #a_prime_conf = 2*a_prime_conf-1
                    norm += np.abs(np.exp(np.sum(np.log(1 + np.exp(weight_h @ v_prime_conf +
                            np.array([bias_h]).T)), axis=0) + np.dot(a_prime_conf.T,
                            weight_a @ v_prime_conf ) + np.dot(bias_v.T, v_prime_conf) +
                            np.dot(bias_a.T, a_prime_conf)))**2

        #return(P/np.sqrt(norm))
        return(P/np.sqrt(norm))

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
            self.fidelities_training[i] = self.fidelity(i, rho_true)

            # output for current step
            print("Step " + str(i))
            print("Current Fidelity: " + str(self.fidelities_training[i]))

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

    def check_rho_valid(self):
        # use np.around(a, decimals=15) to compare arrays online up to acceptable precision
        assert ((np.conj(np.around(self.rho_encoded.T, decimals = 15)) ==
                    np.around(self.rho_encoded, decimals = 15)).all(),
                    "matrix should be hermitian")

        assert (np.round(np.trace(self.rho_encoded), decimals=0) == 1,
                "trace should be one")

        assert (np.all(np.linalg.eigvalsh(self.rho_encoded) >= 0),
                "rho should be positive semidefinite")

        print("The RBM encodes a valid density matrix")



    'now update the density matrix according to the new formalism'

    def UBM_update_single(self, alpha, beta, omega, A, j):
        'modifies the RBM to output a new state'

        'this code has to be adapted to also work for successive gates'

        #set nodeType to {-1, 1}
        self.nodeType = "-11"


        #new hidden node
        self.n_h += 1

        #store old parameters
        old_biases_v = np.copy(self.biases_v)
        old_biases_h = np.copy(self.biases_h)
        old_biases_a = np.copy(self.biases_a)

        old_weights_h = np.copy(self.weights_h)
        old_weights_a = np.copy(self.weights_a)

        self.biases_v[j] = alpha

        self.biases_h = np.zeros(self.n_h, dtype = complex)
        self.biases_h[:-1] = old_biases_h
        self.biases_h[-1] = beta + old_biases_v[j]
        #print(self.biases_h)

        self.weights_h = np.zeros((self.n_h, self.n_v), dtype = complex)
        self.weights_h[:-1, :j] = old_weights_h[:, :j]
        self.weights_h[-1, j] = omega
        self.weights_h[:-1, j+1:] = old_weights_h[:, j+1:]
        #print(self.weights_h)

        self.weights_a[:, j] = np.zeros(self.n_a)


        'update h-h interactions: check if unitary has already been applied'

        if hasattr(self, "A"):
            print("already updated")
            old_weights_X = np.copy(self.weights_X)
            self.weights_X = np.zeros((self.n_h, self.n_h), dtype = complex)
            self.weights_X[:-1, :-1] = old_weights_X
            self.weights_X[-1, :-1] = old_weights_h[:, j].T
            self.weights_X[:-1, -1] = old_weights_h[:, j]
        else:
            print("First RBM update ")
            self.weights_X = np.zeros((self.n_h, self.n_h), dtype = complex)
            self.weights_X[-1, :-1] = old_weights_h[:, j].T
            self.weights_X[:-1, -1] = old_weights_h[:, j]

        if hasattr(self, "A"):
            print("already updated")
            old_weights_Y = np.copy(self.weights_Y)
            self.weights_Y = np.zeros((self.n_a, self.n_h), dtype = complex)
            self.weights_Y[:, :-1] = old_weights_Y
            self.weights_Y[:, -1] = old_weights_a[:, j].T

        else:
            print("First RBM update ")
            self.weights_Y = np.zeros((self.n_a, self.n_h), dtype = complex)
            self.weights_Y[:, -1] = old_weights_a[:, j].T



        # a-h interactions


        #print(self.weights_Y)

        #print(self.weights_X)
        #print(self.weights_Y)
        self.A = A


    def UBM_psi(self, v, a):
        p = 0
        for h_prime in range(2**self.n_h): #different procedure for UBM
            h_prime_conf = sample2conf(index2state(h_prime, self.n_h))
            p +=  np.exp(self.UBM_energy(v, h_prime_conf, a))

        norm = 0
        for v_prime in range(2**self.n_v):
            v_prime_conf = sample2conf(index2state(v_prime, self.n_v))
            for a_prime in range(2**self.n_a):
                a_prime_conf = sample2conf(index2state(a_prime, self.n_a))
                psi = 0
                for h_prime in range(2**self.n_h):
                    h_prime_conf = sample2conf(index2state(h_prime, self.n_h))
                    psi += np.exp(self.UBM_energy(v_prime_conf, h_prime_conf, a_prime_conf))
                norm += np.abs(psi)**2
        p /= np.sqrt(norm)
        return(p)

    def UBM_calc_rho_ij(self, i, j):
        'calculates one density matrix element specified by (v1, v2)'

        rho_ij = 0
        for a in range(2**self.n_a):
            a_conf = sample2conf(index2state(a, self.n_a))
            rho_ij += self.UBM_psi(i, a_conf) * np.conj(self.UBM_psi(j, a_conf))

        return(rho_ij)

    def UBM_rho(self):
        rho_UBM = np.zeros((2**self.n_v, 2**self.n_v), dtype=complex)
        for i in range(2**self.n_v):
            v1 = sample2conf(index2state(i, self.n_v))
            for j in range(2**self.n_v):
                v2 = sample2conf(index2state(j, self.n_v))
                rho_UBM[i, j] = self.UBM_calc_rho_ij(v1, v2)

        self.rho_encoded_UBM = rho_UBM

    def RBM_energy(self, v, h):
        #calculates the energy of a restricted Boltzmann machine
        E = np.dot(self.biases_v, v) + np.dot(self.biases_h, h) + np.dot(h, self.weights_h @ v)
        return(E)

    def UBM_energy(self, v, h, a):
        'calculates the energy of a UBM configuration'
        if self.nodeType == "-11":
            v = 2*v-1
            h = 2*h-1
            a = 2*a-1
            E =(np.dot(self.biases_v, v) + np.dot(self.biases_h, h) +
                np.dot(self.biases_a, a) + np.dot(h, self.weights_h @ v) +
                np.dot(a, self.weights_a @ v) + 1/2*np.dot(h, self.weights_X @ h) +
                np.dot(a, self.weights_Y @ h) + 1/2*np.dot(v, self.weights_Z @ v))
        else:
            E =(np.dot(self.biases_v, v) + np.dot(self.biases_h, h) +
                np.dot(self.biases_a, a) + np.dot(h, self.weights_h @ v) +
                np.dot(a, self.weights_a @ v) + 1/2*np.dot(h, self.weights_X @ h) +
                np.dot(a, self.weights_Y @ h) )
        return(E)

    def RBM_energy_ahid(self, v, h, a):
        "calculates the energy of an RBM configuration where the auxiliary units are in the hidden layer"
        #vhelp = v.reshape((2, 1))
        """E =(np.dot(self.biases_v, v) + np.dot(self.biases_h, h) +
            np.dot(self.biases_a, a) + np.dot(h, (self.weights_h @ vhelp).flatten()) +
            np.dot(a, (self.weights_a @ vhelp).flatten()))"""
        E =(np.dot(self.biases_v, v) + np.dot(self.biases_h, h) +
            np.dot(self.biases_a, a) + np.dot(h, self.weights_h @ v) +
            np.dot(a, self.weights_a @ v))
        return(E)


    def UBM_update_double(self, j_ind, k_ind, alpha_1, alpha_2, beta_1, beta_2, Gamma, Lambda, Omega):
        """Updates the RBM to realize the effect of a 2-qubit gate
            - acts on qubits j, k

        """
        #set nodeType to "-1,1"
        self.nodeType = "-11"


        lambda_entry = Lambda[0, 1]
        gamma_entry = Gamma[0, 1]
        #two new hidden nodes
        self.n_h += 2

        #store old parameters
        old_biases_v = np.copy(self.biases_v)
        old_biases_h = np.copy(self.biases_h)
        old_biases_a = np.copy(self.biases_a)

        old_weights_h = np.copy(self.weights_h)
        old_weights_a = np.copy(self.weights_a)


        #update visible biases
        self.biases_v[j_ind] = alpha_1
        self.biases_v[k_ind] = alpha_2

        #update hidden biases
        self.biases_h = np.zeros(self.n_h, dtype=complex)
        self.biases_h[:-2] = old_biases_h
        self.biases_h[-2] = beta_1 + old_biases_v[j_ind]
        self.biases_h[-1] = beta_2 + old_biases_v[k_ind]

        #update weight_matrix
        "here could be the reason why successiive 2-qubit gates don't work! "
        if hasattr(self, "updated"):
            self.weights_h = np.zeros((self.n_h, self.n_v), dtype=complex)

            self.weights_h[:-2, :j_ind] = old_weights_h[:, :j_ind]
            self.weights_h[:-2, j_ind+1:k_ind] = old_weights_h[:, j_ind+1:k_ind]
            self.weights_h[:-2, k_ind+1:] = old_weights_h[:, k_ind+1:]

            self.weights_h[-2, :] = self.weights_Z[j_ind, :]
            self.weights_h[-1, :] = self.weights_Z[k_ind, :]
            #replace the Z-entries by unitary parameters
            self.weights_h[-2, j_ind] = Omega[0, 0]
            self.weights_h[-1, j_ind] = Omega[1, 0]
            self.weights_h[-2, k_ind] = Omega[0, 1]
            self.weights_h[-1, k_ind] = Omega[1, 1]
        else:
            self.weights_h = np.zeros((self.n_h, self.n_v), dtype=complex)
            self.weights_h[:-2, :j_ind] = old_weights_h[:, :j_ind]
            self.weights_h[:-2, j_ind+1:k_ind] = old_weights_h[:, j_ind+1:k_ind]
            self.weights_h[:-2, k_ind+1:] = old_weights_h[:, k_ind+1:]
            self.weights_h[-2, j_ind] = Omega[0, 0]
            self.weights_h[-1, j_ind] = Omega[1, 0]
            self.weights_h[-2, k_ind] = Omega[0, 1]
            self.weights_h[-1, k_ind] = Omega[1, 1]



        #don't forget a-v zeroing!
        self.weights_a[:, j_ind] = np.zeros(self.n_a)
        self.weights_a[:, k_ind] = np.zeros(self.n_a)


        if hasattr(self, "updated"):
            print("already updated")
            old_weights_X = np.copy(self.weights_X)
            self.weights_X = np.zeros((self.n_h, self.n_h), dtype=complex)
            self.weights_X[-2, :-2] = old_weights_h[:, j_ind].T
            self.weights_X[-1, :-2] = old_weights_h[:, k_ind].T
            self.weights_X[:-2, -2] = old_weights_h[:, j_ind]
            self.weights_X[:-2, -1] = old_weights_h[:, k_ind]
            self.weights_X[-1, -2] = gamma_entry + self.weights_Z[j_ind, k_ind]
            self.weights_X[-2, -1] = gamma_entry + self.weights_Z[j_ind, k_ind]
            self.weights_X[:-2, :-2] = old_weights_X


            self.weights_Z[j_ind, k_ind] = lambda_entry
            self.weights_Z[k_ind, j_ind] = lambda_entry

            # a-h interactions
            old_weights_Y = np.copy(self.weights_Y)
            self.weights_Y = np.zeros((self.n_a, self.n_h), dtype = complex)
            self.weights_Y[:, -2] = old_weights_a[:, j_ind].T
            self.weights_Y[:, -1] = old_weights_a[:, k_ind].T
            self.weights_Y[:, :-2] = old_weights_Y

        else:
            print("First RBM update ")
            self.weights_X = np.zeros((self.n_h, self.n_h), dtype=complex)
            self.weights_X[-2, :-2] = old_weights_h[:, j_ind].T
            self.weights_X[-1, :-2] = old_weights_h[:, k_ind].T
            self.weights_X[:-2, -2] = old_weights_h[:, j_ind]
            self.weights_X[:-2, -1] = old_weights_h[:, k_ind]
            self.weights_X[-1, -2] = gamma_entry
            self.weights_X[-2, -1] = gamma_entry


            self.weights_Z = np.zeros((self.n_v, self.n_v), dtype=complex)
            self.weights_Z[j_ind, k_ind] = lambda_entry
            self.weights_Z[k_ind, j_ind] = lambda_entry


            # a-h interactions
            self.weights_Y = np.zeros((self.n_a, self.n_h), dtype = complex)
            self.weights_Y[:, -2] = old_weights_a[:, j_ind].T
            self.weights_Y[:, -1] = old_weights_a[:, k_ind].T



        self.updated = True



'Functions'

def state2index(state, n_qubits):
    'transforms a state to its associated index'





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

def index2state(index, n_qubits):
    'transforms an index into a n-qubit state'
    return(bin(index)[2:].zfill(n_qubits))


def sample2conf(sample):
    """takes as input a measurent string and returns the corresponding values of
    the visible units as a np.array"""

    sample_list = list(sample)
    conf = np.array(list(map(int, sample_list)))

    return(conf)


def basis_sum(bases_samples):
    """Calculates the derivative over several basis states"""


def exp_unitary(n_spins, alpha, beta, omega):
    'outputs a single-qubit unitary that can be easily implemented in the RBM'

    U = np.zeros((2**n_spins, 2**n_spins), dtype=complex)
    U[0, 0] = np.exp(1j*(alpha + beta) + omega)
    U[0, 1] = -1j*np.exp(1j*(alpha - beta) - omega)
    U[1, 0] = -1j*np.exp(-1j*(alpha - beta) - omega)
    U[1, 1] = np.exp(-1j*(alpha + beta) + omega)

    U *= np.exp(1j*np.pi/4)/np.sqrt(2*np.cosh(2*omega))
    #U *= 1/(2*np.cosh(omega))

    return(U)

def exp2spin_unitary(alpha_p, beta_p, omega_p):
    """converts parameters that describe a unitary in basis vector form into
    parameters that are in binary spin form"""
    alpha = np.log(-1j) - 2j*alpha_p - 2*omega_p
    beta = np.log(-1j) - 2j*beta_p - 2*omega_p
    omega = -2*np.log(-1j) + 4*omega_p
    A =  np.exp(1j*np.pi/4 + 1j*(alpha_p+beta_p) + omega_p)/np.sqrt(2*np.cosh(2*omega_p))
    return(alpha, beta, omega, A)

def simple_exp_unitary(n_spins, a, b, c, A):
    U = np.zeros((2**n_spins, 2**n_spins), dtype=complex)
    U[0, 0] = 1
    U[0, 1] = np.exp(b)
    U[1, 0] = np.exp(a)
    U[1, 1] = np.exp(a + b + c)

    #U *= np.exp(1j*np.pi/4)/(2*np.cosh(omega))
    U *= A

    return(U)


def fidelity(A, B):
    "returns the fidelity of two density matrices"
    return(np.trace( sqrtm( sqrtm(A) @ B @ sqrtm(A))))


def two_body_entangling(lam):
    "returns an easy-to-implement two body unitary"

    U = 1/2 * np.ones((4, 4), dtype=complex)

    U[0, :] *=  np.exp(1j*lam)
    U[0, 0] *= 1j
    U[0, 3] *= -1j

    U[1, :] *=  np.exp(-1j*lam)
    U[1, 1] *= 1j
    U[1, 2] *= -1j

    U[2, :] *=  np.exp(-1j*lam)
    U[2, 1] *= -1j
    U[2, 2] *= 1j

    U[3, :] *=  np.exp(1j*lam)
    U[3, 0] *= -1j
    U[3, 3] *= 1j

    return(U)


def dagger(A):
    "returns hermitian conjugate of matrix"
    return(np.conj(A).T)
