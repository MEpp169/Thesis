from RBM_purification import *

"Testing of Quantum Channel Implementation"

pres = 3
np.set_printoptions(precision = pres, linewidth = 150)

class qa_RBM():
    "RBM with qubits + auxiliaries in visible layer"

    def __init__(self, n_q, n_a, n_h):
        self.n_v = n_q + n_a
        self.n_h = n_h
        self.n_q = n_q
        self.n_a = n_a

        self.b_v =  (np.random.uniform(-1, 1, (self.n_v)) + 1j * np.random.uniform(-1, 1, (self.n_v)))
        self.b_h =  (np.random.uniform(-1, 1, (self.n_h)) + 1j * np.random.uniform(-1, 1, (self.n_h)))
        #self.w_hv = (np.random.uniform(-1, 1, (self.n_h, self.n_v)) + 1j * np.random.uniform(-1, 1, (self.n_h, self.n_v)))
        self.w_hv = np.zeros((self.n_h, self.n_v), dtype=complex)

    def psi(self, q, a):
        "calculates wavefunction for state (q, a) = qubits, auxiliaries"

        psi = 0
        for h_num in range(2**self.n_h):
            h = (2*np.array([sample2conf(index2state(h_num, self.n_h))]) - 1).T
            psi += np.exp(self.E_RBM(q, a, h))

        norm = 0
        for q_num in range(2**self.n_q):
            q = (2*np.array([sample2conf(index2state(q_num, self.n_q))]) - 1).T
            for a_num in range(2**self.n_a):
                a = (2*np.array([sample2conf(index2state(a_num, self.n_a))]) - 1).T
                cf = 0
                for h_num in range(2**self.n_h):
                    h = (2*np.array([sample2conf(index2state(h_num, self.n_h))]) - 1).T
                    cf += np.exp(self.E_RBM(q, a, h))
                norm += np.abs(cf)**2
                #norm += np.exp(self.E_RBM(q, a, h))

        return(psi/np.sqrt(norm))
        #return(psi/norm)

    def E_RBM(self, q, a, h):
        "returns energy of RBM"
        v = np.concatenate([q, a])
        E = (self.b_v).T @ v + (self.b_h).T @ h + h.T @ (self.w_hv @ v)
        return(np.asscalar(E))

    def check_psi_normalized(self):
        s = 0
        for q_num in range(2**self.n_q):
            q = (2*np.array([sample2conf(index2state(q_num, self.n_q))]) - 1).T
            for a_num in range(2**self.n_a):
                a = (2*np.array([sample2conf(index2state(a_num, self.n_a))]) - 1).T
                s += np.abs(self.psi(q, a))**2

        print("normalization:")
        print(s)

    def rho_ij(self, i, j):
        'calculates one density matrix element specified by (v1, v2)'

        rho_ij = 0
        for a in range(2**self.n_a):
            a_conf = (2*sample2conf(index2state(a, self.n_a))-1).T
            rho_ij += self.psi(i, a_conf) * np.conj(self.psi(j, a_conf))

        return(rho_ij)

    def rho(self):
        rho_RBM = np.zeros((2**self.n_q, 2**self.n_q), dtype=complex)
        for i in range(2**self.n_q):
            v1 = (2*sample2conf(index2state(i, self.n_q))-1).T
            for j in range(2**self.n_q):
                v2 = (2*sample2conf(index2state(j, self.n_q))-1).T
                rho_RBM[i, j] = self.rho_ij(v1, v2)

        self.rho = rho_RBM


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
        old_biases_v = np.copy(self.b_v)
        old_biases_h = np.copy(self.b_h)

        old_weights_h = np.copy(self.w_hv)

        #update visible biases
        self.b_v[j_ind] = alpha_1
        self.b_v[k_ind] = alpha_2

        #update hidden biases
        self.b_h = np.zeros(self.n_h, dtype=complex)
        self.b_h[:-2] = old_biases_h
        self.b_h[-2] = beta_1 + old_biases_v[j_ind]
        self.b_h[-1] = beta_2 + old_biases_v[k_ind]

        #update weight_matrix
        self.w_hv = np.zeros((self.n_h, self.n_v), dtype=complex)
        self.w_hv[:-2, :j_ind] = old_weights_h[:, :j_ind]
        self.w_hv[:-2, j_ind+1:k_ind] = old_weights_h[:, j_ind+1:k_ind]
        self.w_hv[:-2, k_ind+1:] = old_weights_h[:, k_ind+1:]
        self.w_hv[-2, j_ind] = Omega[0, 0]
        self.w_hv[-1, j_ind] = Omega[1, 0]
        self.w_hv[-2, k_ind] = Omega[0, 1]
        self.w_hv[-1, k_ind] = Omega[1, 1]

        #introduce X (=h-h matrix)
        self.w_X = np.zeros((self.n_h, self.n_h), dtype=complex)
        self.w_X[-2, :-2] = old_weights_h[:, j_ind].T
        self.w_X[-1, :-2] = old_weights_h[:, k_ind].T
        self.w_X[:-2, -2] = old_weights_h[:, j_ind]
        self.w_X[:-2, -1] = old_weights_h[:, k_ind]
        self.w_X[-1, -2] = gamma_entry
        self.w_X[-2, -1] = gamma_entry

        #introduce Y (=v-v matrix)
        self.w_Z = np.zeros((self.n_v, self.n_v), dtype=complex)
        self.w_Z[j_ind, k_ind] = lambda_entry
        self.w_Z[k_ind, j_ind] = lambda_entry

    def UBM_psi(self, q, a):
        p = 0
        for h_prime in range(2**self.n_h): #different procedure for UBM
            h_prime_conf = 2*sample2conf(index2state(h_prime, self.n_h))-1
            p +=  np.exp(self.UBM_energy(q, a, h_prime_conf))

        norm = 0
        for q_num in range(2**self.n_q):
            q = 2*sample2conf(index2state(q_num, self.n_q))-1
            for a_num in range(2**self.n_a):
                a = 2*sample2conf(index2state(a_num, self.n_a))-1
                psi = 0
                for h_num in range(2**self.n_h):
                    h = 2*sample2conf(index2state(h_num, self.n_h))-1
                    psi += np.exp(self.UBM_energy(q, a, h))
                norm += np.abs(psi)**2
        p /= np.sqrt(norm)
        return(p)

    def UBM_calc_rho_ij(self, i, j):
        'calculates one density matrix element specified by (v1, v2)'

        rho_ij = 0
        for a_num in range(2**self.n_a):
            a = 2*sample2conf(index2state(a_num, self.n_a))-1
            rho_ij += self.UBM_psi(i, a) * np.conj(self.UBM_psi(j, a))

        return(rho_ij)

    def UBM_rho(self):
        rho_UBM = np.zeros((2**self.n_q, 2**self.n_q), dtype=complex)
        for i in range(2**self.n_q):
            v1 = 2*sample2conf(index2state(i, self.n_q))-1
            for j in range(2**self.n_q):
                v2 = 2*sample2conf(index2state(j, self.n_q))-1
                rho_UBM[i, j] = self.UBM_calc_rho_ij(v1, v2)

        self.rho_encoded_UBM = rho_UBM

    def UBM_energy(self, q, a, h):
        v = np.concatenate([q, a])
        'calculates the energy of a UBM configuration'
        E =(np.dot(self.b_v, v) + np.dot(self.b_h, h) +
            np.dot(h, self.w_hv @ v) + 1/2*np.dot(h, self.w_X @ h)
            + 1/2*np.dot(v, self.w_Z @ v))

        return(E)

test = qa_RBM(2, 2, 6)

q = np.array([[1, 1]]).T
a = np.array([[1, 1]]).T

test.rho()
print(np.trace(test.rho))

test.check_psi_normalized()

#interaction parameters
alpha_1 = 0
alpha_2 = 0
beta_1 = 0
beta_2 = 0
Gamma =  np.zeros((2, 2))
lam = np.pi/4
Lambda = 1j*lam*np.array([[0, 1], [1, 0]])
n = 2
Omega = 1j*(2 * n + 1) * np.pi / 4 * np.array([[1, 0], [0, 1]]) # actually (2 * n + 1)


print("squared trace before unitary: ")
print(np.trace(test.rho @ test.rho))

print("applying gate...")
test.UBM_update_double(1, 2, alpha_1, alpha_2, beta_1, beta_2, Gamma, Lambda, Omega)
test.UBM_rho()
print("... done")

print("squared trace after unitary: ")
print(np.trace(test.rho_encoded_UBM @ test.rho_encoded_UBM))
