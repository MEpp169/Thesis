import numpy as np
from generate_data import *
from RBM_purification import *
from unitary_channels import *
np.set_printoptions(precision = 2, linewidth = 200) # for nice format



"--------------- construct a less symmetric channel --------------------------"
alpha = 7
beta = 4
omega = 0.3

c_factor = 2 # if even: works, odd or fraction: not
d_factor = 8

U2 = two_body_entangling(np.pi/4*c_factor)


U_test = exp_unitary(1, alpha, beta, omega)
alpha_RBM, beta_RBM, omega_RBM, A_RBM = exp2spin_unitary(alpha, beta, omega)


id = np.array([[1, 0], [0, 1]])

U = np.kron(id, U_test)


U_tot =  U2 @ U

E1 = U_tot[:2, :2]
E2 = U_tot[2:, :2]

E1_dag = np.conj(E1).T
E2_dag = np.conj(E2).T

print("channel?")
print(E1_dag @ E1 + E2_dag @ E2)

"--------------- test out if it works --------------------------"

# 2-body unitary: RBM rep
alpha_1 = 0
alpha_2 = 0
beta_1 = 0
beta_2 = 0
Gamma =  np.zeros((2, 2))
lam = c_factor*np.pi/4
lamb = c_factor*np.pi/4*d_factor

Lambda = 1j*lam*np.array([[0, 1], [1, 0]])
Lambdab = 1j*lamb*np.array([[0, 1], [1, 0]])
n = 2 #should be even I think
Omega = 1j*(2 * n + 1) * np.pi / 4 * np.array([[1, 0], [0, 1]])



single_qubit_rho = qa_RBM(1, 1, 3)

single_qubit_rho.rho()
print("Density matrix before channel: ")
print(single_qubit_rho.rho)

rho_before_channel = np.copy(single_qubit_rho.rho)

rho_after_channel = E1 @ rho_before_channel @ E1_dag +  E2 @ rho_before_channel @ E2_dag

print("Density matrix after channel (analytical): ")
print(rho_after_channel)

print("applying unitary to whole system to realize channel")
#single_qubit_rho.UBM_update_double(0, 1, alpha_1, alpha_2, beta_1, beta_2, Gamma, Lambda, Omega)
single_qubit_rho.UBM_update_single(alpha_RBM, beta_RBM, omega_RBM, A_RBM, 1)
single_qubit_rho.UBM_update_double_prime( 0, 1, alpha_1, alpha_2, beta_1, beta_2, Gamma, Lambda, Omega)
single_qubit_rho.UBM_rho()
print("... done")

print("density matrix afterwards is")
print(single_qubit_rho.rho_encoded_UBM)

print("Deviation between both density matrices: ")
print(single_qubit_rho.rho_encoded_UBM - rho_after_channel)

print("\n")
print("\n")
print("\n")
