import numpy as np
import matplotlib.pyplot as plt
import qtensor.states as states 
import qtensor.operators as ops
from qtensor.simulation.finiteTDVP import tdvp, right_mpo_contractions
import qtensor.thermofield as thf 
from importlib import reload

# import os
# abspath = os.path.abspath(__name__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

# a = np.random.rand(2,3,3) + 1j * np.random.rand(2,3,3)
# b = np.random.rand(2,3,3) + 1j * np.random.rand(2,3,3)

# psi = states.mps([a, a, b, a, b])
# psi.right_canonical()

# H = ops.tilted_ising(N=5)

# total_z_mpo = ops.total_z(5)
# middle_z = ops.single_site_pauli(2, 'z')
# middle_x = ops.single_site_pauli(2, 'x')
# middle_H = ops.tilted_ising_local_term(1)

# ops.expect(psi, H)

# ops.expect(psi, total_z_mpo)

# ops.local_expect(psi, middle_z)

# ops.local_expect(psi, middle_H)

# state_hist, L_con, R_con = tdvp(psi, H, 1, 0.01, history=True)

# ipsi = thf.infinite_T_thermofield(5, 8, noise=0.01)
# H_th = ops.thermofield_hamiltonian(H)
# ipsi.right_canonical()
# hist_th, l_th, r_th = tdvp(ipsi, H_th, 0.1, 0.01, history=True, verbose=True)
# ipsi = hist_th[max(hist_th.keys())]
# ipsi.right_canonical()
# print(states.partite_entropy(ipsi, 2))

# ent = states.partite_entropy(psi, 2)
N = 8
D = 8
H_usual = ops.tilted_ising(N=N)
beta = 0.1
H_th = thf.thermofield_hamiltonian(H_usual)
H_usual.sites
beta_psi = thf.finite_T_thermofield(beta, N, D, H_th, noise=0.1, steps=20)
print(H_usual[0].shape)
#