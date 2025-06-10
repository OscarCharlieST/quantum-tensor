import numpy as np
from ncon import ncon
from qtensor.states import *
from qtensor.operators import *

def verify_norm(psi):
    if np.allclose(overlap(psi,psi), 1.0+0j):
        return True
    print("State not normalised")
    return False

def verify_lcan(A):
    if np.allclose(
        ncon((A, A.conj()), ((1, 2, -1), (1, 2, -2))), np.eye(A.shape[2])):
        return True
    print("Not left canonical")
    return False

def verify_rcan(B): 
    if np.allclose(
        ncon((B, B.conj()), ((1, -1, 2), (1, -2, 2))), np.eye(B.shape[1])):
        return True
    print("Not right canonical")
    return False

def verify_expectation(state1, state2, operator):
    """
    Verify that the expectation value of an operator is the same for two states.
    """
    expect1 = mpo_expect(state1, operator)
    expect2 = mpo_expect(state2, operator)
    if np.allclose(expect1, expect2):
        return True
    print("Expectation values do not match")
    return False

