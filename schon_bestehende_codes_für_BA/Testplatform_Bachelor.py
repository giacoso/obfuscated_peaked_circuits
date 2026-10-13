#%config InlineBackend.figure_formats = ['svg']
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize import TNOptimizer
from scipy.stats import ortho_group
import matplotlib.pyplot as plt

L=5

psi = qtn.MPS_computational_state('0' * L)

psi.draw()