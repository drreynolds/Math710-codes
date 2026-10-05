#!/usr/bin/env python
#
# Driver that uses MRI_stab_region to plot the slow stability regions
# S_{rho,alpha} of the explicit MRI-GARK-ERK33a method, for rho = inf
# (reproducing Figure 1(b) of Sandu, SIAM J. Numer. Anal. 57(5), 2019)
# and for rho = 10.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import matplotlib.pyplot as plt
from MRI import MRIGARKERK33a
from MRI_stab_region import MRI_stab_region

alphas = [10, 45, 80]
box = [-3.5, 0.5, -3, 3]
C = MRIGARKERK33a()

fig, axs = plt.subplots(1, 2, figsize=(10, 6))
MRI_stab_region(C, alphas, np.inf, box, axs[0])
axs[0].set_title(r'MRI-GARK-ERK33a: $\mathcal{S}_{\rho=\infty,\alpha}$')
MRI_stab_region(C, alphas, 10.0, box, axs[1])
axs[1].set_title(r'MRI-GARK-ERK33a: $\mathcal{S}_{\rho=10,\alpha}$')
plt.tight_layout()
plt.savefig('MRIGARKERK33a_stability.png')

plt.show()
