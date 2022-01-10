import numpy as np
import matplotlib.pyplot as plt

from abtem.ionization import SubshellTransitions, TransitionPotential, EELSDetector
from abtem import SMatrix, Potential, GridScan
from ase import units
from ase.io import read

Z = 13 # atomic number
n = 2 # principal quantum number
l = 1 # azimuthal quantum number
order =1
gpernode =150
xc = 'PBE' # exchange-correlation functional

for e in [0.0001]:
    transitions = SubshellTransitions(Z = Z, n = n, l = l, xc = 'PBE',epsilon=e*units.Rydberg,order=order)
    gos,k = transitions.get_gos(dirac=False)
    index = (units.Bohr*k)**2>0.01
    for lp in range(3):
        # plt.plot(((new_k[index])**2),np.sum(gos[:],axis=0)[index],label=f'E = {e}')
        plt.plot(((k[index]*units.Bohr)**2),gos[lp][index],label=f'E = {e}, lprime = {lp}')
plt.xscale('log')
plt.legend()