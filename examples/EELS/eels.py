# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt

from abtem.ionization import SubshellTransitions, TransitionPotential, EELSDetector
from abtem import SMatrix, Potential, GridScan

from ase import units
from ase.io import read


# %%
Z = 8 # atomic number
n = 1 # principal quantum number
l = 0 # azimuthal quantum number
xc = 'PBE' # exchange-correlation functional

O_transitions = SubshellTransitions(Z = Z, n = n, l = l, xc = 'PBE')

print('bound electron configuration:', O_transitions.bound_configuration)
print('ionic electron configuration:', O_transitions.excited_configuration)


# %%
for bound_state, continuum_state in O_transitions.get_transition_quantum_numbers():
    print(f'(l, ml) = {bound_state} -> {continuum_state}')


# %%
atomic_transition_potentials = O_transitions.get_transition_potentials(extent = 5,
                                                                       gpts = 256,
                                                                       energy = 100e3)


# %%
fig, ax1 = plt.subplots(1,1, figsize = (10,5))

atomic_transition_potentials[0].show(ax=ax1)


# %%
fig, axes = plt.subplots(1,3, figsize = (10,5))

for ax, atomic_transition_potential in zip(axes, atomic_transition_potentials):
    # print(str(atomic_transition_potential))
    atomic_transition_potential.show(ax = ax, title = str(atomic_transition_potential))


# %%
atoms = read('../data/srtio3_100.cif') * (2,2,1)
atoms.center(axis = 2)

from abtem import show_atoms
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10,2))

show_atoms(atoms, ax = ax1, title = 'Beam view')
show_atoms(atoms, ax = ax2, plane = 'yz', title = 'Side view')
show_atoms(atoms[atoms.numbers == 8], ax = ax3, plane = 'xy', title = 'Beam view (Oxygen)');


# %%
len(O_transitions)


# %%
transition_potential = TransitionPotential(O_transitions,
                                           atoms = atoms,
                                           sampling = .05,
                                           energy = 100e3,
                                           slice_thickness = 2)


# %%
transition_potential.num_edges


# %%
transition_potential.show()

#%%

# %%
S = SMatrix(energy = 100e3, semiangle_cutoff = 25) # interpolation not implemented!


detector = EELSDetector(collection_angle = 100, interpolation = 4)


potential = Potential(atoms, sampling = .05, slice_thickness = .5,
                      projection = 'infinite', parametrization = 'kirkland')


transition_potential = TransitionPotential(O_transitions)


scan = GridScan((0,0), potential.extent, sampling = .9*S.ctf.nyquist_sampling)


measurement = S.coreloss_scan(scan, detector, potential, transition_potential)


# %%
print(potential.extent,.9*S.ctf.nyquist_sampling)


# %%
s_array=scan.get_positions()
s_array


# %%
measurement.tile((2,2)).interpolate(.02).show(figsize = (6,6));


# %%
O_transitions = SubshellTransitions(Z = 8, n = 1, l = 0, xc = 'PBE')
Ti_transitions = SubshellTransitions(Z = 22, n = 2, l = 1, xc = 'PBE')
Sr_transitions = SubshellTransitions(Z = 38, n = 2, l = 1, xc = 'PBE')

transitions = [O_transitions, Ti_transitions, Sr_transitions]

transition_potential = TransitionPotential(transitions)


# %%
print('Oxygen:')
for bound_state, continuum_state in O_transitions.get_transition_quantum_numbers():
    print(f'(l, ml) = {bound_state} -> {continuum_state}')
print('Titanium:')
for bound_state, continuum_state in Ti_transitions.get_transition_quantum_numbers():
    print(f'(l, ml) = {bound_state} -> {continuum_state}')
print('Strontium:')
for bound_state, continuum_state in Sr_transitions.get_transition_quantum_numbers():
    print(f'(l, ml) = {bound_state} -> {continuum_state}')


# %%
measurements = S.coreloss_scan(scan, detector, potential, transition_potential)


# %%
measurements = S.coreloss_scan(scan, detector, potential, transition_potential)


# %%
fig, (ax1, ax2, ax3)= plt.subplots(1, 3, figsize = (10,2.7))

measurements[0].tile((2, 2)).interpolate(.1).show(ax = ax1)
measurements[1].tile((2, 2)).interpolate(.1).show(ax = ax2)
measurements[2].tile((2, 2)).interpolate(.1).show(ax = ax3);

plt.savefig('low_res_eels.jpg', bbox_inches='tight')


# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,4))

measurements[0].interpolate_line((0,0), (0, potential.extent[1]),
                                 sampling = .1).show(ax = ax1, label = 'O')
measurements[1].interpolate_line((0,0), (0, potential.extent[1]),
                                 sampling = .1).show(ax = ax1, label = 'Ti')
measurements[2].interpolate_line((0,0), (0, potential.extent[1]),
                                 sampling = .1).show(ax = ax1, label = 'Sr')
ax1.legend()

measurements[0].interpolate_line((atoms[3].x, 0), (atoms[3].x,potential.extent[1]),
                                 sampling = .1).show(ax = ax2, label = 'O')
measurements[1].interpolate_line((atoms[3].x, 0),(atoms[3].x, potential.extent[1]),
                                 sampling = .1).show(ax = ax2, label = 'Ti')
measurements[2].interpolate_line((atoms[3].x, 0),(atoms[3].x,potential.extent[1]),
                                 sampling = .1).show(ax = ax2, label = 'Sr')
ax2.legend();


# %%



