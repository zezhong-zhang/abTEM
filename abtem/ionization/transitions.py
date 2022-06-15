import contextlib
import os
from abc import ABCMeta, abstractmethod
from typing import Union, Sequence, Tuple

import numpy as np
from ase import units
from ase.data import chemical_symbols

from scipy import integrate
from scipy.interpolate import interp1d
from scipy.special import spherical_jn, sph_harm

from abtem.base_classes import HasAcceleratorMixin, HasGridMixin, Grid, Accelerator, Cache, cached_method
from abtem.device import get_array_module, get_device_function
from abtem.ionization.utils import check_valid_quantum_number, config_str_to_config_tuples, \
    remove_electron_from_config_str, load_electronic_configurations
from abtem.measure import Measurement, calibrations_from_grid
from abtem.utils import energy2wavelength, spatial_frequencies, polar_coordinates, \
    relativistic_mass_correction, relativistic_velocity, fourier_translation_operator
from abtem.utils import ProgressBar
from abtem.structures import SlicedAtoms
from quadpy.c1 import integrate_adaptive
from multiprocessing import Pool
from itertools import repeat
from sympy.physics.wigner import wigner_3j
from scipy.special import legendre
from abtem.ionization.dirac import orbital
from joblib import Parallel, delayed
import logging as log
# from numba import jit,njit
# from concurrent.futures import ThreadPoolExecutor

class AbstractTransitionCollection(metaclass=ABCMeta):

    def __init__(self, Z):
        self._Z = Z

    @property
    def Z(self):
        return self._Z

    @abstractmethod
    def get_transition_potentials(self):
        pass


class SubshellTransitions(AbstractTransitionCollection):

    def __init__(self, Z, n, l, order=1, min_contrast=1., epsilon=1, gpernode=150, xc='PBE', dirac=False):
        check_valid_quantum_number(Z, n, l)
        self._n = n
        self._l = l
        self._order = order
        self._min_contrast = min_contrast
        self._epsilon = epsilon
        self._xc = xc
        self.gpernode = gpernode
        self.dirac = dirac

        self._bound_cache = Cache(1)
        self._continuum_cache = Cache(1)
        self._bound_potential_cache = Cache(1)
        self._continuum_potential_cache = Cache(1)
        super().__init__(Z)

    @property
    def order(self):
        return self._order

    @property
    def min_contrast(self):
        return self._min_contrast

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def xc(self):
        return self._xc

    @property
    def n(self):
        return self._n

    @property
    def l(self):
        return self._l

    @property
    def lprimes(self):
        min_new_l = max(self.l - self.order, 0)
        return np.arange(min_new_l, self.l + self.order + 1)

    @property
    def subshell_occupancy(self):
        config_tuples = config_str_to_config_tuples(load_electronic_configurations()[chemical_symbols[self.Z]])
        subshell_index = [shell[:2] for shell in config_tuples].index((self.n, self.l))
        subshell_occupancy = config_tuples[subshell_index][-1]
        return subshell_occupancy

    def __len__(self):
        return len(self.get_transition_quantum_numbers())

    @property
    def ionization_energy(self):
        atomic_energy, _ = self._calculate_bound()
        if self.dirac == True:
            ionic_energy = 0.0
        else:
            ionic_energy, _ = self._calculate_continuum()
        return ionic_energy - atomic_energy

    @property
    def energy_loss(self):
        return self.ionization_energy + self.epsilon
    
    @property
    def bound_configuration(self):
        return load_electronic_configurations()[chemical_symbols[self.Z]]

    @property
    def excited_configuration(self):
        return remove_electron_from_config_str(self.bound_configuration, self.n, self.l)

    @cached_method('_bound_cache')
    def _calculate_bound(self):
        if self.dirac is False:
            from gpaw.atom.all_electron import AllElectron

            check_valid_quantum_number(self.Z, self.n, self.l)
            config_tuples = config_str_to_config_tuples(load_electronic_configurations()[chemical_symbols[self.Z]])
            subshell_index = [shell[:2] for shell in config_tuples].index((self.n, self.l))

            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                ae = AllElectron(chemical_symbols[self.Z], xcname=self.xc, gpernode=self.gpernode)
                ae.run()
            
            self.rmax = max(ae.r)
            wave = interp1d(ae.r, ae.u_j[subshell_index], kind='cubic', fill_value='extrapolate', bounds_error=False)
            # return ae.ETotal * units.Hartree, (ae.r, ae.u_j[subshell_index])
            return ae.ETotal * units.Hartree, wave
        else:
            from abtem.ionization.dirac import orbital
            orb = orbital(Z=self.Z,n=self.n,l=self.l,lprimes=self.lprimes,epsilon=self.epsilon)
            wave = orb.get_bound_wave()
            ETotal = orb.energy
            self.rmax = max(orb.r)
            return ETotal, wave 

    @cached_method('_continuum_cache')
    def _calculate_continuum(self):
        if self.dirac is False:
            _,vr = self.get_bound_potential()
            etot,_ = self.get_continuum_potential()
            def schroedinger_derivative(y, r, l, e, vr):
                (u, up) = y
                # note vr is effective potential multiplied by radius:
                return np.array([up, (l * (l + 1) / r ** 2 + 2 * vr(r) / r - e) * u])

            continuum_waves = {}
            for lprime in self.lprimes:
                e=self.epsilon/units.Rydberg
                rc = min(1/np.sqrt(e),50)
                r0 = max(10/np.sqrt(e),5*lprime*(lprime+1),rc,200)
                rcore = np.geomspace(1e-7,rc,10000)
                step_size = 1/2/e**0.25
                num_step = int(np.ceil((r0-rc)/step_size))
                rvac = np.linspace(rc,r0,num_step)
                r = np.unique(np.concatenate((rcore,rvac)))
                # note: epsilon in the atomic unit for the ODE
                ur = integrate.odeint(schroedinger_derivative, [0.0, 1.], r, args=(lprime, e, vr))[:,0]

                # sqrt_k = (2 * self.epsilon / units.Hartree * (
                #         1 + units.alpha ** 2 * self.epsilon / units.Hartree / 2)) ** .25
                sqrt_k = (self.epsilon/units.Rydberg) ** .25

                from scipy.interpolate import InterpolatedUnivariateSpline
                ur_i = InterpolatedUnivariateSpline(r, ur, k=4)
                cr_pts = ur_i.derivative().roots()
                cr_vals = ur_i(cr_pts)
                rf = cr_pts[-1]
                A = 1 - 1/2/e/rf*(1 - 5/3/e/rf - lprime*(lprime+1)/2/rf)
                B = abs(cr_vals[-1])
                ur = ur *A / B / sqrt_k / np.sqrt(np.pi)

                # note: sqrt_k = (epsilon in atomic unit)**0.25, see Manson 1972

                # continuum_waves[lprime] = (r, ur)  
                continuum_waves[lprime] = interp1d(r, ur, kind='cubic', fill_value='extrapolate', bounds_error=False)
            return etot, continuum_waves
        else:
            from abtem.ionization.dirac import orbital
            orb = orbital(Z=self.Z,n=0,l=self.l,lprimes=self.lprimes,epsilon=self.epsilon)
            continuum_waves = orb.get_continuum_waves()
            ETotal = self.epsilon
            return ETotal, continuum_waves
    
    @cached_method('_continuum_potential_cache')
    def get_continuum_potential(self):
        from gpaw.atom.all_electron import AllElectron

        check_valid_quantum_number(self.Z, self.n, self.l)
        config_tuples = config_str_to_config_tuples(load_electronic_configurations()[chemical_symbols[self.Z]])
        subshell_index = [shell[:2] for shell in config_tuples].index((self.n, self.l))

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            ae = AllElectron(chemical_symbols[self.Z], xcname=self.xc)
            ae.f_j[subshell_index] -= 1.
            ae.run()

        vr = interp1d(ae.r, ae.vr, fill_value='extrapolate', bounds_error=False)
        return ae.ETotal * units.Hartree, vr

    @cached_method('_bound_potential_cache')
    def get_bound_potential(self):
        from gpaw.atom.all_electron import AllElectron

        check_valid_quantum_number(self.Z, self.n, self.l)
        config_tuples = config_str_to_config_tuples(load_electronic_configurations()[chemical_symbols[self.Z]])
        subshell_index = [shell[:2] for shell in config_tuples].index((self.n, self.l))

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            ae = AllElectron(chemical_symbols[self.Z], xcname=self.xc)
            ae.run()

        vr = interp1d(ae.r, ae.vr, fill_value='extrapolate', bounds_error=False)
        return ae.ETotal * units.Hartree, vr

    def get_bound_wave(self):
        return self._calculate_bound()[1]

    def get_continuum_waves(self):
        return self._calculate_continuum()[1]

    def get_transition_quantum_numbers(self):
        transitions = []
        for ml in np.arange(-self.l, self.l + 1):
            for new_l in self.lprimes:
                for new_ml in np.arange(-new_l, new_l + 1):
                    # not sure why enforce the dipole selection rule here? As large collection angle will have selection-forbiden selection
                    # if not abs(new_l - self.l) == 1:
                    #     continue

                    # if not (abs(ml - new_ml) < 2):
                    #     continue
                    
                    transitions.append([(self.l, ml), (new_l, new_ml)])
        return transitions

    def as_arrays(self):

        _, bound_wave = self._calculate_bound()
        _, continuum_waves = self._calculate_continuum()

        bound_state = self.get_transition_quantum_numbers()[0][0]
        continuum_states = [state[1] for state in self.get_transition_quantum_numbers()]
        _, continuum_waves = self._calculate_continuum()

        arrays = SubshellTransitionsArrays(Z=self.Z,
                                           bound_wave=bound_wave,
                                           continuum_waves=continuum_waves,
                                           bound_state=bound_state,
                                           continuum_states=continuum_states,
                                           energy_loss=self.energy_loss,
                                           )

        return arrays

    def get_transition_potentials(self,
                                  extent: Union[float, Sequence[float]] = None,
                                  gpts: Union[float, Sequence[float]] = None,
                                  sampling: Union[float, Sequence[float]] = None,
                                  energy: float = None,
                                  pbar=True):

        transitions = []
        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(self), desc='Transitions', disable=(not pbar))
            _, bound_wave = self._calculate_bound()
            _, continuum_waves = self._calculate_continuum()
        energy_loss = self.energy_loss

        # bound_wave = interp1d(*bound_wave, kind='cubic', fill_value='extrapolate', bounds_error=False)

        for bound_state, continuum_state in self.get_transition_quantum_numbers():
            continuum_wave = continuum_waves[continuum_state[0]]

            # continuum_wave = interp1d(*continuum_wave, kind='cubic', fill_value='extrapolate', bounds_error=False)

            transition = ProjectedAtomicTransition(Z=self.Z,
                                                   bound_wave=bound_wave,
                                                   continuum_wave=continuum_wave,
                                                   bound_state=bound_state,
                                                   continuum_state=continuum_state,
                                                   energy_loss=energy_loss,
                                                   extent=extent,
                                                   gpts=gpts,
                                                   sampling=sampling,
                                                   energy=energy
                                                   )
            transitions += [transition]
            pbar.update(1)

        pbar.refresh()
        pbar.close()
        return transitions

class SubshellTransitionsArrays:

    def __init__(self, Z, bound_wave, continuum_waves, bound_state, continuum_states, energy_loss):
        self._Z = Z
        self._bound_wave = bound_wave
        self._continuum_waves = continuum_waves
        self._bound_state = bound_state
        self._continuum_states = continuum_states
        self._energy_loss = energy_loss

    @property
    def Z(self):
        return self._Z

    def get_transition_quantum_numbers(self):

        for continuum_state in self._continuum_states:
            yield self._bound_state, continuum_state

    def get_transition_potentials(self,
                                  extent: Union[float, Sequence[float]] = None,
                                  gpts: Union[float, Sequence[float]] = None,
                                  sampling: Union[float, Sequence[float]] = None,
                                  energy: float = None,
                                  pbar=True):
        transitions = []

        bound_wave = self._bound_wave
        continuum_waves = self._continuum_waves
        energy_loss = self._energy_loss

        bound_state = self._bound_state
        # bound_wave = interp1d(*bound_wave, kind='cubic', fill_value='extrapolate', bounds_error=False)

        for continuum_state in self._continuum_states:
            continuum_wave = continuum_waves[continuum_state[0]]

            # continuum_wave = interp1d(*continuum_wave, kind='cubic', fill_value='extrapolate', bounds_error=False)

            transition = ProjectedAtomicTransition(Z=self._Z,
                                                   bound_wave=bound_wave,
                                                   continuum_wave=continuum_wave,
                                                   bound_state=bound_state,
                                                   continuum_state=continuum_state,
                                                   energy_loss=energy_loss,
                                                   extent=extent,
                                                   gpts=gpts,
                                                   sampling=sampling,
                                                   energy=energy
                                                   )
            transitions += [transition]

        return transitions

    def write(self, f):
        d = {'Z': self._Z, 'bound_wave': self._bound_wave, 'continuum_waves': self._continuum_waves,
             'bound_state': self._bound_state, 'continuum_states': self._continuum_states,
             'energy_loss': self._energy_loss}

        np.savez(f, **d)

    @classmethod
    def read(cls, f):

        data = np.load(f, allow_pickle=True)
        Z = data['Z']
        bound_wave = data['bound_wave']
        continuum_waves = data['continuum_waves'][()]
        bound_state = tuple(data['bound_state'])
        continuum_states = [tuple(state) for state in data['continuum_states']]
        energy_loss = data['energy_loss']

        #print(bound_state)
        #print(continuum_states)
        #print(Z)
        #print(bound_wave)
        #print(continuum_waves)
        #print(energy_loss)


        return cls(Z=Z, bound_wave=bound_wave, continuum_waves=continuum_waves, bound_state=bound_state,
                   continuum_states=continuum_states, energy_loss=energy_loss)


class AbstractProjectedAtomicTransition(HasAcceleratorMixin, HasGridMixin):

    def __init__(self, Z, extent, gpts, sampling, energy):
        self._Z = Z
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)


class ProjectedAtomicTransition(AbstractProjectedAtomicTransition):

    def __init__(self,
                 Z: int,
                 bound_wave: callable,
                 continuum_wave: callable,
                 bound_state: Tuple[int, int],
                 continuum_state: Tuple[int, int],
                 energy_loss: float = 1.,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None):

        self._bound_wave = bound_wave
        self._continuum_wave = continuum_wave
        self._bound_state = bound_state
        self._continuum_state = continuum_state
        self._energy_loss = energy_loss

        self._cache = Cache(1)

        super().__init__(Z, extent, gpts, sampling, energy)

    def __str__(self):
        return (f'{self._bound_state} -> {self._continuum_state}')

    @property
    def energy_loss(self):
        return self._energy_loss

    @property
    def momentum_transfer(self):
        return self.k0 - self.kn
    
    @property
    def kn(self):
        return 1 / energy2wavelength(self.energy - self.energy_loss)
    
    @property
    def k0(self):
        return  1 / energy2wavelength(self.energy)

    def _fourier_translation_operator(self, positions):
        return fourier_translation_operator(positions, self.gpts)

    def build(self, positions=None):
        if positions is None:
            positions = np.zeros((1, 2), dtype=np.float32)
        else:
            positions = np.array(positions, dtype=np.float32)

        if len(positions.shape) == 1:
            positions = np.expand_dims(positions, axis=0)

        positions /= self.sampling
        potential = np.fft.ifft2(self._evaluate_potential() * self._fourier_translation_operator(positions))
        return potential

    def calculate_total_intensity(self):
        return (np.abs(self.build()) ** 2).sum()

    def overlap_integral(self, k, lprimeprime):
        rmax = self.rmax
        # rmax = max(self._bound_wave[1])
        grid = 2 * np.pi * k * units.Bohr
        # r = np.geomspace(1e-7, rmax, 1000000)
        r = np.linspace(0, rmax, 10000)

        values = (self._bound_wave(r) *
                  spherical_jn(lprimeprime, grid[:, None] * r[None]) *
                  self._continuum_wave(r))
        return np.trapz(values, r, axis=1)

    @cached_method('_cache')
    def _evaluate_potential(self):
        from sympy.physics.wigner import wigner_3j

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        potential = np.zeros(self.gpts, dtype=np.complex64)

        kx, ky = spatial_frequencies(self.gpts, self.sampling)
        kt, phi = polar_coordinates(kx, ky)
        theta = np.arcsin(kt/self.kn)
        k = np.sqrt(self.k0**2+self.kn**2-2*self.k0*self.kn*np.cos(theta))
        # kz = self.momentum_transfer
        # k = np.sqrt(kt ** 2 + kz ** 2)
        # theta = np.arctan(kt / kz)

        radial_grid = np.arange(0, np.nanmax(k) * 1.05, 1 / max(self.extent))

        l, ml = self._bound_state
        lprime, mlprime = self._continuum_state
        # lprimeprime only valid from |l-lprime| to l+lprime in step of 2, see Manson 1972 
        for lprimeprime in range(abs(l - lprime), np.abs(l + lprime) + 1, 2):
            prefactor1 = (np.sqrt(4 * np.pi) * ((-1.j) ** lprimeprime) *
                          np.sqrt((2 * lprime + 1) * (2 * lprimeprime + 1) * (2 * l + 1)))
            jk = None

            for mlprimeprime in range(-lprimeprime, lprimeprime + 1):

                if ml - mlprime - mlprimeprime != 0:  # Wigner3j selection rule
                    continue

                # Evaluate Eq. (14) from Dwyer Ultramicroscopy 104 (2005) 141-151
                prefactor2 = ((-1.0) ** (mlprime + mlprimeprime)
                              * float(wigner_3j(lprime, lprimeprime, l, 0, 0, 0))
                              * float(wigner_3j(lprime, lprimeprime, l, -mlprime, -mlprimeprime, ml)))

                if np.abs(prefactor2) < 1e-12:
                    continue

                if jk is None:
                    jk = interp1d(radial_grid, self.overlap_integral(radial_grid, lprimeprime))(k)

                Ylm = sph_harm(mlprimeprime, lprimeprime, phi, np.pi-theta)
                potential += prefactor1 * prefactor2 * jk * Ylm

        potential *= np.prod(self.gpts) / np.prod(self.extent)

        # Multiply by orbital filling
        # if orbital_filling_factor:
        potential *= np.sqrt(4 * l + 2)

        # Apply constants:
        # sqrt(Rdyberg) to convert to 1/sqrt(eV) units
        # 1 / (2 pi**2 a0 kn) as as per paper
        # Relativistic mass correction to go from a0 to relativistically corrected a0*
        # divide by q**2

        potential *= relativistic_mass_correction(self.energy) / (
                2 * units.Bohr * np.pi ** 2 * np.sqrt(units.Rydberg) * self.kn * k ** 2
        )
        potential = np.nan_to_num(potential)
        return potential

    def measure(self):
        array = np.fft.fftshift(self.build())[0]
        calibrations = calibrations_from_grid(self.gpts, self.sampling, ['x', 'y'])
        abs2 = get_device_function(get_array_module(array), 'abs2')
        return Measurement(array, calibrations, name=str(self))

    def show(self, ax, **kwargs):
        # array = np.fft.fftshift(self.build())[0]
        # calibrations = calibrations_from_grid(self.gpts, self.sampling, ['x', 'y'])
        # abs2 = get_device_function(get_array_module(array), 'abs2')
        self.measure().show(ax=ax, **kwargs)
        # Measurement(abs2(array), calibrations, name=str(self)).show(**kwargs)


class TransitionPotential(HasAcceleratorMixin, HasGridMixin):

    def __init__(self,
                 transitions,
                 atoms=None,
                 slice_thickness=None,
                 gpts: Union[int, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None,
                 min_contrast=.95):

        if isinstance(transitions, (SubshellTransitions, SubshellTransitionsArrays)):
            transitions = [transitions]

        self._slice_thickness = slice_thickness

        self._grid = Grid(gpts=gpts, sampling=sampling)

        self.atoms = atoms
        self._transitions = transitions

        self._accelerator = Accelerator(energy=energy)

        self._sliced_atoms = SlicedAtoms(atoms, slice_thicknesses=self._slice_thickness)

        self._potentials_cache = Cache(1)

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        self._atoms = atoms

        if atoms is not None:
            self.extent = np.diag(atoms.cell)[:2]
            self._sliced_atoms = SlicedAtoms(atoms, slice_thicknesses=self._slice_thickness)
        else:
            self._sliced_atoms = None

    @property
    def num_edges(self):
        return len(self._transitions)

    @property
    def num_slices(self):
        return self._sliced_atoms.num_slices

    @cached_method('_potentials_cache')
    def _calculate_potentials(self, transitions_idx):
        transitions = self._transitions[transitions_idx]
        return transitions.get_transition_potentials(extent=self.extent, gpts=self.gpts, energy=self.energy, pbar=False)

    def _generate_slice_transition_potentials(self, slice_idx, transitions_idx):
        transitions = self._transitions[transitions_idx]
        Z = transitions.Z

        atoms_slice = self._sliced_atoms.get_subsliced_atoms(slice_idx, atomic_number=Z).atoms

        for transition in self._calculate_potentials(transitions_idx):
            for atom in atoms_slice:
                t = np.asarray(transition.build(atom.position[:2]))
                yield t

    def show(self, transitions_idx=0):
        intensity = None

        if self._sliced_atoms.slice_thicknesses is None:
            none_slice_thickess = True
            self._sliced_atoms.slice_thicknesses = self._sliced_atoms.atoms.cell[2, 2]
        else:
            none_slice_thickess = False

        for slice_idx in range(self.num_slices):
            for t in self._generate_slice_transition_potentials(slice_idx, transitions_idx):
                if intensity is None:
                    intensity = np.abs(t) ** 2
                else:
                    intensity += np.abs(t) ** 2

        if none_slice_thickess:
            self._sliced_atoms.slice_thicknesses = None

        calibrations = calibrations_from_grid(self.gpts, self.sampling, ['x', 'y'])
        Measurement(intensity[0], calibrations, name=str(self)).show()

class GeneralOsilationStrength:
    def __init__(self,
                 Z,
                 n,
                 l,
                 order,
                 energy: float = 3e5,
                 qmin: float = 0.1,
                 qmax: float = 50,
                 qgpts: int = 256,
                 collection_angle: float = 1,
                 ionization_energy: float = None,
                 epsilons: list = np.geomspace(0.01,1000,128),
                 ):

        self.Z = Z
        self.n = n
        self.l = l
        self.order = order
        self.transitions = SubshellTransitions(Z = Z, n = n, l = l, epsilon = 1, order = order, dirac = True)
        self.lprimes = self.transitions.lprimes
        self.subshell_occupancy=self.transitions.subshell_occupancy
        self.energy = energy
        self.qmax = qmax
        self.qmin = qmin
        self.qgpts = qgpts
        self.collection_angle = collection_angle
        self._bound_wave = self.transitions.get_bound_wave()
        self.epsilons = epsilons
        if ionization_energy is not None:
            self.ionization_energy =  ionization_energy
        else:
            self.ionization_energy = self.transitions.ionization_energy
        self.energy_losses = self.epsilons + self.ionization_energy

        self._gos_cache = Cache(1)
        # self._cache = Cache(1)

    # @cached_method('_dynamic_form_factor_cache')
    def dynamic_form_factor(self):
        s2_list = []
        l = self.l
        for lprime in self.lprimes:
            s2 = np.zeros(self.qsampling.shape, dtype=np.float64)
            # lprimeprime only valid from |l-lprime| to l+lprime in step of 2, see Manson 1972 
            for lprimeprime in range(abs(l - lprime), np.abs(l + lprime) + 1, 2):
                wigners = float(wigner_3j(lprime, lprimeprime, l, 0, 0, 0))**2
                jk = self.overlap_integral(self.qsampling, lprime, lprimeprime)
                s2 += self.subshell_occupancy*(2*lprime+1)*(2*lprimeprime+1)*wigners*jk**2
            s2_list.append(s2)
        return np.array(s2_list)

    @cached_method('_gos_cache')
    def get_gos(self,sumed=True,unit='eV'):
        gos_matrix = []
        for epsilon in self.epsilons:
            self.transitions._epsilon = epsilon
            self.energy_loss = epsilon + self.ionization_energy
            self.transitions._continuum_cache.clear()
            self._continuum_waves = self.transitions.get_continuum_waves()

            gos_list = self.energy_loss/units.Rydberg/(self.qsampling*units.Bohr)**2*self.dynamic_form_factor()
            # to remove the divergence behaviour at q->0 when delta l = 0
            mask = self.qsampling > 0.1
            qsampling_new = np.insert(self.qsampling[mask], 0, 0, axis=0)
            gos_new  = np.insert(gos_list[self.l][mask], 0, 0, axis=0)
            gos_list[self.l] = interp1d(qsampling_new,gos_new,kind='cubic')(self.qsampling)

            if unit =='eV':
                gos_list = np.array(gos_list)/units.Ry
            if sumed == True:
                gos_list = np.sum(gos_list,axis=0)
            gos_matrix.append(gos_list)
        return np.array(gos_matrix)

    def scs_dE_dOmega(self):
        scs_list=[]
        for epsilon in self.epsilons:
            self.transitions._epsilon = epsilon
            self.energy_loss = epsilon + self.ionization_energy
            self.transitions._continuum_cache.clear()
            self._continuum_waves = self.transitions.get_continuum_waves()
            scs = 4*relativistic_mass_correction(self.energy)**2/(units.Bohr**2*self.qsampling**4)*self.kn/self.k0*np.sum(self.dynamic_form_factor(),axis=0)/units.Rydberg
            scs_list.append(scs)
        return scs_list
    
    def scs_dE_dlnQ(self):
        scs = 4*np.pi*relativistic_mass_correction(self.energy)**2/self.k0**2*self.get_gos()/self.energy_losses.reshape(-1,1)*units.Rydberg
        return scs

    def scs_integrate_q(self, voltage, collection_angle):
        self.energy = voltage
        self.collection_angle = collection_angle
        scs_list = []
        for idx,loss in enumerate(self.energy_losses):
            self.energy_loss = loss
            # theta = np.arcsin(self.k0/self.kn*np.tan(self.collection_angle/1000))
            Qmax = (self.k0**2+self.kn**2-2*self.kn*self.k0*np.cos(self.collection_angle/1000))*units.Bohr**2
            Qmin = ((self.k0-self.kn)*units.Bohr)**2
            func = interp1d(np.log(self.Q), self.scs_dE_dlnQ()[idx,:])
            value,err = integrate_adaptive(func,[np.log(Qmin),np.log(Qmax)],eps_rel=1e-10)
            scs_list.append(value)
        return scs_list
    
    def characteristic_angle(self, unit = 'mrad', relativisitc = True):
        if relativisitc:
            print('relativisitc corrected angle')
            thetaE = self.energy_loss/relativistic_mass_correction(self.energy)/units._me/relativistic_velocity(self.energy)**2/units.J
        else:
            print('classical angle theta=E/2E0')
            thetaE = self.energy_loss/2/self.energy
        if unit == 'mrad':
            print('unit in mrad')
            return thetaE*1e3
        if unit == 'A-1':
            print('unit in A-1')
            return thetaE/energy2wavelength(self.energy)

    def overlap_integral(self, q, lprime, lprimeprime):
        grid = q * units.Bohr
        rmax = self.transitions.rmax

        func = lambda r:(self._bound_wave(r) *
                        spherical_jn(lprimeprime, grid[:, None] * r[None]) 
                        * self._continuum_waves[lprime](r))

        value,err = integrate_adaptive(func,[0,rmax],eps_rel=1e-10)
        return value

    @property
    def qsampling(self):
        return np.geomspace(self.qmin, self.qmax, num=self.qgpts)

    @property
    def Q(self):
        return (units.Bohr*self.qsampling)**2

    @property
    def theta(self):
        print('unit in mrad')
        return np.arccos((self.k0**2+self.kn**2-(self.qsampling)**2)/(2*self.k0*self.kn))*1e3

    @property
    def k0(self):
        return 2*np.pi /energy2wavelength(self.energy)
    
    @property
    def kn(self):
        return 2*np.pi /energy2wavelength(self.energy-self.energy_loss)
    
    @property
    def spin_orbital_occupancy(self):
        if self.l == 0:
            return 1
        else: 
            return np.array([2*(self.l-1/2)+1,2*(self.l+1/2)+1])/(4*self.l+2)
    @property
    def edge_name(self):
        edge = ["K","L","M","N","O","P","Q"][self.n]
        allow_index = [self.l*2, self.l*2+1].remove(0)
        return [edge+str(index) for index in allow_index]

class AtomicScatteringFactor:
    def __init__(self,
                 Z,
                 n,
                 l,
                 energy: float = 3e5,
                 smax: float =20,
                 spts: int = 256,
                 qpts: int = 32,
                 ionization_energy: float = None,
                 epsilon_list: list =  None,
                 epsilon_sampling: int = 32,
                 convergence: float = 0.001,
                 ):

        self.Z = Z
        self.n = n
        self.l = l
        self.energy = energy
        self.smax = smax
        self.spts = spts
        self.qpts = qpts
        self.convergence = convergence
        orb = orbital(Z=self.Z,n=self.n,l=self.l,lprimes=0,epsilon=0)
        self._bound_wave = orb.get_bound_wave()
        self.rmax = max(orb.r)
        
        if ionization_energy is not None:
            self.ionization_energy =  ionization_energy
        else:
            self.ionization_energy = np.abs(orb.energy)

        if epsilon_list is None:
            self.epsilon_list = np.geomspace(1e-3,self.energy-self.ionization_energy-1e3,epsilon_sampling)
        else:
            self.epsilon_list = epsilon_list
        self.energy_loss_list = self.epsilon_list + self.ionization_energy
        self._energy_loss = self.energy_loss_list[0]

    def sum_states(self,epsilon):
        order = 2*int(np.ceil((epsilon/units.Rydberg)**(1/3))) + 1
        min_new_l = max(self.l - order, 0)
        lprimes = np.arange(min_new_l, self.l + order + 1)
        orb = orbital(Z=self.Z,n=0,l=self.l,lprimes=lprimes,epsilon=epsilon)
        self._continuum_waves = orb.get_continuum_waves()
        s2_sum = np.zeros([self.spts,self.qpts], dtype=np.float64)
        for lprime in lprimes:
            s2 = np.zeros([self.spts,self.qpts], dtype=np.float64)
            # lprimeprime only valid from |l-lprime| to l+lprime in step of 2, see Manson 1972 
            for lprimeprime in range(abs(self.l - lprime), np.abs(self.l + lprime) + 1, 2):
                s2 += self.dynamic_form_factor(self.l,lprime,lprimeprime)
            s2_sum += s2
            contribution = (np.sum(s2,axis=1)/np.sum(s2_sum,axis=1))[0]
            log.info(f'epsilon = {epsilon:.2f} eV, lprime = {lprime}, contriubtion = {contribution:.3f}, kn ={self.kn}\n')
            if np.abs(contribution) < self.convergence: #convergence criteria
                break
        return s2_sum

    def dynamic_form_factor(self,l,lprime,lprimeprime):
        wigners = float(wigner_3j(lprime, lprimeprime, l, 0, 0, 0))**2
        jk0 = self.overlap_integral(self.q, lprime, lprimeprime)
        jks = self.overlap_integral(self.qs, lprime, lprimeprime)
        s2 = self.subshell_occupancy*(2*lprime+1)*(2*lprimeprime+1)*wigners*jk0*jks*np.polyval(legendre(lprimeprime),np.cos(self.alpha))
        # s2[s2<0]=0
        return s2

    def overlap_integral(self, q, lprime, lprimeprime):
        grid = q * units.Bohr
        func = lambda r:(self._bound_wave(r) *
                        spherical_jn(lprimeprime, grid[..., None] * r) 
                        * self._continuum_waves[lprime](r))
        value,err = integrate_adaptive(func,[0,self.rmax],eps_rel=1e-10)
        return value

    def test_integral_solid_angle(self,epsilon,kind):
        self._energy_loss = epsilon + self.ionization_energy
        I_sum = self.sum_states(epsilon)/self.q**2/self.qs**2
        I_sum_interp = interp1d(self.theta,I_sum,kind=kind)
        func_solid = lambda theta:(np.sin(theta)*2*np.pi*I_sum_interp(theta))
        I_intsolid,err = integrate_adaptive(func_solid,[0,np.pi],eps_rel=1e-10)
        fs = 4*relativistic_mass_correction(self.energy)**2/(units.Bohr**2)*I_intsolid*self.kn/units.Rydberg
        return fs, I_sum, I_sum_interp

    def integral_solid_angle(self,epsilon):
        self._energy_loss = epsilon + self.ionization_energy
        I_sum = self.sum_states(epsilon)/self.q**2/self.qs**2
        I_sum_interp = interp1d(self.theta,I_sum,kind='cubic')
        func_solid = lambda theta:(np.sin(theta)*2*np.pi*I_sum_interp(theta))
        I_intsolid,err = integrate_adaptive(func_solid,[0,np.pi],eps_rel=1e-10)
        fs = 4*relativistic_mass_correction(self.energy)**2/(units.Bohr**2)*I_intsolid*self.kn/units.Rydberg
        return fs

    def form_factor_dOmega(self,epsilon):
        self._energy_loss = epsilon + self.ionization_energy
        s2_sum = self.sum_states(epsilon)/self.q**2/self.qs**2
        before_integral = 4*relativistic_mass_correction(self.energy)**2/(units.Bohr**2)*s2_sum*self.kn/units.Rydberg
        func_solid = interp1d(self.theta,before_integral*np.sin(self.theta)*2*np.pi,kind='quadratic')
        after_integral,err = integrate_adaptive(func_solid,[0,np.pi],eps_rel=1e-10)
        return before_integral,after_integral

    def get_edx_scattering_form_factor(self):
        # fs_per_energy=Parallel(n_jobs=-2)(delayed(self.integral_solid_angle)(epsilon) for epsilon in self.epsilon_list)

        with Pool() as pool: #parallel computing
            fs_per_energy = np.array(pool.map(self.integral_solid_angle, self.epsilon_list))

        # fs_per_energy = []
        # for epsilon in self.epsilon_list:
        #     fs_per_energy.append(self.integral_solid_angle(epsilon))

        func_energy = interp1d(self.epsilon_list,fs_per_energy.T,kind='quadratic') 
        fs_ienery,err = integrate_adaptive(func_energy,[self.epsilon_list[0],self.epsilon_list[-1]],eps_rel=1e-10)

        # epsilon_list_new = np.append(self.epsilon_list,self.energy-self.ionization_energy)
        # fs_per_energy_new = np.append(fs_per_energy,np.zeros((1,self.spts)),axis=0)
        # func_energy = interp1d(epsilon_list_new,fs_per_energy_new.T,kind='quadratic') 
        # fs_ienery,err = integrate_adaptive(func_energy,[self.epsilon_list[0],self.energy-self.ionization_energy],eps_rel=1e-10)
        return fs_ienery,fs_per_energy


    def test_get_edx_scattering_form_factor(self):
        with Pool() as pool: #parallel computing
            fs_per_energy = np.array(pool.map(self.integral_solid_angle, self.epsilon_list))
        func_energy_lin = interp1d(self.epsilon_list,fs_per_energy.T,kind='slinear') 
        fs_ienery_lin,err = integrate_adaptive(func_energy_lin,[self.epsilon_list[0],self.epsilon_list[-1]],eps_rel=1e-10)
        func_energy_qua = interp1d(self.epsilon_list,fs_per_energy.T,kind='quadratic') 
        fs_ienery_qua,err = integrate_adaptive(func_energy_qua,[self.epsilon_list[0],self.epsilon_list[-1]],eps_rel=1e-10)
        func_energy_cub = interp1d(self.epsilon_list,fs_per_energy.T,kind='cubic') 
        fs_ienery_cub,err = integrate_adaptive(func_energy_cub,[self.epsilon_list[0],self.epsilon_list[-1]],eps_rel=1e-10)
        return fs_ienery_lin,fs_ienery_qua,fs_ienery_cub,fs_per_energy

    # def sum_states_parallel(self,epsilon):
    #     order = 2*int(np.ceil((epsilon/units.Rydberg)**(1/3))) + 1
    #     min_new_l = max(self.l - order, 0)
    #     lprimes = np.arange(min_new_l, self.l + order + 1)
    #     orb = orbital(Z=self.Z,n=0,l=self.l,lprimes=lprimes,epsilon=epsilon)
    #     self._continuum_waves = orb.get_continuum_waves()
    #     states = []
    #     for lprime in lprimes:
    #         # lprimeprime only valid from |l-lprime| to l+lprime in step of 2, see Manson 1972 
    #         for lprimeprime in range(abs(self.l - lprime), np.abs(self.l + lprime) + 1, 2):
    #             states.append((self.l,lprime,lprimeprime))
    #     s2_sum = Parallel(n_jobs=-2)(delayed(self.dynamic_form_factor)(l,lprime,lprimeprime) for l,lprime,lprimeprime in states)
    #     s2_sum = np.sum(s2_sum, axis=0)
    #     return s2_sum       
    #  
    # def states_and_waves(self,epsilon):
    #     order = 2*int(np.ceil((epsilon/units.Rydberg)**(1/3))) + 1
    #     min_new_l = max(self.l - order, 0)
    #     lprimes = np.arange(min_new_l, self.l + order + 1)
    #     orb = orbital(Z=self.Z,n=0,l=self.l,lprimes=lprimes,epsilon=epsilon)
    #     continuum_waves = orb.get_continuum_waves()
    #     states = []
    #     for lprime in lprimes:
    #         # lprimeprime only valid from |l-lprime| to l+lprime in step of 2, see Manson 1972 
    #         for lprimeprime in range(abs(self.l - lprime), np.abs(self.l + lprime) + 1, 2):
    #             states.append((self.l,lprime,lprimeprime))      
    #     return states, continuum_waves

    # def dynamic_form_factor_parallel(self,l,lprime,lprimeprime,epsilon,continuum_waves):
    #     self._energy_loss = epsilon + self.ionization_energy
    #     wigners = float(wigner_3j(lprime, lprimeprime, l, 0, 0, 0))**2
    #     jk0 = self.overlap_integral_parallel(self.q, lprime, lprimeprime,continuum_waves)
    #     jks = self.overlap_integral_parallel(self.qs, lprime, lprimeprime,continuum_waves)
    #     s2 = 2*(2*l+1)*(2*lprime+1)*(2*lprimeprime+1)*wigners*jk0*jks*np.polyval(legendre(lprimeprime),np.cos(self.alpha))
    #     s2[s2<0]=0
    #     return s2

    # def overlap_integral_parallel(self, q, lprime, lprimeprime,continuum_waves):
    #     grid = q * units.Bohr
    #     func = lambda r:(self._bound_wave(r) *
    #                     spherical_jn(lprimeprime, grid[..., None] * r) 
    #                     * continuum_waves[lprime](r))
    #     value,err = integrate_adaptive(func,[0,self.rmax],eps_rel=1e-10)
    #     return value

    # def integral_solid_angle_parallel(self,s2_orbitals,epsilon):
    #     self._energy_loss = epsilon + self.ionization_energy
    #     I_sum = np.sum(s2_orbitals,axis=0)/self.q**2/self.qs**2
    #     I_sum = interp1d(self.theta,I_sum,kind='slinear')
    #     func_solid = lambda theta:(np.sin(theta)*2*np.pi*I_sum(theta))
    #     I_intsolid,err = integrate_adaptive(func_solid,[0,np.pi],eps_rel=1e-10)
    #     fs = 4*relativistic_mass_correction(self.energy)**2/(units.Bohr**2)*I_intsolid*self.kn/units.Rydberg
    #     return fs    
    
    # def get_edx_scattering_form_factor_parallel(self):
    #     from joblib import Parallel, delayed
    #     states_waves_per_energy = Parallel(n_jobs=-2)(delayed(self.states_and_waves)(epsilon) for epsilon in self.epsilon_list)
    #     linear_states_waves_per_energy = np.array(states_waves_per_energy).ravel()
    #     s2_per_energy = Parallel(n_jobs=-2)(delayed(self.dynamic_form_factor_parallel)(l,lprime,lprimeprime,continuum_waves,epsilon) for l,lprime,lprimeprime,continuum_waves,epsilon in zip(linear_states_waves_per_energy,self.epsilon_list))
    #     fs_per_energy = Parallel(n_jobs=-2)(delayed(self.integral_solid_angle_parallel)(s2_orbitals,epsilon) for s2_orbitals,epsilon in zip(s2_per_energy,self.epsilon_list))
    #     func_energy = interp1d(self.epsilon_list,np.array(fs_per_energy).T,kind='slinear') 
    #     fs_ienery,err = integrate_adaptive(func_energy,[self.epsilon_list[0],self.epsilon_list[-1]],eps_rel=1e-10)
    #     return fs_ienery,fs_per_energy

    @property
    def energy_loss(self):
        if self._energy_loss > self.energy:
            raise ValueError("The energy loss should be less than the incident beam energy")
        return self._energy_loss

    @property
    def subshell_occupancy(self):
        config_tuples = config_str_to_config_tuples(load_electronic_configurations()[chemical_symbols[self.Z]])
        subshell_index = [shell[:2] for shell in config_tuples].index((self.n, self.l))
        subshell_occupancy = config_tuples[subshell_index][-1]
        return subshell_occupancy

    @property
    def s(self):
        sampling = np.geomspace(1e-5,self.smax,num=self.spts-1)
        return np.insert(sampling,0,0,axis=0)

    @property
    def theta(self):
        sampling = np.geomspace(1e-5,np.pi,num=self.qpts-1)
        return np.insert(sampling,0,0,axis=0)

    @property
    def qs(self):
        return np.sqrt(np.tile(self.q**2,[self.spts,1])+np.tile((4*np.pi*self.s)**2,[self.qpts,1]).T-2*np.outer((4*np.pi*self.s),self.q*np.cos(self.beta)))
    
    @property
    def q(self):
        return np.sqrt(self.k0**2+self.kn**2-2*self.k0*self.kn*np.cos(self.theta))

    @property
    def qz(self):
        return self.k0-self.kn*np.cos(self.theta)
    
    @property
    def qt(self):
        return self.kn*np.sin(self.theta)

    @property
    # the angle between q and s
    def beta(self):
        with np.errstate(divide='ignore'):
            return np.pi-np.arctan(self.qz/self.qt)
    
    @property
    # the angle between q and qs 
    def alpha(self):
        return np.arcsin(np.sin(self.beta)/self.qs*4*np.pi*np.tile(self.s,[self.qpts,1]).T)
        
    @property
    def k0(self):
        return 2*np.pi /energy2wavelength(self.energy)
    
    @property
    def kn(self):
        return 2*np.pi /energy2wavelength(self.energy-self.energy_loss)

# for evulate cross-section via gos outside of class
def scs_integrate_q(voltage,gos,energy_loss,qsampling,collection_angle):
    k0 = 2*np.pi /energy2wavelength(voltage)
    Q = (qsampling*units.Bohr)**2
    scs_dE_dlnQ = 4*np.pi*relativistic_mass_correction(voltage)**2/k0**2*gos/energy_loss.reshape(-1,1)*units.Rydberg
    scs_list = []
    for idx,loss in enumerate(energy_loss):
        kn = 2*np.pi /energy2wavelength(voltage-loss)
        Qmax = (k0**2+kn**2-2*kn*k0*np.cos(collection_angle/1000))*units.Bohr**2
        Qmin = ((k0-kn)*units.Bohr)**2
        func = interp1d(np.log(Q), scs_dE_dlnQ[idx,:])
        value,err = integrate_adaptive(func,[np.log(Qmin),np.log(Qmax)],eps_rel=1e-10)
        scs_list.append(value)
    return np.array(scs_list)


