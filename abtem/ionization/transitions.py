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
                 transitions,
                 energy: float = 3e5,
                 qmin: float = 0.1,
                 qmax: float = 20,
                 qgpts: int = 1024,
                 ionization_energy: float = None,
                 collection_angle: float = 1,
                 ):

        self.Z = transitions.Z
        self._bound_wave = transitions.get_bound_wave()
        self._continuum_waves = transitions.get_continuum_waves()
        self.l = transitions.l
        self.lprimes = transitions.lprimes
        self.subshell_occupancy=transitions.subshell_occupancy
        self.rmax = transitions.rmax
        
        if ionization_energy is not None:
            self.energy_loss = transitions.epsilon + ionization_energy
        else:
            self.energy_loss = transitions.energy_loss

        self.energy = energy
        self.qmax = qmax
        self.qmin = qmin
        self.qgpts = qgpts
        self.collection_angle = collection_angle
        self._dynamic_form_factor_cache = Cache(1)
        # self._cache = Cache(1)

    @cached_method('_dynamic_form_factor_cache')
    def dynamic_form_factor(self):
        from sympy.physics.wigner import wigner_3j
        s2_list = []
        l = self.l
        for lprime in self.lprimes:
            s2 = np.zeros(self.qsampling.shape, dtype=np.float64)
            prefactor0 = 2*lprime+1
            # lprimeprime only valid from |l-lprime| to l+lprime in step of 2, see Manson 1972 
            for lprimeprime in range(abs(l - lprime), np.abs(l + lprime) + 1, 2):
                prefactor1 = 2*lprimeprime+1
                prefactor2 = float(wigner_3j(lprime, lprimeprime, l, 0, 0, 0))**2
                jk = self.overlap_integral(self.ksampling, lprime, lprimeprime)
                s2 += self.subshell_occupancy*prefactor0*prefactor1*prefactor2*jk**2
            s2_list.append(s2)
        return s2_list

    def get_gos(self):
        gos_list=[]
        for lprime in self.lprimes:
            gos = self.energy_loss/units.Rydberg/(self.qsampling*units.Bohr)**2*self.dynamic_form_factor()[lprime]
            gos_list.append(gos)
        # to remove the divergence behaviour at q->0 when delta l = 0
        # if self._lprime == self._l:
        #     idx = self.qsampling > 1
        #     qsampling_new = np.insert(self.qsampling[idx], 0, 0, axis=0) 
        #     gos_new  = np.insert(gos[idx], 0, 0, axis=0) 
        #     gos = interp1d(qsampling_new,gos_new)(self.qsampling)
        return gos_list
    
    def double_diff_cross_section_dE_dOmega(self):
        scs_list=[]
        for lprime in self.lprimes:
            scs = 4*relativistic_mass_correction(self.energy)**2/(units.Bohr**2*self.qsampling**4)*self.kn/self.k0*self.dynamic_form_factor()[lprime]
            scs_list.append(scs)
        return scs_list
    
    def double_diff_cross_section_dE_dlnQ(self):
        scs_list=[]
        for lprime in self.lprimes:
            scs = 4*np.pi*relativistic_mass_correction(self.energy)**2/self.Q/self.k0**2*self.dynamic_form_factor()[lprime]
            scs_list.append(scs)
        return scs_list

    def get_diff_cross_section(self):
        scs_list=[]
        Qmax = (self.k0**2+self.kn**2-2*self.kn*self.k0*np.cos(self.collection_angle/1000))*units.Bohr**2
        ln_Qmax = np.log(Qmax)
        for lprime in self.lprimes:
            func = interp1d(np.log(self.Q), self.double_diff_cross_section_dE_dlnQ()[lprime])
            value,err = integrate_adaptive(func,[0,ln_Qmax],eps_rel=1e-10)
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

    def overlap_integral(self, k, lprime, lprimeprime):
        grid = 2 * np.pi * k * units.Bohr
        rmax = self.rmax

        func = lambda r:(self._bound_wave(r) *
                        spherical_jn(lprimeprime, grid[:, None] * r[None]) 
                        * self._continuum_waves[lprime](r))

        value,err = integrate_adaptive(func,[0,rmax],eps_rel=1e-10)
        return value

    @property
    def qsampling(self):
        return np.geomspace(self.qmin, self.qmax, num=self.qgpts)
    
    @property
    def ksampling(self):
        return np.geomspace(self.qmin, self.qmax, num=self.qgpts)/2/np.pi

    @property
    def Q(self):
        return (units.Bohr*self.qsampling)**2

    @property
    def angle(self):
        theta = np.arccos((self.k0**2+self.kn**2-(self.ksampling)**2)/(2*self.k0*self.kn))
        print('unit in mrad')
        return theta*1e3

    @property
    def k0(self):
        return 1 /energy2wavelength(self.energy)
    
    @property
    def kn(self):
        return 1 /energy2wavelength(self.energy-self.energy_loss)
