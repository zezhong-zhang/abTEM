"""Module to describe the effect of temperature on the atomic positions."""
from abc import abstractmethod, ABCMeta
from collections import Iterable
from copy import copy
from numbers import Number
from typing import Mapping, Union, Sequence, List

import dask
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols


class AbstractFrozenPhonons(metaclass=ABCMeta):
    """Abstract base class for frozen phonons objects."""

    @abstractmethod
    def __len__(self):
        pass

    @property
    @abstractmethod
    def cell(self):
        pass

    @abstractmethod
    def get_configurations(self, lazy: bool = True):
        pass

    def __getitem__(self, item):
        configurations = self.get_configurations(lazy=False)
        return configurations[item].jiggle_atoms()

    @abstractmethod
    def __copy__(self):
        pass

    def copy(self):
        """
        Make a copy.
        """
        return copy(self)


class FrozenPhonons(AbstractFrozenPhonons):
    """
    Frozen phonons object.

    Generates atomic configurations for thermal diffuse scattering.
    Randomly displaces the atomic positions of an ASE Atoms object to emulate thermal vibrations.

    Parameters
    ----------
    atoms: ASE Atoms object
        Atoms with the average atomic configuration.
    num_configs: int
        Number of frozen phonon configurations.
    sigmas: float or dict or list
        If float, the standard deviation of the displacements is assumed to be identical for all atoms.
        If dict, a displacement standard deviation should be provided for each species. The atomic species can be
        specified as atomic number or symbol.
        If list or array, a displacement standard deviation should be provided for each atom.
    directions: str
        The displacement directions of the atoms as a string; for example 'xy' for displacement in the x- and
        y-direction.
    seed: int
        Seed for random number generator.
    """

    def __init__(self,
                 atoms: Atoms,
                 num_configs: int,
                 sigmas: Union[float, Mapping[Union[str, int], float], Sequence[float]],
                 directions: str = 'xyz',
                 seed: int = None):

        self._unique_numbers = np.unique(atoms.numbers)
        unique_symbols = [chemical_symbols[number] for number in self._unique_numbers]

        if isinstance(sigmas, Number):
            new_sigmas = {}
            for symbol in unique_symbols:
                new_sigmas[symbol] = sigmas

            sigmas = new_sigmas

        elif isinstance(sigmas, dict):
            if not all([symbol in unique_symbols for symbol in sigmas.keys()]):
                raise RuntimeError('Displacement standard deviation must be provided for all atomic species.')

        elif isinstance(sigmas, Iterable):
            sigmas = np.array(sigmas, dtype=np.float32)
            if len(sigmas) != len(atoms):
                raise RuntimeError('Displacement standard deviation must be provided for all atoms.')
        else:
            raise ValueError()

        self._sigmas = sigmas
        self._directions = directions
        self._atoms = atoms
        self._num_configs = num_configs
        self._seed = seed

    @property
    def num_configs(self) -> int:
        return self._num_configs

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def sigmas(self) -> Union[Mapping[Union[str, int], float], np.ndarray]:
        return self._sigmas

    @property
    def cell(self) -> np.ndarray:
        return self._atoms.cell

    @property
    def atoms(self) -> Atoms:
        return self._atoms

    @property
    def directions(self) -> str:
        return self._directions

    def __len__(self) -> int:
        return self._num_configs

    @property
    def axes(self) -> List[int]:
        axes = []
        for direction in list(set(self._directions.lower())):
            if direction == 'x':
                axes += [0]
            elif direction == 'y':
                axes += [1]
            elif direction == 'z':
                axes += [2]
            else:
                raise RuntimeError('Directions must be "x", "y" or "z" not {}.')
        return axes

    def get_configurations(self, lazy: bool = False) -> List[Atoms]:
        if self.seed:
            np.random.seed(self.seed)

        def load_atoms():
            return self.atoms

        def jiggle_atoms(atoms, sigmas, directions):

            if isinstance(sigmas, Mapping):
                temp = np.zeros(len(atoms.numbers), dtype=np.float32)
                for unique in np.unique(atoms.numbers):
                    temp[atoms.numbers == unique] = np.float32(sigmas[chemical_symbols[unique]])
                sigmas = temp
            elif not isinstance(sigmas, np.ndarray):
                raise RuntimeError()

            atoms = atoms.copy()

            for direction in directions:
                atoms.positions[:, direction] += sigmas * np.random.randn(len(atoms))

            atoms.wrap()

            return atoms

        configurations = []
        for i in range(self.num_configs):
            if lazy:
                atoms = dask.delayed(load_atoms)()
                configurations.append(dask.delayed(jiggle_atoms)(atoms, self.sigmas, self.axes))
            else:
                configurations.append(jiggle_atoms(self.atoms, self.sigmas, self.axes))

        return configurations

    def __copy__(self) -> 'FrozenPhonons':
        return self.__class__(atoms=self.atoms.copy(), num_configs=len(self), sigmas=copy(self.sigmas),
                              seed=self.seed, directions=self.directions)


class MDFrozenPhonons(AbstractFrozenPhonons):
    """
    Molecular dynamics frozen phonons object.

    Parameters
    ----------
    trajectory: List of ASE Atoms objects
        Sequence of Atoms objects representing a thermal distribution of atomic configurations.
    """

    def __init__(self, trajectory: Sequence[Atoms]):
        self._trajectory = trajectory

    def __len__(self) -> int:
        return len(self._trajectory)

    @property
    def atoms(self) -> Atoms:
        return self[0]

    @property
    def cell(self) -> np.ndarray:
        return self[0].cell

    def __getitem__(self, item) -> Atoms:
        return self._trajectory[0]

    def standard_deviations(self) -> np.ndarray:
        mean_positions = np.mean([atoms.positions for atoms in self], axis=0)
        squared_deviations = [(atoms.positions - mean_positions) ** 2 for atoms in self]
        return np.sqrt(np.sum(squared_deviations, axis=0) / (len(self) - 1))

    def get_configurations(self, lazy: bool = True):
        return self._trajectory

    # def generate_atoms(self):
    #    for i in range(len(self)):
    #        yield self._trajectory[i]

    def __copy__(self):
        return self.__class__(trajectory=[atoms.copy() for atoms in self._trajectory])
