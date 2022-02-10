from pfac import fac 
from abtem.ionization.utils import config_str_to_config_tuples,load_electronic_configurations
import re
import numpy as np
from scipy.interpolate import interp1d
from ase import units

class orbital:
    """
    A class for storing the results of a fac atomic structure calculation,

    Code originally from py_multislice by Hamish Brown, adpated for abTEM by Zezhong Zhang. 
    """
    def __init__(self, Z: int, n: int, l: int, lprimes, epsilon):
        self.Z = Z
        self.n = n
        self.l = l
        self.lprimes = lprimes
        if isinstance(epsilon, np.ndarray):
            self.epsilon = epsilon
        else:
            self.epsilon = np.asarray([epsilon]).flatten()

    @property
    def config(self,ionised=False):
        config_tuples = config_str_to_config_tuples(load_electronic_configurations()[fac.ATOMICSYMBOL[self.Z]])
        if ionised is False:
            config = ' '.join(["".join(str(n)+["s","p","d","f"][l]+str(f)) for n,l,f in config_tuples]) 
        else:
            for n,l,f in config_tuples:
                if n == self.n and l == self.l:
                    config.append(str(n)+["s","p","d","f","g","h","i"][l]+str(f-1))
                else:
                    config.append(str(n)+["s","p","d","f","g","h","i"][l]+str(f))
        return config

    @property
    def configstring(self):
        if self.n == 0:
            configstring = fac.ATOMICSYMBOL[self.Z] + "_ex"
        else:
            configstring = fac.ATOMICSYMBOL[self.Z] + "_bound"
        return configstring

    # @property 
    # def title(self):
    #     if self.n > 0:
    #         # Bound wave function case
    #         angmom = ["s","p","d","f","g","h","i"][self.l]
    #         # Title in the format "Ag 1s", "O 2s" etc..
    #         return "{0} {1}{2}".format(fac.ATOMICSYMBOL[self.Z], self.n, angmom)
    #     else:
    #         # Continuum wave function case
    #         # Title in the format "Ag e = 10 eV l'=2" etc..
    #         return "{0} e = {1} l' = {2}".format(fac.ATOMICSYMBOL[self.Z], self.epsilon, self.lprimes)

    @property 
    def kappa(self):
        # Calculate relativstic quantum number from
        # non-relativistic input
        if self.n == 0:
            return -1 - self.lprimes
        else:
            return -1 - self.l

    def get_bound_wave(self):
        assert self.n > 0
        # Get atom
        fac.SetAtom(fac.ATOMICSYMBOL[self.Z])
        # Set up configuration
        fac.Config(self.configstring, self.config)
        # Optimize atomic energy levels
        fac.ReinitRadial(0)
        fac.ConfigEnergy(0)
        # Optimize radial wave functions
        fac.OptimizeRadial(self.configstring)
        # Optimize energy levels
        fac.ConfigEnergy(1)
        # Output desired wave function from table
        fac.WaveFuncTable("orbital.txt", self.n, self.kappa)
        fac.Reinit(config=1)

        with open("orbital.txt", "r") as content_file:
            content = content_file.read()

        self.ilast = int(re.search("ilast\\s+=\\s+([0-9]+)", content).group(1))
        self.energy = float(re.search("energy\\s+=\\s+([^\\n]+)\\n", content).group(1))
        # Load information into table
        table = np.loadtxt("orbital.txt", skiprows=15)

        # Load radial grid (in atomic units)
        self.r = table[:, 1]

        # Load large component of wave function
        self.wfn_table = table[: self.ilast, 4]
        bound_wave = interp1d(table[: self.ilast, 1], table[: self.ilast, 4], kind="cubic", fill_value=0)
        return bound_wave

    def get_continuum_waves(self):
        assert self.n == 0 
        # Get atom
        fac.SetAtom(fac.ATOMICSYMBOL[self.Z])
        # Set up configuration
        fac.Config(self.configstring, self.config)
        # Optimize atomic energy levels
        fac.ReinitRadial(0)
        fac.ConfigEnergy(0)
        # Optimize radial wave functions
        fac.OptimizeRadial(self.configstring)
        # Optimize energy levels
        fac.ConfigEnergy(1)

        continum_waves = []
        for epsilon in self.epsilon:
            continum_waves_per_epsilon = []
            for k in self.kappa:
                fac.WaveFuncTable("orbital.txt", self.n, k, epsilon)
                fac.Reinit(config=1)

                with open("orbital.txt", "r") as content_file:
                    content = content_file.read()

                self.ilast = int(re.search("ilast\\s+=\\s+([0-9]+)", content).group(1))
                self.energy = float(re.search("energy\\s+=\\s+([^\\n]+)\\n", content).group(1))
                # Load information into table
                table = np.loadtxt("orbital.txt", skiprows=15)

                # Load radial grid (in atomic units)
                self.r = table[:, 1]

                # Load large component of wave function
                self.wfn_table = table[: self.ilast, 4]

                # For continuum wave function also change normalization units from
                # 1/sqrt(k) in atomic units to units of 1/sqrt(Angstrom eV)
                # wavenumber in atomic units
                ke = np.sqrt(2 * epsilon / units.Hartree * (1 + units.alpha ** 2 * epsilon / units.Hartree / 2))
                # Normalization used in flexible atomic code
                # sqrt_ke =  np.sqrt(ke)
                # Desired normalization from Manson 1972 1 / np.sqrt(np.pi) / (epsilon / units.Rydberg) ** 0.25
                norm = 1 / np.sqrt(np.pi) 

                # If continuum wave function load phase-amplitude solution
                self.amplitude = table[:, 2] * norm
                self.phase = table[:, 3]
                self.wfn_table *= norm 

                wvfn = np.zeros(self.r.shape, dtype=np.float)

                wvfn[:self.ilast-1] = self.wfn_table[:self.ilast-1]
                # Tabulated
                TB = self.wfn_table[self.ilast-1]
                # Phase amplitude
                PA = self.amplitude[self.ilast+1] * np.sin(self.phase[self.ilast+1])
                r1 = self.r[self.ilast-1]
                r2 = self.r[self.ilast+1]
                wvfn[self.ilast-1:self.ilast+2] = interp1d([r1,r2],[TB,PA])(self.r[self.ilast-1:self.ilast+2])
                wvfn[self.ilast+2:] = self.amplitude[self.ilast+2:] * np.sin(self.phase[self.ilast+2:])
                # For bound wave functions we simply interpolate the
                # tabulated values of a0 the wavefunction
                cwave = interp1d(self.r, wvfn, kind="cubic", fill_value=0)
                continum_waves_per_epsilon.append(cwave)
            continum_waves.append(continum_waves_per_epsilon)
        return continum_waves



    

    
    