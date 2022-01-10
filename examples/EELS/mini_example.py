import pfac.fac

# bounded state
Z=14
config = "1s2 2s2 2p6 3s2 3p2"
configstring = pfac.fac.ATOMICSYMBOL[Z] + "bound"
pfac.fac.SetAtom(pfac.fac.ATOMICSYMBOL[Z])
# Set up configuration
pfac.fac.Config(configstring, config)
# Optimize atomic energy levels
pfac.fac.ConfigEnergy(0)
# Optimize radial wave functions
pfac.fac.OptimizeRadial(configstring)
# Optimize energy levels
pfac.fac.ConfigEnergy(1)


# continuum state with core-electron inoized
Z=14
ell = 1
epsilon =1
config = "1s1 2s2 2p6 3s2 3p2"
configstring = pfac.fac.ATOMICSYMBOL[Z] + "ex"
pfac.fac.ReinitRadial(0)
pfac.fac.SetAtom(pfac.fac.ATOMICSYMBOL[Z])
# Set up configuration
pfac.fac.Config(configstring, config)
# Optimize atomic energy levels
pfac.fac.ConfigEnergy(0)
# Optimize radial wave functions
pfac.fac.OptimizeRadial(configstring)
# Optimize energy levels
pfac.fac.ConfigEnergy(1)

# # Calculate relativstic quantum number from
# # non-relativistic input
kappa = -1 - ell

# # Output desired wave function from table
pfac.fac.WaveFuncTable("orbital.txt", 0, kappa, epsilon)

# # Clear table
# # ClearOrbitalTable ()
pfac.fac.Reinit(config=1)

import numpy as np 
import re

with open("orbital.txt", "r") as content_file:
    content = content_file.read()

ilast = int(re.search("ilast\\s+=\\s+([0-9]+)", content).group(1))
energy = float(re.search("energy\\s+=\\s+([^\\n]+)\\n", content).group(1))
# Load information into table
table = np.loadtxt("orbital.txt", skiprows=15)

# Load radial grid (in atomic units)
r = table[:, 1]

# Load large component of wave function
wfn_table = table[: ilast, 4]