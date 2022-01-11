from pylab import *
from scipy.integrate import odeint
from scipy.optimize import brentq
 
a=1 # Never changes. Constant? yes
B=4 # Never changes. Constant? yes
L= B+a # Never changes. Constant? yes these tree are here more for me to easily change
Vmax= 50
Vpot = False
 
 
# V() would work better as a closure or class method
# Needs a more descriptive name; what is V? Vector? Velocity?
# V is the potential function for the sistem, eg imagin ammonia molecule,
# it is shaped like a pyramid
#  with three hydrogen at the baseand nitrogen
# at the top. the potential Vo represents the area with the the hydrogen
# atoms. the area where V(x)= 0 are the places that you'd probably find N.
# when Wave_function is called and all the zeros are found i have the energy
# with that energy I can plot the wave function onto a graph, but like the
# wave function is not being solved correctly because when I plot  the wave
# using the correct energy that is found by find_analytic_energies, at x =-a-B
# the wave function is zero( whichis correct) but at x = L the wave function is not zero
# that leads me to conclude that the problem is with the solution of the system of ordinary
# differential equations. I can't tell if I followed the advice given to me on scicomp
# "If you really want to deal with an infinite potential well, then you should set  b=L
# and enforce the boundary condition ψ(b)=0. In this case it also makes sense
# to start shooting at x=−b, with ψ(−b)=0and ψ′(b)nonzero." - LonelyProf
 
def V(x):
    '''
    #Potential function in the finite square well.
    '''
    if -a <=x <=a:
        val = Vo
    elif x<=-a-B:
        val = Vmax
    elif x>=L:
        val = Vmax
    else:
        val = 0
    # This conditional can never be entered     #### this is here for parts of the code that come later on
    ##                                               I tried reducing things here to a min. for the problem to be clearer
    if Vpot==True: # never the case, Vpot does not change
          if -a-B-(10/N) < x <= L+(1/N):
             Ypotential.append(val) # sequence does not exist
             Xpotential.append(x) # sequence does not exist
    return val
 
def SE(psi, x):
    """
    Returns derivatives for the 1D schrodinger eq.
    Requires global value E to be set somewhere. State0 is first derivative of the
    wave function psi, and state1 is its second derivative.
    """
    state0 = psi[1]
    state1 = 2.0*(V(x) - E)*psi[0]
    return array([state0, state1])
 
def Wave_function(energy):
    """
    Calculates wave function psi for the given value
    of energy E and returns value at point b
    """
    global psi # Functions should not call global variables
    global E # Functions should not call global variables
    E = energy # Functions should not set global variables from within
    psi = odeint(SE, psi0, x) # Functions should not set global variables from within
    return psi[-1,0]
 
def find_all_zeroes(x,y):
    """
    Gives all zeroes in y = Psi(x)
    """
    all_zeroes = []
    s = sign(y)
    for i in range(len(y)-1):
        if s[i]+s[i+1] == 0:
            zero = brentq(Wave_function, x[i], x[i])
            all_zeroes.append(zero)
    return all_zeroes
 
def find_analytic_energies(en):
    """
    Calculates Energy values for the finite square well using analytical
    model (Griffiths, Introduction to Quantum Mechanics, 1st edition, page 62.)
    """
    z = sqrt(2*en)
    z0 = sqrt(2*Vo)
    z_zeroes = []
    f_sym = lambda z: tan(z)-sqrt((z0/z)**2-1)      # Formula 2.138, symmetrical case
    f_asym = lambda z: -1/tan(z)-sqrt((z0/z)**2-1)  # Formula 2.138, antisymmetrical case
 
    # first find the zeroes for the symmetrical case
    s = sign(f_sym(z))
    for i in range(len(s)-1):   # find zeroes of this crazy function
       if s[i]+s[i+1] == 0:
           zero = brentq(f_sym, z[i], z[i+1])
           z_zeroes.append(zero)
    print ("Energies from the analyitical model are: ")
    print ("Symmetrical case)")
    for i in range(0, len(z_zeroes),2):   # discard z=(2n-1)pi/2 solutions cause that's where tan(z) is discontinous
        print ("%.4f" %(z_zeroes[i]**2/2))
    # Now for the asymmetrical
    z_zeroes = []
    s = sign(f_asym(z))
    for i in range(len(s)-1):   # find zeroes of this crazy function
       if s[i]+s[i+1] == 0:
           zero = brentq(f_asym, z[i], z[i+1])
           z_zeroes.append(zero)
    print ("(Antisymmetrical case)")
    for i in range(0, len(z_zeroes),2):   # discard z=npi solutions cause that's where ctg(z) is discontinous
        print ("%.4f" %(z_zeroes[i]**2/2))
 
N = 1000                  # number of points to take
psi = np.zeros([N,2])     # Wave function values and its derivative (psi and psi')
psi0 = array([0,1])   # Wave function initial states
Vo = 50
E = 0.0                   # global variable Energy  needed for Sch.Eq, changed in function "Wave function"
b = L                     # point outside of well where we need to check if the function diverges
x = linspace(-B-a, L, N)    # x-axis
 
def main():
    # main program
 
    en = linspace(0, Vo, 1000000)   # vector of energies where we look for the stable states
 
    psi_b = []      # vector of wave function at x = b for all of the energies in en
    for e1 in en:
        psi_b.append(Wave_function(e1))     # for each energy e1 find the the psi(x) at x = b
    E_zeroes = find_all_zeroes(en, psi_b)   # now find the energies where psi(b) = 0
 
    # Print energies for the bound states
    print ("Energies for the bound states are: ")
    for E in E_zeroes:
        print ("%.2f" %E)
    # Print energies of each bound state from the analytical model
    find_analytic_energies(en)
 
    # Plot wave function values at b vs energy vector
    figure()
    plot(en/Vo,psi_b)
    title('Values of the $\Psi(b)$ vs. Energy')
    xlabel('Energy, $E/V_0$')
    ylabel('$\Psi(x = b)$', rotation='horizontal')
    for E in E_zeroes:
        plot(E/Vo, [0], 'go')
        annotate("E = %.2f"%E, xy = (E/Vo, 0), xytext=(E/Vo, 30))
    grid()
 
    # Plot the wavefunctions for first 4 eigenstates
    figure(2)
    for E in E_zeroes[0:4]:
        Wave_function(E)
        plot(x, psi[:,0], label="E = %.2f"%E)
    legend(loc="upper right")
    title('Wave function')
    xlabel('x, $x/L$')
    ylabel('$\Psi(x)$', rotation='horizontal', fontsize = 15)
    grid()
 
    figure(3)
    pot =[]
    for i in x:
        pot.append(V(i))
    plot(x,pot)
    show()
if __name__ == "__main__":
    main()