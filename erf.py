import numpy as np
import matplotlib.pyplot as plt

# the integral from 0 to x of e^-t^2 is the error function, erf(x) ! 

set = np.linspace(0.0, 3.0, 30)     

dt = set[1] - set[0]

f = np.exp(-set**2)

# i'm going to use the trapezoidal method with riemann sums to solve this integral numerically,
# since it is unsolveable analytically

f_left = np.cumsum(f) * dt       # left Riemann sum, using np.cumsum, calculating the total elements in an array or df
f_trap = f_left - 0.5*f[0]*dt     # to get the trapezoidal sum we subtract half of the first component
f_trap[1:] -= 0.5*f[1:]*dt       # we also subtract the half of each current component

# adding a very small number to our 2nd input value in np.arange is trick i learned in undergrad

x = np.arange(0.0, 3.0 + 1e-12, 0.1)
E = np.interp(x, set, f_trap) # using np.interp which we discussed in lecture

# now we plot in crimson, my favorite color! 
plt.plot(x, E, marker='*', markersize=9, linestyle='-', color='crimson', label='E(x)') # i wanted stars as a marker
plt.xlabel('x')
plt.ylabel('E(x)')
plt.title(r'$E(x)=\int_0^x e^{-t^2}\,dt$')
plt.grid()
plt.legend()
plt.show()
