import numpy as np
from astropy import units as u # good for maintaining units to avoid error
from astropy.constants import G, M_earth #easier to import these values from astropy

M = M_earth 
m = 7.348e22 * u.kg #kgs
R = 3.844e8 * u.m # meters
omega = 2.662e-6 / u.s # 1/s or s^-1


# need to define f and fprime to do newton's method, where we find roots using 
# the functions derivative to estimate the next guess 

def f(r):
    return G * M / r**2 - G * m /(R - r)**2 - (omega**2)*r  # m/s^2

def fprime(r):
    return -2*G*M/r**3 - 2*G*m/(R - r)**3 - omega**2 # 1/s^2 

r = 3.10e8 * u.m # initial guess, in meters
# Newton's method
for i in range(400):
    r_mod = r - f(r)/fprime(r)   
    if abs((r_mod - r).to_value(u.m)) < 1e-6: # this line checks if the change between old and new guesses is miniscule
        r = r_mod
        break # stops the loop if close enough
    r = r_mod

print("r = {:.4e}".format(r)) # prints the value for r to 4 decimal points
