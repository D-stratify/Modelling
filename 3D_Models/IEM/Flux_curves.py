import numpy as np
import matplotlib.pyplot as plt

ε = .025


l = lambda e,g : e**(1/2) / (e + g)**(1/2) 

f = lambda e,g : l(e,g) * e**(1/2) * g


g = np.linspace(0.001,.25,1000)

A_t = 1
B_t = -1 + g + g/ε
C_t = -g


e_p = (-B_t + np.sqrt(B_t**2 - (4*A_t*C_t) ) )/ (2 * A_t)
plt.plot(g, f(e_p, g), 'k:',label=r'$e^+$')

e_m =  (-B_t - np.sqrt(B_t**2 - (4*A_t*C_t)) )/ (2 * A_t)
plt.plot(g, f(e_m, g), 'k-',label=r'$e^-$')

plt.xlabel(r'$E[dB/dz]$')
plt.ylabel(r'$E[WB]$')
plt.legend()
plt.show()