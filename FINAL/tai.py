import kwant
import kwant.continuum
import scipy.sparse.linalg
import scipy.linalg
import numpy as np
import time

import multiprocessing as mp
from functools import partial

## 24 hour format ##
def print_t(str_):
  print( "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + str(str_) )
  #print(str(str_))

s0 = np.eye(2)
sx = np.array([[0,1], [1,0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1,0],[0,-1]])

def no_disorder(x, y):
  return 0

######## SIMULATION PARAMETERS
params_1 = dict(A = 364.5e-3, B=-686e-3, C=0.0, D=-512e-3, M=1.2e-2, V=no_disorder)
params_2 = dict(A = 364.5e-3, B=-686e-3, C=0.0, D=-512e-3, M=-1e-3, V=no_disorder)

e_1 = -20e-3
e_2 = 25e-3
e_3 = 9e-3

a=5
Lx=200
Ly=500

Wmin=0.005
Wmax=1.00
npoints=60
nsamples=4
nthreads=3

Ws = np.logspace(np.log10(Wmin), np.log10(Wmax), npoints)

"""
  Generating QSH system
"""
def qsh_system(a=5, Lx=5000, Ly=500):
  
    hamiltonian = """
    (C-D*(k_x**2+k_y**2))*identity(4)
    + (M-B*(k_x**2+k_y**2))*kron(sigma_0, sigma_z)
    + A*k_x*kron(sigma_z, sigma_x)
    + A*k_y*kron(sigma_z, sigma_y)
    + V(x,y)*identity(4)
    """
  
    template = kwant.continuum.discretize(hamiltonian, grid=a)

    def shape(site):
        (x, y) = site.pos
        return (0 <= y < Ly and 0 <= x < Lx)

    def lead_shape(site):
        (x, y) = site.pos
        return (0 <= y < Ly)

    syst = kwant.Builder()
    syst.fill(template, shape, (0, 0))

    lead = kwant.Builder(kwant.TranslationalSymmetry([-a, 0]))
    lead.fill(template, lead_shape, (0, 0))

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    syst = syst.finalized()
    return syst

# The system itself
syst = qsh_system(a, Lx, Ly)

def calculate_transmission(W, params, nsamples):

  def disorder(x,y):
    return np.random.uniform(-W/2, W/2)
  
  params['V'] = disorder

  t1 = []
  t2 = []
  t3 = []
  for k in range(nsamples):
    print_t("Calculating transmissions for W={0}, sample {1}".format(W, k))
    t1.append(kwant.smatrix(syst, energy=e_1, params=params).transmission(1,0))
    t2.append(kwant.smatrix(syst, energy=e_2, params=params).transmission(1,0))
    t3.append(kwant.smatrix(syst, energy=e_3, params=params).transmission(1,0))

  return [[np.mean(t1), np.std(t1)], [np.mean(t2), np.std(t2)], [np.mean(t3), np.std(t3)]]

transmissions_1 = partial(calculate_transmission, params=dict(params_1), nsamples=nsamples)
transmissions_2 = partial(calculate_transmission, params=dict(params_2), nsamples=nsamples)

pool = mp.Pool(processes=nthreads)
results = pool.map(transmissions_1, Ws)

p1g1 = np.array([res[0] for res in results])
p1g2 = np.array([res[1] for res in results])
p1g3 = np.array([res[2] for res in results])

np.savetxt("Ws.txt", Ws)
np.savetxt("params_1_G1_nsamples={0}.txt".format(nsamples), p1g1)
np.savetxt("params_1_G2_nsamples={0}.txt".format(nsamples), p1g2)
np.savetxt("params_1_G3_nsamples={0}.txt".format(nsamples), p1g3)

#pool = mp.Pool(processes=nthreads)
#results = pool.map(transmissions_2, Ws)

#p2g1 = np.array([res[0] for res in results])
#p2g2 = np.array([res[1] for res in results])
#p2g3 = np.array([res[2] for res in results])

#np.savetxt("params_2_G1.txt", p2g1)
#np.savetxt("params_2_G2.txt", p2g2)
#np.savetxt("params_2_G3.txt", p2g3)
print_t("DONE.")
