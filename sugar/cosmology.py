"""cosmology function for sugar."""

import numpy as np 
from scipy import integrate


def distance_metric(z,Omega_M=0.3):
    """
    metric distance formula.
    """
    return 1./np.sqrt(Omega_M*(1+z)**3+(1.-Omega_M))


def luminosity_distance(zh,zcmb,H0=70.):
    """
    luminosity distance formula.
    """
    c=299792.458
    if type(zcmb)==np.ndarray:
        integr=np.zeros_like(zcmb)
        for i in range(len(zcmb)):
            integr[i]=integrate.quad(distance_metric,0,zcmb[i])[0]
    else:
        integr=integrate.quad(distance_metric,0,zcmb)[0]

    return (1+zh)*(c/H0)*integr*10**6


def distance_modulus(zhel,zcmb,cst=0):
    """distance_modulus formula"""
    return 5.*np.log(luminosity_distance(zhel,zcmb))/np.log(10.)-5.+cst
