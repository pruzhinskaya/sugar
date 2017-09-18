""" extinction law used in the paper to modeled sugar sed."""

import numpy as np


def CCMextinctionParameters(lbda):
    """Extinction law parameters a(lbda),b(lbda), to be used in
    A(lambda)/A(V) = a(lbda) + b(lbda)/Rv for lbda [A] in the 1000
    A-33 microns range (for a V-filter centered on 5494.5
    A). R(V):=A(V)/E(B-V) ranges from 2.75 to 5.3, Rv=3.1 being the
    canonical Milky Way value. Uses expressions from Cardelli, Clayton
    and Mathis (1989ApJ...345..245C), with optical/NIR part updated
    from O'Donnell (1994ApJ...422..158O)."""

    def cardelliIR(x):
        """A/Av in InfraRed: 0.3-1.1 micron^-1"""
        assert ((x>=0.3) & (x<=1.1)).all()
        a = +0.574 * x**1.61
        b = -0.527 * x**1.61
        return a,b

    def cardelliOpt(x):
        """A/Av in Optical/Near IR: 1.1-3.3 micron^-1"""
        assert ((x>=1.1) & (x<=3.3)).all()
        y = x - 1.82
        pa = [+0.32999,-0.77530,+0.01979,+0.72085,-0.02427,-0.50447,+0.17699,1]
        pb = [-2.09002,+5.30260,-0.62251,-5.38434,+1.07233,+2.28305,+1.41338,0]
        a = np.polyval(pa, y)
        b = np.polyval(pb, y)
        return a,b

    def cardelliUV(x):
        """A/Av in UV: 3.3-8 micron^-1"""
        assert ((x>=3.3) & (x<=8)).all()
        y = x - 5.9
        fa = np.where(x>=5.9, ( -0.04473 - 0.009779*y ) * y**2, 0)
        fb = np.where(x>=5.9, ( +0.2130  + 0.1207  *y ) * y**2, 0)
        a = +1.752 - 0.316*x - 0.104/( (x-4.67)**2 + 0.341 ) + fa
        b = -3.090 + 1.825*x + 1.206/( (x-4.62)**2 + 0.263 ) + fb
        return a,b

    def cardelliFUV(x):
        """A/Av in Far UV: 8-10 micron^-1"""
        assert ((x>=8) & (x<=10)).all()
        y = x - 8.
        a = np.polyval([-0.070,+0.137,-0.628,-1.073], y)
        b = np.polyval([+0.374,-0.420,+4.257,+13.67], y)
        return a,b

    x = 1e4/np.atleast_1d(lbda)          # [micron^-1]
    iIR = (x>=0.3) & (x<1.1)            # IR
    iOpt= (x>=1.1) & (x<3.3)            # Opt/Near IR
    iUV = (x>=3.3) & (x<8)              # UV
    iFUV= (x>=8)   & (x<=10)            # Far UV

    a = np.zeros(len(x),'d') * np.nan
    b = np.zeros(len(x),'d') * np.nan
    a[iIR],b[iIR]   = cardelliIR(x[iIR])
    a[iOpt],b[iOpt] = cardelliOpt(x[iOpt]) # Original CCM89 expression
    a[iUV],b[iUV]   = cardelliUV(x[iUV])
    a[iFUV],b[iFUV] = cardelliFUV(x[iFUV])

    return a,b

def extinctionLaw(lbda, Rv=3.1):
    """
    Return extinction law A(lbda)/Av for a given value of Rv and a given
    extinction law:
    CCM89: Cardelli, Clayton and Mathis (1989ApJ...345..245C)
    """
    a, b = CCMextinctionParameters(lbda)

    return a + b/Rv
