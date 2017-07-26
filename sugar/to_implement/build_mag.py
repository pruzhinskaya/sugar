import pylab as P
import numpy as N
import scipy
from ToolBox.Wrappers import SALT2model



def mag(lbda, flux,lambda_min=6360,lambda_max=6600,var=None, step=None,Model=None,AB=True):
    """
    Computes the magnitude of a given spectrum.
    
    :param array lbda: array of wavelength
    :param array flux: array of flux. Same length as lbda
    :param array var: array of variance. Same length as lbda
    :param array/float int: binning
   
    :return: magnitude (error)
    """
    if Model is not None :
        model=Model
    else:
        model=SALT2model.Salt2Model()


    filt_lbda=N.linspace(lambda_min,lambda_max,lambda_max-lambda_min+1)
    filt_flux=N.ones(len(filt_lbda))
    filt_mean_wlength=N.average(filt_lbda,weights=filt_flux)

    filt_lbda=N.linspace(lbda[1],lbda[-2],int(lbda[-2]-lbda[1])+1)
    filt_flux=N.zeros(len(filt_lbda))
    Transmition=((filt_lbda>lambda_min) & (filt_lbda<lambda_max))
    filt_flux[Transmition]=1.
    
    Lambda_AB=N.linspace(1000,10000,9001)
    AB_mag_flux=3631./(3.34*(10**4)*Lambda_AB)
        
    photons = SALT2model.integ_photons(lbda,flux,step,filt_lbda,filt_flux)
    if AB:
        refphotons = SALT2model.integ_photons(Lambda_AB, AB_mag_flux,
                                              None, filt_lbda, filt_flux)
    else:
        refphotons = SALT2model.integ_photons(model.RefSpec.lbda, model.RefSpec.flux,
                                              None, filt_lbda, filt_flux)
        
    if photons is None or refphotons is None:
        if var is None:
            return -float(N.inf), float(N.inf)
        else:
            return -float(N.inf), float(N.inf)
        
    outmag = -2.5 / N.log(10) * N.log(photons / refphotons)


    if model.VegaMags is not None:
        outmag += N.interp(filt_mean_wlength,
                            model.VegaMags.lbda,
                            model.VegaMags.flux)
        
    if var is not None:
        var = SALT2model.integ_photons_variance(lbda, var, step, filt_lbda, filt_flux)
        magerr = 2.5 / N.log(10) * N.sqrt(var) / photons
        return float(outmag), float(magerr)
    else:
        return float(outmag), None





