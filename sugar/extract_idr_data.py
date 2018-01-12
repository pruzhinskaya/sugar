"""read idr from SNFactory data and build data to train sugar model."""

import numpy as np
import scipy as sp
import cPickle
import copy
import sugar
import pySnurp
import os
from ToolBox import Astro, Cosmology
import scipy.interpolate as inter


class spec:
    """to do."""
    def __init__(self, y, x, v, step):
        """to do."""
        self.y=y
        self.x=x
        self.v=v
        self.step=step

def bin_spec(spec,x):
    """
    Returns the bin in which x is in the spectrum.
    In case x is not in the spectrum, the return value is an empty array
    """
    # comput the true/false array
    ubound = spec.x + spec.step/2
    lbound = np.concatenate(( [(spec.x-spec.step/2)[0]],ubound[:-1]))
    cond = (x<ubound) & (x>=lbound)
    return np.nonzero(cond)

def mean_spec(spec, x1, x2):
    """
    Returns the integral of the flux over the wavelength range defined as [x1,x2], divided by the wavelength range in order to get a flux/wavelength.
    the variance of this quantity is returned as a 2nd parameter
    Raises ValueError if the spec range soesn't cover the intended bin width
    """
    # determine first, middle and last bins : upper bound belongs to upper bin
    bin1 = bin_spec(spec,x1)
    bin2 = bin_spec(spec,x2)
    if bin1 == bin2 or x2 == (spec.x + spec.step / 2)[-1]:
        return spec.y[bin1], spec.v[bin1]
    binbetween = range(bin1[0][0]+1,bin2[0][0])
    # compute flux integral
    flux1 = spec.y[bin1] * ((spec.x+spec.step/2)[bin1]-x1)
    flux2 = spec.y[bin2] * (x2+(-spec.x+spec.step/2)[bin2])
    fluxbetween = sum((spec.y*spec.step)[binbetween])
    retflux = (flux1+flux2+fluxbetween)/(x2-x1)
    # compute variance of the previous quantity
    var1 = spec.v[bin1] * ((spec.x+spec.step/2)[bin1]-x1)**2
    var2 = spec.v[bin2] * (x2+(-spec.x+spec.step/2)[bin2])**2
    varbetween = sum((spec.v*spec.step**2)[binbetween])
    retvar = (var1+var2+varbetween)/(x2-x1)**2
    if len(retflux) == 0:
        raise ValueError("Bound error %f %f"%(x1,x2))
    return retflux, retvar

def rebin(spec, xarray):
    """xarray is the array of bin edges (1 more than number of bins)."""
    outx = (xarray[1:]+xarray[:-1])/2
    outflux = np.zeros(len(outx))
    outvar = np.zeros(len(outx))
    for i in xrange(len(outx)):
        outflux[i], outvar[i] = mean_spec(spec,xarray[i],xarray[i+1])
    return outx, outflux, outvar

def go_to_flux(X, Y, ABmag0=48.59):
    """Convert AB mag to flux."""
    Flux_nu = 10**(-0.4*(Y+ABmag0))
    f = X**2 / 299792458. * 1.e-10
    Flux_lambda = Flux_nu/f
    return Flux_lambda

def check_bounds(X):
    """check the bounds."""
    spec_min = max([np.min(X[sn]) for sn in range(len(X))])
    spec_max = min([np.max(X[sn]) for sn in range(len(X))])
    return spec_min, spec_max

class build_spectral_data:
    """
    For a given idr, will extract spectra of SNIa.
    It will be used for gaussian process interpolation, 
    and SNIa fitting.
    """
    def __init__(self, idr_rep='data_input/SNF-0203-CABALLOv2/', redshift_min=0.01, redshift_max=9999.,
                 mjd_min = 0., mjd_max=55250., day_max=2.5, guy10=True):
        """
        Init build spectral data.
        """
        self.idr_rep = idr_rep
        self.meta = cPickle.load(open(os.path.join(self.idr_rep,'META.pkl')))
        self.redshift_min = redshift_min
        self.redshift_max = redshift_max
        self.mjd_min = mjd_min
        self.mjd_max = mjd_max
        self.guy10 = guy10
        self.day_max = day_max

        sn_name = self.meta.keys()
        filtre = np.array([True] * len(sn_name))

        for i in range(len(sn_name)):
            sn = sn_name[i]
            day_max_mjd = self.meta[sn]['salt2.DayMax']
            closest_to_day_max = min(abs(np.array([self.meta[sn]['spectra'][exposure]['salt2.phase'] \
                                                   for exposure in self.meta[sn]['spectra'].keys()])))
            zhelio = self.meta[sn]['host.zhelio']
            sample = self.meta[sn]['idr.subset']

            if day_max_mjd < self.mjd_min or day_max_mjd > self.mjd_max:
                filtre[i] = False
            if zhelio < self.redshift_min or zhelio > self.redshift_max:
                filtre[i] = False
            if closest_to_day_max > self.day_max:
                filtre[i] = False
            if self.guy10 and sample not in ['training','validation']:
                filtre[i] = False

        self.sn_name = np.array(sn_name)[filtre]
        self.sn_name.sort()

        self.dico_spectra = {}
        self.observed_wavelength = []

    def load_spectra(self):
        """
        Load spectra select within the idr and the init.
        Will correct also from Milky way extinction and
        de-redshift the spectra using heliocentric redshift. 
        """
        self.dico_spectra = {}
        self.observed_wavelength = []

        min_wavelength = [] 
        max_wavelength = []
        for i,sn in enumerate(self.sn_name):
            spec = {}
            ind = 0
            for j,pause in enumerate(self.meta[sn]['spectra'].keys()):
                get_spectra = pySnurp.Spectrum(os.path.join(self.idr_rep,self.meta[sn]['spectra'][pause]['idr.spec_merged']))

                get_spectra.deredden(self.meta[sn]['target.mwebv'])
                get_spectra.deredshift(self.meta[sn]['host.zhelio'])

                spectra_ok = (len(get_spectra.x) == 2691)
                if 'procB.Quality' in self.meta[sn]['spectra'][pause].keys():
                    spectra_ok = (spectra_ok & (self.meta[sn]['spectra'][pause]['procB.Quality'] == 1))
                if 'procR.Quality' in self.meta[sn]['spectra'][pause].keys():
                    spectra_ok = (spectra_ok & (self.meta[sn]['spectra'][pause]['procR.Quality'] == 1))
                
                if spectra_ok: #R and B channel alive and flag quality ok within both chanel.
                    min_wavelength.append(min(get_spectra.x))
                    max_wavelength.append(max(get_spectra.x))
                    spec.update({'%i'%(ind):{'Y': get_spectra.y,
                                             'X': get_spectra.x,
                                             'V': get_spectra.v,
                                             'step': get_spectra.step,
                                             'days': self.meta[sn]['spectra'][pause]['obs.mjd'],
                                             'z_cmb': self.meta[sn]['host.zcmb'],
                                             'z_helio': self.meta[sn]['host.zhelio'],
                                             'z_err': self.meta[sn]['host.zhelio.err'],
                                             'phase_salt2': self.meta[sn]['spectra'][pause]['salt2.phase'],
                                             'pause': pause}})
                    self.observed_wavelength.append(get_spectra.x)
                    ind += 1
            self.dico_spectra.update({sn:spec})


    def resampled_spectra(self, velocity=1500.):
        """
        Resample spectra on the same grid.
        Sampled in velocity.
        """
        delta_lambda = velocity / 3.e5
        dico_spectra = {}
        lmin, lmax = check_bounds(self.observed_wavelength)
        rebinarray = [lmin]
        i = 0
        while rebinarray[i]<lmax:
            rebinarray.append((rebinarray[i] * delta_lambda) + rebinarray[i])
            i += 1
        if rebinarray[i]>lmax:
            del rebinarray[i]
        rebinarray = np.array(rebinarray)
        print "%i filters created"%(len(rebinarray)-1)
        
        for i,sn in enumerate(self.sn_name):
            spectra = copy.deepcopy(self.dico_spectra[sn])
            for j,pause in enumerate(self.dico_spectra[sn].keys()):
                print 'processing '+sn+' pause '+pause 
                spec_object=spec(spectra[pause]['Y'], spectra[pause]['X'], spectra[pause]['V'], spectra[pause]['step'])
                spectra[pause]['X'], spectra[pause]['Y'], spectra[pause]['V'] = rebin(spec_object,rebinarray)

            dico_spectra.update({sn:spectra})

        self.dico_spectra = dico_spectra

    def to_ab_mag(self):
        """
        Convert the actual dic of spectra to AB mag.
        """
        dic_ab = {}
        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            spectra = copy.deepcopy(self.dico_spectra[sn])
            for j,pause in enumerate(spectra.keys()):
                print 'processing '+sn+' pause '+pause
                spectra[pause].update({'V_flux':copy.deepcopy(spectra[pause]['V'])})
                spectra[pause].update({'Y_flux_without_cosmology':copy.deepcopy(spectra[pause]['Y'])})
                spectra[pause]['V'] = Astro.Coords.flbda2ABmag(spectra[pause]['X'], spectra[pause]['Y'], var=spectra[pause]['V'])
                spectra[pause]['Y'] = Astro.Coords.flbda2ABmag(spectra[pause]['X'], spectra[pause]['Y'])
            dic_ab.update({sn:spectra})
        self.dico_spectra = dic_ab

    def cosmology_corrected(self):
        """
        Correct spectra from distance using LCDM cosmology.
        """
        dic_spectra = {}
        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            spectra = copy.deepcopy(self.dico_spectra[sn])
            for j,pause in enumerate(spectra.keys()):
                print 'processing ' + sn + ' pause ' + pause
                spectra[pause]['Y'] += -5.*np.log10(sugar.distance_modulus(spectra[pause]['z_helio'], spectra[pause]['z_cmb'])) + 5.
                spectra[pause].update({'Y_flux':go_to_flux(spectra[pause]['X'],copy.deepcopy(spectra[pause]['Y']))})
            dic_spectra.update({sn:spectra})
        self.dico_spectra = dic_spectra

    def write_pkl(self,pkl_name):
        """
        write output in a pkl file.
        """
        File=open(pkl_name,'w')
        cPickle.dump(self.dico_spectra,File)
        File.close()


    def reorder_and_clean(self):
        """
        Reorder each SNIa spectra by incrasing phasing and put infinite weight for negative flux.
        """
        dic_spectra = {}
        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            spectra = copy.deepcopy(self.dico_spectra[sn])
            spec = {}
            ind_new = 0
            for j in range(len(spectra.keys())):
                minimum = (10.**23)
                ind = '0'
                for k,pause in enumerate(spectra.keys()):
                    phase = spectra[pause]['phase_salt2']
                    if phase < minimum:
                        minimum = phase
                        ind = pause
                if np.sum(np.isfinite(spectra[ind]['Y'])) == len(spectra[ind]['Y']):
                    spec.update({'%i'%(ind_new):copy.deepcopy(spectra[ind])})
                    ind_new += 1
                else:
                    filtre = np.isfinite(spectra[ind]['Y'])
                    spectra[ind]['Y'][~filtre] = np.average(spectra[ind]['Y'][filtre], weights=1./spectra[ind]['V'][filtre])
                    spectra[ind]['V'][~filtre] = 100.
                    spec.update({'%i'%(ind_new):copy.deepcopy(spectra[ind])})
                    ind_new += 1

                del spectra[ind]

            dic_spectra.update({sn:spec})

        self.dico_spectra = dic_spectra
        
    def control_plot(self):
        """
        Display all spectal time series.
        """
        import pylab as plt
        for sn in self.dico_spectra.keys():
            plt.figure()
            cst = 0
            for key in self.dico_spectra[sn].keys():
                plt.plot(self.dico_spectra[sn][key]['X'],self.dico_spectra[sn][key]['Y']-cst,'b')
                cst += 2.
            plt.title(sn)
            plt.gca().invert_yaxis()
            plt.show()


class build_at_max_data:

    def __init__(self, dico_cosmo, phrenology):
        self.si_list = ['EWCaIIHK', 'EWSiII4000', 'EWMgII',
                        'EWFe4800', 'EWSIIW', 'EWSiII5972',
                        'EWSiII6355', 'EWOI7773', 'EWCaIIIR',
                        'vSiII_4128_lbd', 'vSiII_5454_lbd',
                        'vSiII_5640_lbd', 'vSiII_6355_lbd']
        self.dic_cosmo = cPickle.load(open(dico_cosmo))
        self.phrenology = cPickle.load(open(phrenology))
        self.sn_name = self.dic_cosmo.keys()
        
    def select_spectra_at_max(self, window = [-2.5, 2.5]):
        """
        Select spectrum closest to max within the define time window.
        """
        self.dic_cosmo_at_max = {}

        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC = copy.deepcopy(self.dic_cosmo[sn])
            PPhase = -999
            for j,pause in enumerate(self.dic_cosmo[sn].keys()):
                if abs(SPEC[pause]['phase_salt2'])<abs(PPhase):
                    spec=SPEC[pause]
                    PPhase=SPEC[pause]['phase_salt2']
            if PPhase>window[0] and PPhase<window[1]:
                self.dic_cosmo_at_max.update({sn:spec})
        self.sn_name=self.dic_cosmo_at_max.keys()


    def select_spectral_indicators(self):
        """
        Select spectral indicators closest to max.

        The spectra where the spectral indicators
        were computed, come from the select_spectra_at_max(). 
        """
        dic_cosmo_at_max = {}

        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC = copy.deepcopy(self.dic_cosmo_at_max[sn])
            spectral_indicators = []
            spectral_indicators_error = []
            for SI in range(len(self.si_list)):
                PAUSE = SPEC['pause']
                spectral_indicators.append(self.phrenology[sn]['spectra'][PAUSE]['phrenology.'+self.si_list[SI]])
                spectral_indicators_error.append(self.phrenology[sn]['spectra'][PAUSE]['phrenology.'+self.si_list[SI]+'.err'])
            dic_sn = {'spectra':SPEC,
                      'spectral_indicators':np.array(spectral_indicators),
                      'spectral_indicators_error':np.array(spectral_indicators_error)}

            dic_cosmo_at_max.update({sn:dic_sn})
        
        self.dic_cosmo_at_max = dic_cosmo_at_max

    def write_pkl(self,pkl_name):
        """
        Write the output.
        """
        File = open(pkl_name,'w')
        cPickle.dump(self.dic_cosmo_at_max,File)
        File.close()


if __name__=="__main__":

    bsd = build_spectral_data('data_input/SNF-0203-CABALLOv2')
    bsd.load_spectra()
    bsd.resampled_spectra(velocity=1500.)
    bsd.to_ab_mag()
    bsd.cosmology_corrected()
    bsd.reorder_and_clean()
    bsd.write_pkl('test_output.pkl')
    bsd.control_plot()

    bmd = build_at_max_data('test_output.pkl', 'data_input/phrenology_2016_12_01_CABALLOv1.pkl')
    bmd.select_spectra_at_max()
    bmd.select_spectral_indicators()
    bmd.write_pkl('test_output_si.pkl')
