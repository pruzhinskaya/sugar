"""read idr from SNFactory data and build data to train sugar model."""

import numpy as np
import scipy as sp
import cPickle
import copy
import sugar
#import pySnurp ##de-reddenning, de-redshift
#import merged_spectrum as ms ## do the rebinning
#from ToolBox import Astro,Cosmology ## flux to ABmag, distance
import scipy.interpolate as inter
#import SnfMetaData # select training
#import mpl_to_delate as mpl # remove by eye strange target


class spec:
    """to do."""
    def __init__(self,y,x,v,step):
        """to do."""
        self.y=y
        self.x=x
        self.v=v
        self.step=step


def go_to_flux(X,Y,ABmag0=48.59):
    """Convert AB mag to flux."""
    Flux_nu=10**(-0.4*(Y+ABmag0))
    f = X**2 / 299792458. * 1.e-10
    Flux_lambda=Flux_nu/f
    return Flux_lambda


def check_bounds(X,lmin,lmax,step):
    """check the bounds."""
    spec_min=max([N.min(X[sn]) for sn in range(len(X))])
    spec_max=min([N.max(X[sn]) for sn in range(len(X))])
    if spec_min > lmin:
        lmin=spec_min
    if spec_max<lmax+step:
        lmax=spec_max-step
    return lmin,lmax


class build_spectral_data:
    """
    For a given idr, will extract spectra of SNIa.
    It will be used for gaussian process interpolation, 
    and SNIa fitting.
    """
    def __init__(self, meta, redshift_min=0.01, redshift_max=9999.,
                 mjd_min = 0., mjd_max=55250., day_max=2.5, guy10=True):
        """
        Init build spectral data.
        """
        self.meta = cPickle.load(open(meta))
        self.redshift_min = redshift_min
        self.redshift_max = redshift_max
        self.mjd_min = mjd_min
        self.mjd_max = mjd_max
        self.guy10 = guy10
        self.day_max = day_max
        sn_name = self.meta.keys()

        filtre = np.array([True] * len(sn_name))

        lsd = sugar.load_data_sugar()
        
        for i in range(len(sn_name)):
            sn = sn_name[i]
            day_max_mjd = self.meta[sn]['salt2.DayMax']
            closest_to_day_max = min(abs(np.array([self.meta[sn]['spectra'][exposure]['salt2.phase'] \
                                                   for exposure in self.meta[sn]['spectra'].keys()])))
            AA = len(self.meta[sn]['spectra'].keys())
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
            if filtre[i]:
                if sn not in lsd.sn_name:
                    print sn, day_max_mjd, closest_to_day_max, zhelio, sample, AA

        self.sn_name = np.array(sn_name)[filtre]
        self.sn_name.sort()

#    def load_spectra(self):
#
#        self.dic={}
#        self.X=[]
#        for i,sn in enumerate(self.sn_name):
#            print '%i/%i'%(((i+1),len(self.sn_name)))
#            SPEC={}
#            ind=0
#            for j,pause in enumerate(self.dicM[sn]['spectra'].keys()):
#                print 'processing '+sn+' pause '+pause 
#                Spec=pySnurp.Spectrum(self.dicM[sn]['spectra'][pause]['idr.spec_merged'])
#                Spec.deredden(self.dicM[sn]['target.mwebv'])
#                Spec.deredshift(self.dicM[sn]['host.zhelio'])
#                    
#                if len(Spec.x)==2691:
#
#                    SPEC.update({'%i'%(ind):{'Y':Spec.y,
#                                             'X':Spec.x,
#                                             'V':Spec.v,
#                                             'step':Spec.step,
#                                             'days':self.dicM[sn]['spectra'][pause]['obs.mjd'],
#                                             'z_cmb':self.dicM[sn]['host.zcmb'],
#                                             'z_helio':self.dicM[sn]['host.zhelio'],
#                                             'z_err':self.dicM[sn]['host.zhelio.err'],
#                                             'phase_salt2':self.dicM[sn]['spectra'][pause]['salt2.phase'],
#                                             'pause':pause}})
#                    self.X.append(Spec.x)
#                    ind+=1
#
#            self.dic.update({sn:SPEC})
#
#
#    def build_resampled_specs(self,lmin=3200,lmax=8900,step=10,delta_lambda=0.005):
#        
#        self.dic_binned={}
#        lmin,lmax=check_bounds(self.X,lmin,lmax,step=step)
#        #rebinarray=N.arange(lmin,lmax+step,step)
#        rebinarray=[lmin]
#        i=0
#        while rebinarray[i]<lmax:
#            rebinarray.append(rebinarray[i]*delta_lambda+rebinarray[i])
#            i+=1
#        if rebinarray[i]>lmax:
#            del rebinarray[i]
#
#        rebinarray=N.array(rebinarray)    
#        print rebinarray
#        print "%i filters created"%(len(rebinarray)-1)
#
#        
#        for i,sn in enumerate(self.sn_name):
#            print '%i/%i'%(((i+1),len(self.sn_name)))
#            SPEC=copy.deepcopy(self.dic[sn])
#            for j,pause in enumerate(self.dic[sn].keys()):
#                print 'processing '+sn+' pause '+pause 
#            
#                Spec=spec(SPEC[pause]['Y'],SPEC[pause]['X'],SPEC[pause]['V'],SPEC[pause]['step'])
#                SPEC[pause]['X'],SPEC[pause]['Y'],SPEC[pause]['V']=ms.rebin(Spec,rebinarray)
#
#            self.dic_binned.update({sn:SPEC})
#
#
#    def to_AB_mag(self):
#
#        self.dic_AB={}
#        for i,sn in enumerate(self.sn_name):
#            print '%i/%i'%(((i+1),len(self.sn_name)))
#            SPEC=copy.deepcopy(self.dic_binned[sn])
#            for j,pause in enumerate(self.dic_binned[sn].keys()):
#                print 'processing '+sn+' pause '+pause
#                SPEC[pause].update({'V_flux':copy.deepcopy(SPEC[pause]['V'])})
#                SPEC[pause].update({'Y_flux_without_cosmology':copy.deepcopy(SPEC[pause]['Y'][(SPEC[pause]['X']>3340.)])})
#                SPEC[pause]['V']=Astro.Coords.flbda2ABmag(SPEC[pause]['X'],SPEC[pause]['Y'],var=SPEC[pause]['V'])
#                SPEC[pause]['Y']=Astro.Coords.flbda2ABmag(SPEC[pause]['X'],SPEC[pause]['Y'])
#
#            self.dic_AB.update({sn:SPEC})
#
#
#    def Cosmology_corrected(self):
#        self.dic_cosmo={}
#        for i,sn in enumerate(self.sn_name):
#            print '%i/%i'%(((i+1),len(self.sn_name)))
#            SPEC=copy.deepcopy(self.dic_AB[sn])
#            for j,pause in enumerate(self.dic_AB[sn].keys()):
#                print 'processing '+sn+' pause '+pause
#                SPEC[pause]['Y']+= -5.*N.log10(d_l(SPEC[pause]['z_cmb'],SNLS=True))+5.
#                SPEC[pause].update({'Y_flux':go_to_flux(SPEC[pause]['X'],copy.deepcopy(SPEC[pause]['Y']))})    
# 
# 
#            self.dic_cosmo.update({sn:SPEC})
#
#            
#
#    def reorder_and_clean_dico_cosmo(self,SALT2=True):
#        dic_cosmo={}
#        for i,sn in enumerate(self.sn_name):
#            print '%i/%i'%(((i+1),len(self.sn_name)))
#            SPEC=copy.deepcopy(self.dic_cosmo[sn])
#            spec={}
#            Ind=0
#            for j in range(len(self.dic_cosmo[sn].keys())):
#                Min=(10.**23)
#                ind='0'
#                for k,pause in enumerate(SPEC):
#                    phase=SPEC[pause]['phase_salt2']
#                    if phase<Min:
#                        Min=phase
#                        ind=pause
#                if N.sum(N.isfinite(SPEC[ind]['Y']))==len(SPEC[ind]['Y']):
#                    SPEC[ind]['Y']=SPEC[ind]['Y'][(SPEC[ind]['X']>3340.)]
#                    SPEC[ind]['V']=SPEC[ind]['V'][(SPEC[ind]['X']>3340.)]
#                    SPEC[ind]['Y_flux']=SPEC[ind]['Y_flux'][(SPEC[ind]['X']>3340.)]
#                    SPEC[ind]['V_flux']=SPEC[ind]['V_flux'][(SPEC[ind]['X']>3340.)]
#                    SPEC[ind]['X']=SPEC[ind]['X'][(SPEC[ind]['X']>3340.)]
#                    spec.update({'%i'%(Ind):copy.deepcopy(SPEC[ind])})
#                    Ind+=1
#                else:
#                    Delta=abs(N.sum(N.isfinite(SPEC[ind]['Y']))-len(SPEC[ind]['Y']))
#                    if Delta<100:
#                        Filtre=N.isfinite(SPEC[ind]['Y'])
#                        SPLINE=inter.InterpolatedUnivariateSpline(SPEC[ind]['X'][Filtre],SPEC[ind]['Y'][Filtre])
#                        SPEC[ind]['Y'][~Filtre]=SPLINE(SPEC[ind]['X'][~Filtre])
#                        SPEC[ind]['V'][~Filtre]=100.
#                        SPEC[ind]['Y']=SPEC[ind]['Y'][(SPEC[ind]['X']>3340.)]
#                        SPEC[ind]['V']=SPEC[ind]['V'][(SPEC[ind]['X']>3340.)]
#                        SPEC[ind]['Y_flux']=SPEC[ind]['Y_flux'][(SPEC[ind]['X']>3340.)]
#                        SPEC[ind]['V_flux']=SPEC[ind]['V_flux'][(SPEC[ind]['X']>3340.)]
#                        SPEC[ind]['X']=SPEC[ind]['X'][(SPEC[ind]['X']>3340.)]
#                        spec.update({'%i'%(Ind):copy.deepcopy(SPEC[ind])})
#                        Ind+=1
#
#
#                del SPEC[ind]
#
#
#            dic_cosmo.update({sn:spec})
#    
#        self.dic_cosmo=dic_cosmo
#
#        
#    def kill_blue_runaway(self):
#        X=self.dic_cosmo['PTF09dlc']['0']['X']
#        Filtre_UV=N.array([False]*len(X))
#        for i in range(len(Filtre_UV)):
#            if X[i]>3700 and X[i]<3900:
#                Filtre_UV[i]=True
#
#        dic_cosmo={}
#        for i,sn in enumerate(self.sn_name):
#            print '%i/%i'%(((i+1),len(self.sn_name)))
#            SPEC=copy.deepcopy(self.dic_cosmo[sn])
#            spec={}
#            Ind=0
#            for j in range(len(self.dic_cosmo[sn].keys())):
#                x=X[Filtre_UV]
#                y=SPEC['%i'%(j)]['Y'][Filtre_UV]
#                y_err=N.sqrt(SPEC['%i'%(j)]['V'][Filtre_UV])
#                Multi=M.Multilinearfit(x,y,xerr=None,yerr=None,covx=None,Beta00=None) 
#                Multi.Multilinearfit(adddisp=False) 
#
#                if Multi.alpha[0]<0 and N.mean(y)<-10.:
#                    spec.update({'%i'%(Ind):copy.deepcopy(SPEC['%i'%j])})
#                    Ind+=1
#                del SPEC['%i'%(j)]
#            dic_cosmo.update({sn:spec})
#
#        self.dic_cosmo=dic_cosmo
#
#
#    def select_sn_in_good_sample(self,META):
#
#        meta = SnfMetaData.SnfMetaData(META)
#        
#        meta.add_filter(idr__subset__in = ['training','validation'])#,'bad','auxiliary'])
#        sn_name=meta.targets('target.name',sort_by='target.name')
#        dic_cosmo={}
#        for i,sn in enumerate(sn_name):
#            print '%i/%i'%(((i+1),len(sn_name)))
#            SPEC=copy.deepcopy(self.dic_cosmo[sn])
#            #if len(SPEC.keys())>4:
#            #    dic_cosmo.update({sn:SPEC})
#            dic_cosmo.update({sn:SPEC})
#            
#        self.sn_name=dic_cosmo.keys()
#        self.dic_cosmo=dic_cosmo
#
#
#
#    def select_night_in_previous_dico(self,dico_pkl):
# 
#        dic = cPickle.load(open(dico_pkl))
#
#        sn_name=dic.keys()
#        dic_cosmo={}
#        for i,sn in enumerate(self.sn_name):
#            print sn 
#            if sn in sn_name:
#                SPEC=copy.deepcopy(self.dic_cosmo[sn])
#                SPec={}
#                PAUSE=[]
#                IND=0
#                for j in range(len(dic[sn].keys())):
#                    PAUSE.append(dic[sn]['%i'%(j)]['pause'])
#                for j in range(len(SPEC.keys())):
#                    if SPEC['%i'%(j)]['pause'] in PAUSE:
#                        SPec.update({'%i'%(IND):SPEC['%i'%(j)]})
#                        IND+=1
#                dic_cosmo.update({sn:SPec})
#                if len(SPec.keys())!=len(dic[sn].keys()):
#                    print 'ANDALOUSE ma gueule'
#
#        self.sn_name=dic_cosmo.keys()
#        self.dic_cosmo=dic_cosmo
#
#    def kill_Blue_runaway_eye_control(self):
#        X=self.dic_cosmo['PTF09dlc']['0']['X']
#        dic_cosmo={}
#        dic_bad={}
#        self.BAD=True
#        for i,sn in enumerate(self.sn_name):
#            print '%i/%i'%(((i+1),len(self.sn_name)))
#            SPEC=copy.deepcopy(self.dic_cosmo[sn])
#            OFFSET=[]
#
#            P.figure()
#            indice_spec=[]
#            Wave=[]
#            Flux=[]
#            for j in range(len(self.dic_cosmo[sn].keys())):
#                OFFSET.append(j+0.2+15)
#                indice_spec.append(j)
#                Wave.append(SPEC['%i'%(j)]['X'][60])
#                Flux.append(SPEC['%i'%(j)]['Y'][60]+OFFSET[j])
#
#                P.plot(SPEC['%i'%(j)]['X'],SPEC['%i'%(j)]['Y']+OFFSET[j],'k')
#                moins=SPEC['%i'%(j)]['Y']+OFFSET[j]-N.sqrt(SPEC['%i'%(j)]['V'])
#                plus=SPEC['%i'%(j)]['Y']+OFFSET[j]+N.sqrt(SPEC['%i'%(j)]['V'])
#                P.fill_between(SPEC['%i'%(j)]['X'],moins,plus,color='k',alpha=0.5 )
#                
#            indice_spec=N.array(indice_spec)
#            Wave=N.array(Wave)
#            Flux=N.array(Flux)
#            
#            scat=P.scatter(Wave,Flux,s=50,c='b')
#            browser=mpl.PointBrowser_TO_DELATE(Wave,Flux,indice_spec,scat)
#            P.title(sn)
#            P.ylabel('Mag AB + cst')
#            P.xlabel('wavelength [$\AA$]')
#            P.gca().invert_yaxis()
#            
#            P.show()
#            
#
#            if len(browser.LIST_TO_DELATE) == 0 :
#                spec=SPEC
#            
#            else:
#                spec={}
#                spec_bad={}
#                Ind=0
#
#                for j in range(len(self.dic_cosmo[sn].keys())):
#                    if j in browser.LIST_TO_DELATE:
#                        print 'au revoir'
#                        spec_bad.update({SPEC['%i'%j]['pause']:SPEC['%i'%j]})
#                    else:
#                        spec.update({'%i'%(Ind):copy.deepcopy(SPEC['%i'%j])})
#                        Ind+=1
#                    del SPEC['%i'%(j)]
#                dic_bad.update({sn:spec_bad})
#            dic_cosmo.update({sn:spec})
#
#
#        self.dic_cosmo=dic_cosmo
#        self.dic_bad=dic_bad
#
#
#    def Build_without_Cosmology_corrected(self):
#        self.dic_without_cosmo={}
#        for i,sn in enumerate(self.sn_name):
#            print '%i/%i'%(((i+1),len(self.sn_name)))
#            SPEC=copy.deepcopy(self.dic_cosmo[sn])
#            for j,pause in enumerate(self.dic_cosmo[sn].keys()):
#                print 'processing '+sn+' pause '+pause
#                SPEC[pause]['Y']-= -5.*N.log10(d_l(SPEC[pause]['z_cmb'],SNLS=True))+5.
#
#            self.dic_without_cosmo.update({sn:SPEC})
#
#        
#    def write_pkl(self,pkl_name):
#        
#        File=open(pkl_name,'w')
#        cPickle.dump(self.dic_cosmo,File)
#        File.close()
#
#        if self.BAD:
#            File=open('/sps/snovae/user/leget/ALLAIRE/bad_spectra.pkl','w')
#            cPickle.dump(self.dic_bad,File)
#            File.close()
#
#    def write_without_cosmo_pkl(self,pkl_name):
#        
#        File=open(pkl_name,'w')
#        cPickle.dump(self.dic_without_cosmo,File)
#        File.close()
#
#
#class build_at_max_data:
#
#    def __init__(self,dico_cosmo,phrenology):
#
#        self.Token_list=['EWCaIIHK','EWSiII4000','EWMgII','EWFe4800','EWSIIW','EWSiII5972','EWSiII6355','EWOI7773','EWCaIIIR','vSiII_4128_lbd','vSiII_5454_lbd','vSiII_5640_lbd','vSiII_6355_lbd']
#        self.dic_cosmo=cPickle.load(open(dico_cosmo))
#        self.phrenology=cPickle.load(open(phrenology))
#        self.sn_name=self.dic_cosmo.keys()
#        
#    def select_spectra_at_max(self,RANGE=[-2.5,2.5]):
#        self.dic_cosmo_at_max={}
#
#        for i,sn in enumerate(self.sn_name):
#            print '%i/%i'%(((i+1),len(self.sn_name)))
#            SPEC=copy.deepcopy(self.dic_cosmo[sn])
#            PPhase=-999
#            for j,pause in enumerate(self.dic_cosmo[sn].keys()):
#                if abs(SPEC[pause]['phase_salt2'])<abs(PPhase):
#                    spec=SPEC[pause]
#                    PPhase=SPEC[pause]['phase_salt2']
#            if PPhase>RANGE[0] and PPhase<RANGE[1]:
#                self.dic_cosmo_at_max.update({sn:spec})
#        self.sn_name=self.dic_cosmo_at_max.keys()
#
#    def select_spectral_indicators(self):
#
#        dic_cosmo_at_max={}
#
#        for i,sn in enumerate(self.sn_name):
#            print '%i/%i'%(((i+1),len(self.sn_name)))
#            SPEC=copy.deepcopy(self.dic_cosmo_at_max[sn])
#            spectral_indicators=[]
#            spectral_indicators_error=[]
#            for SI in range(len(self.Token_list)):
#                PAUSE=SPEC['pause']
#                spectral_indicators.append(self.phrenology[sn]['spectra'][PAUSE]['phrenology.'+self.Token_list[SI]])
#                spectral_indicators_error.append(self.phrenology[sn]['spectra'][PAUSE]['phrenology.'+self.Token_list[SI]+'.err'])
#            dic_sn={'spectra':SPEC,
#                    'spectral_indicators':N.array(spectral_indicators),
#                    'spectral_indicators_error':N.array(spectral_indicators_error)}
#
#            dic_cosmo_at_max.update({sn:dic_sn})
#        
#        self.dic_cosmo_at_max=dic_cosmo_at_max
#
#        
#    def select_sn_list(self,sn_name):       
#
#        dic_cosmo_at_max={}
#        for i,sn in enumerate(sn_name):
#            if sn in self.sn_name:
#                print '%i/%i'%(((i+1),len(sn_name)))
#                SPEC=copy.deepcopy(self.dic_cosmo_at_max[sn])
#                dic_cosmo_at_max.update({sn:SPEC})
#
#        self.sn_name=dic_cosmo_at_max.keys()
#        self.dic_cosmo_at_max=dic_cosmo_at_max 
#
#
#    def select_sn_in_good_sample(self,META,SAMPLE=['training']):
#
#        meta = SnfMetaData.SnfMetaData(META)
#
#        meta.add_filter(idr__subset__in = SAMPLE)
#        sn_name=meta.targets('target.name',sort_by='target.name')
#        dic_cosmo_at_max={}
#        for i,sn in enumerate(sn_name):
#            if sn in self.sn_name:
#                print '%i/%i'%(((i+1),len(sn_name)))
#                SPEC=copy.deepcopy(self.dic_cosmo_at_max[sn])
#                dic_cosmo_at_max.update({sn:SPEC})
#
#        self.sn_name=dic_cosmo_at_max.keys()
#        self.dic_cosmo_at_max=dic_cosmo_at_max
#
#    def write_pkl(self,pkl_name):
#
#        File=open(pkl_name,'w')
#        cPickle.dump(self.dic_cosmo_at_max,File)
#        File.close()
#
if __name__=="__main__":

    bsd = build_spectral_data('data_input/SNF-0203-CABALLOv2/META.pkl')
