import numpy as N
import cPickle 
import copy 
from ToolBox import EMfa as EM_manu

class EMFA_SI_analysis:
    def __init__(self,data_at_max,Missing_data=True):

        dic=cPickle.load(open(data_at_max))
        self.dic_at_max=dic
        sn_name=dic.keys()
        self.number_si=len(dic[sn_name[0]]['spectral_indicators'])
        indicator_data=N.zeros((len(sn_name),self.number_si))
        indicator_error=N.zeros((len(sn_name),self.number_si))

        for i in range(len(sn_name)):
            indicator_data[i]=dic[sn_name[i]]['spectral_indicators']
            indicator_error[i]=dic[sn_name[i]]['spectral_indicators_error']

        if Missing_data:
            error=indicator_error[:,N.all(N.isfinite(indicator_data),axis=0)]
            data=indicator_data[:,N.all(N.isfinite(indicator_data),axis=0)]

            self.sn_name=sn_name
            self.error=indicator_error
            self.data=indicator_data
            self.filter=N.array([True]*len(self.data))

            for sn in range(len(self.sn_name)):
                for SI in range(self.number_si):
                    if not N.isfinite(self.data[sn,SI]):
                        self.data[sn,SI]=N.average(data[:,SI],weights=1./error[:,SI]**2)
                        self.error[sn,SI]=10**8
        else:
            # we assume that catching all nan in the data set will also catch all nans for the error set
            self.sn_name=sn_name[N.all(N.isfinite(indicator_data),axis=0)]
            self.error=indicator_error[:,N.all(N.isfinite(indicator_data),axis=0)]
            self.data=indicator_data[:,N.all(N.isfinite(indicator_data),axis=0)]
            self.filter=N.array([True]*len(self.data))



    def center(self,wnorm=True):
        # input data and error are left untouched
        # filt will be used to select the data allowed in order to perform the mean and variance
        # it will also compute the covariance for future usage
        # if wnorm is set, weights will be used

        # center
        if wnorm:
            center = N.average(self.data[self.filter].T,axis=1,weights=1./self.error[self.filter].T**2)
        else:
            center = N.mean(self.data[self.filter].T,axis=1) 
        self.data_center = ( self.data - center.T )
        self.error_center =  self.error
        self.Covar=1./(N.sum(self.filter)-1) * N.dot(self.data_center[self.filter].T,self.data_center[self.filter])
        
        # after normalization chi2 is not any longer correct
        if hasattr(self,'chi2'):
            del self.chi2

    def Norm_varr(self,Filter=None):
        if Filter != None :
            norm = N.sqrt(N.var(self.data_center[Filter].T,axis=1))
        else:
            norm = N.sqrt(N.var(self.data_center.T,axis=1))

        self.norm=norm

        return norm
    
 

    def filter_and_prepare_data(self,chi2empca=False):
        # this is a high level method is order to update the filter
        # ensure the data is normalize in input

        self.center()
        norm=self.Norm_varr(self.filter)
        
        dat=(self.data_center[self.filter] /norm)
        err=(self.error_center[self.filter]/norm)
        DAT=(self.data_center /norm)
        ERR=(self.error_center/norm)

        if chi2empca != False:
            
            emfa=EM_manu.EMPCA(dat,1./err**2)
            emfa.converge(13,niter=150,center=True)
            new_varr=N.dot(emfa.Lambda,emfa.Lambda.T)
            chi2=N.zeros(len(self.data_center)) 
        
            for sn in range(len(self.data_center)):
                chi2[sn]=N.dot(N.dot(DAT[sn],N.linalg.inv(new_varr+N.diag(ERR[sn]**2))),DAT[sn].reshape((len(DAT[sn]),1)))[0]
                self.chi2_empca=chi2
               
        else:
            Covar=1./(N.sum(self.filter)-1) * N.dot(dat.T,dat)
            chi2=N.diag(N.dot(N.dot(DAT,N.linalg.inv(Covar)),DAT.T))
            self.chi2_pca=chi2
               
        
        
        ndf = len(self.data_center[0])
        
        self.filter = self.filter & (chi2<ndf + 3 * N.sqrt(2*ndf))
        # now reset norm and covar
        self.center()

        

    def iterative_filter(self,Chi2empca=False,MFR_issue=False):

        #iteration pour rester dans mes contours a 3 sigmas
        a=[sum(self.filter)+1,sum(self.filter)+2]
        i=0
        
        while a[i+1]!=a[i]:
            self.filter_and_prepare_data(chi2empca=Chi2empca)
 
            a.append(sum(self.filter))
            i +=1


        self.DAYS=[]
        for i in range(len(self.filter)):
            self.DAYS.append(self.dic_at_max[self.sn_name[i]]['spectra']['days'])
            if MFR_issue:
                if self.dic_at_max[self.sn_name[i]]['spectra']['days']>55250.:
                    self.filter[i]=False

         

    def BIC_number_eigenvector(self):

        self.BIC=N.zeros(len(self.data_center[0]))
        self.Log_L=N.zeros(len(self.data_center[0]))
        self.chi2_Tot=N.zeros(len(self.data_center[0]))

        dat=(self.data_center[self.filter]/self.Norm_varr(self.filter))
        err=(self.error_center[self.filter]/self.Norm_varr(self.filter))

        for i in range(len(self.data_center[0])):
            EMPCA=EM_manu.EMPCA(dat,1./err**2)
            EMPCA.converge(i+1,niter=500)
            self.Log_L[i],self.chi2_Tot[i]=EMPCA.log_likelihood(z_solved=True)

            #self.BIC[i]=-2*self.Log_L[i]+(sum(self.filter)+13+((i+1)*(2*13-i-1+1))/2.)*N.log(sum(self.filter))


    def EM_FA(self,pkl_file,BIC=False):
        
        dat=(self.data_center[self.filter]/self.Norm_varr(self.filter))
        err=(self.error_center[self.filter]/self.Norm_varr(self.filter))

        emfa=EM_manu.EMPCA(dat,1./err**2)
        emfa.converge(len(self.data_center[0]),niter=500)
        varr=N.dot(emfa.Lambda,emfa.Lambda.T)

        val,vec= N.linalg.eig(varr)
        self.EM_FA_Cov=varr
        self.val=val.real
        self.vec=vec.real

        self.Norm_data=(self.data_center/self.Norm_varr(self.filter))
        self.Norm_err=(self.error_center/self.Norm_varr(self.filter))

        if BIC:
            self.BIC_number_eigenvector()
        toto=self.__dict__
        File=open(pkl_file,'w')
        cPickle.dump(toto,File)
        File.close()


if __name__=='__main__':

    
    #fa=EMFA_SI_analysis('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl')
    fa=EMFA_SI_analysis('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI_test_RV.pkl')
    fa.error[~(fa.error >= 0)]=10.**8
    
    dic_emfa=cPickle.load(open('/sps/snovae/user/leget/CABALLO/SUGAR_validation/emfa_CABALLO_without_MFR_issue.pkl'))

    fa.filter=N.array([True]*len(fa.sn_name))
    for i in range(len(fa.sn_name)):
        if fa.sn_name[i] in dic_emfa['sn_name']:
            print 'pouet'
        else:
            fa.filter[i]=False


    fa.center()
    fa.iterative_filter(Chi2empca=True,MFR_issue=True)
    fa.EM_FA('/sps/snovae/user/leget/CABALLO/SUGAR_validation/emfa_CABALLO_with_sn_nico.pkl',BIC=False)
    #fa.EM_FA('/sps/snovae/user/leget/CABALLO/SUGAR_validation/emfa_without_filter_CABALLO.pkl',BIC=True)


    sn_nico_2011=N.array(['SN2004ef', 'SNF20050728-006','SN2005hc', 'SNF20060511-014',
                          'SNF20060526-003', 'SNF20060621-015', 'SN2006dm', 'SN2006do',
                          'SNF20060911-014', 'SNF20060919-007', 'SNF20061020-000',
                          'SNF20061021-003', 'SNF20061024-000', 'SNF20061108-004',
                          'SNF20061111-002', 'SN2006ob', 'SNF20070403-001', 'SN2007bd',
                          'SNF20070424-003', 'SNF20070427-001', 'SNF20070506-006',
                          'SNF20070531-011', 'SN2007cq', 'SNF20070630-006', 'SNF20070701-005',
                          'SNF20070712-000', 'SNF20070712-003', 'SNF20070717-003',
                          'SNF20070725-001', 'SNF20070727-016', 'SNF20070802-000',
                          'SNF20070803-005', 'SNF20070806-026', 'SNF20070810-004',
                          'SNF20070818-001', 'SNF20070817-003', 'SN2007kk', 'SNF20071015-000',
                          'SN2007le', 'SNF20071021-000', 'SN2007nq', 'SNF20080323-009',
                          'SNF20080510-001', 'SNF20080512-010', 'SNF20080514-002',
                          'SNF20080516-000', 'SNF20080516-022', 'SNF20080522-000',
                          'SNF20080522-011', 'SNF20080531-000', 'SNF20080610-000',
                          'SNF20080612-003', 'SNF20080614-010', 'SNF20080623-001',
                          'SNF20080626-002', 'SNF20080714-008', 'SN2008ec', 'SNF20080720-001',
                          'SNF20080802-006', 'SNF20080803-000', 'SNF20080810-001',
                          'SNF20080822-005', 'SNF20080825-010', 'SNF20080909-030',
                          'SNF20080913-031', 'SNF20080914-001', 'SNF20080918-000',
                          'SNF20080918-002', 'SNF20080918-004', 'SNF20080919-001',
                          'SNF20080919-002', 'SNF20080920-000', 'PTF09dlc', 'PTF09dnp',
                          'PTF09fox', 'PTF09foz'])

    fa.filter=N.array([True]*len(fa.sn_name))
    for i in range(len(fa.sn_name)):
        if fa.sn_name[i] in sn_nico_2011:
            print 'pouet'
        else:
            fa.filter[i]=False

    fa.vec=dic_emfa['vec']
    toto=fa.__dict__
    File=open('/sps/snovae/user/leget/CABALLO/SUGAR_validation/emfa_CABALLO_with_sn_nico.pkl','w')
    cPickle.dump(toto,File)
    File.close()
