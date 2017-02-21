import pylab as P
import numpy as N
import cPickle
import emfa_analysis
import sys,os,optparse

def read_option():

    usage = "usage: [%prog] -p pca_input -s spectra_input -m model_output [otheroptions]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--Data_at_max","-d",dest="Data_at_max",help="data at max file",default=None)
    parser.add_option("--bin","-b",dest="bin",help="bin of the supernova",default=None)
    parser.add_option("--directoryOUT","-D",dest="directory_output",help="directory where you will put the jacknife_solution",default=None)
    parser.add_option("--sigma_clipping","-S",dest="sigma_clipping",help="sigma clipping or not ",default=False,action="store_true")
    option,args = parser.parse_args()


    return option


def jacknife_emfa():
    
    option=read_option()
    fa=emfa_analysis.EMFA_SI_analysis(option.Data_at_max)
    fa.center()
    if option.sigma_clipping:
        fa.iterative_filter(Chi2empca=True)
    fa.filter[int(option.bin)]=False
    fa.EM_FA(option.directory_output+'emfa_without_'+fa.sn_name[int(option.bin)]+'.pkl')


class plot_jackknife_result:

    def __init__(self,rep_output,emfa_at_max,Filter=True):

        self.dic_at_max=cPickle.load(open(emfa_at_max))
        self.vec_jack=[]
        self.val_jack=[]
        self.sn_name=self.dic_at_max['sn_name']
        if Filter:
            FF=self.dic_at_max['filter']
        else:
            FF=N.array([True]*len(self.sn_name))

        self.sn_name=N.array(self.sn_name)[FF]


        for i,sn in enumerate(self.sn_name):

            print sn+' %i/%i'%((i+1,len(self.sn_name)))
            dic=cPickle.load(open(rep_output+'emfa_without_'+sn+'.pkl'))
            self.vec_jack.append(dic['vec'])
            self.val_jack.append(dic['val'])

        self.vec_jack=N.array(self.vec_jack)
        self.val_jack=N.array(self.val_jack)



    def Plot_result(self):

        TT=1
        nsil =  ['pEWCa II H&K', r'pEWSi II $\lambda$4131', 'pEWMg II',
                 'pEWFe $\lambda$4800', 'pEWS II W', 'pEWSi II $\lambda$5972',
                 'pEWSi II $\lambda$6355', 'pEWO I $\lambda$7773', 'pEWCa II IR',
                 'VSi II $\lambda$4131','VWS II $\lambda$5454','VWS II $\lambda$5640',
                 'VSi II $\lambda$6355']

        for vec in range(13): 
            if vec%5==0:
                P.figure(figsize=(15,9))
                P.subplots_adjust(hspace = 0.01,top=0.85)
                TT=1

            P.subplot(5,1,TT)
            P.plot(N.zeros(13),'k',linewidth=2)
            for i in range(len(self.sn_name)):
                
                if self.dic_at_max['vec'][:,vec][0]*self.vec_jack[i][:,vec][0]>0:
                    if i==0:
                        P.plot(self.vec_jack[i][:,vec],'b',label='PC %i'%(vec+1))
                    else:
                        P.plot(self.vec_jack[i][:,vec],'b')
                else:
                    if i==0:
                        P.plot(-self.vec_jack[i][:,vec],'b',label='PC %i'%(vec+1))
                    else:
                        P.plot(-self.vec_jack[i][:,vec],'b')

            if TT==5 or (vec>10 and TT%5==3) :
                rotation=45
                xstart, xplus, ystart = -0.004, 0.082 ,5.01 
                xx=[0,0,0,0,0,0,0,0,0,0,0,0,0]
                if vec>10 and TT%5==3:
                    ystart-=2
                
                x = xstart
                toto=1
                for leg in nsil:
                    P.annotate(leg, (x+xx[toto-1],ystart), xycoords='axes fraction',
                               size='large', rotation=rotation, ha='left', va='bottom')
                    #if toto==12:
                    #    toto+=1
                    #    x+= 0.06
                    #else:
                    toto+=1
                    x+= xplus



            P.yticks([-0.5,0,0.5],['-0.5','0','0.5'])
            P.ylim(-1,1)


            P.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],['', '', '', '','','', '', '', '', '','','',''])   
            P.legend(loc=4)
            TT+=1


    def Plot_result_error(self):

        vec_jack=N.zeros(N.shape(self.vec_jack))
        
        for i in range(len(self.sn_name)):
            for vec in range(13):
                if self.dic_at_max['vec'][:,vec][0]*self.vec_jack[i][:,vec][0]>0:
                    vec_jack[i][:,vec]=self.vec_jack[i][:,vec]
                else:
                    vec_jack[i][:,vec]=-self.vec_jack[i][:,vec]

        self.vec_err=N.zeros((13,13))
        
        for vec in range(13):
            self.vec_err[:,vec]=((len(self.sn_name)-1.)/len(self.sn_name))*(N.sum((vec_jack[:,:,vec]-self.dic_at_max['vec'][:,vec])**2,axis=0))

        self.vec_err=N.sqrt(self.vec_err)

        TT=1
        nsil =  ['pEWCa II H&K', r'pEWSi II $\lambda$4131', 'pEWMg II',
                 'pEWFe $\lambda$4800', 'pEWS II W', 'pEWSi II $\lambda$5972',
                 'pEWSi II $\lambda$6355', 'pEWO I $\lambda$7773', 'pEWCa II IR',
                 'VSi II $\lambda$4131','VWS II $\lambda$5454','VWS II $\lambda$5640',
                 'VSi II $\lambda$6355']

        for vec in range(13):
            if vec%5==0:
                P.figure(figsize=(15,15))
                P.subplots_adjust(hspace = 0.01,top=0.85)
                TT=1

            P.subplot(5,1,TT)
            P.plot(N.zeros(13),'k',linewidth=2)
            P.fill_between(N.linspace(0,12,13),self.dic_at_max['vec'][:,vec]-self.vec_err[:,vec],self.dic_at_max['vec'][:,vec]+self.vec_err[:,vec],color='b',alpha=0.5)
            #P.errorbar(N.linspace(0,12,13),self.dic_at_max['vec'][:,vec], linestyle='', xerr=None,yerr=self.vec_err[:,vec],ecolor='blue',alpha=1.,marker='.',zorder=0)
            P.plot(self.dic_at_max['vec'][:,vec],'b',label='$\Lambda_{%i}$'%(vec+1))

            if TT==5 or (vec>10 and TT%5==3) :
                rotation=45
                xstart, xplus, ystart = -0.004, 0.082 ,5.01
                xx=[0,0,0,0,0,0,0,0,0,0,0,0,0]
                if vec>10 and TT%5==3:
                    ystart-=2

                x = xstart
                toto=1
                for leg in nsil:
                    P.annotate(leg, (x+xx[toto-1],ystart), xycoords='axes fraction',
                               size='large', rotation=rotation, ha='left', va='bottom')

                    toto+=1
                    x+= xplus

            TT+=1
            P.legend(loc=4)
            P.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],['', '', '', '','','', '', '', '', '','','',''])
            P.yticks([-0.5,0,0.5],['-0.5','0','0.5'])
            P.ylim(-1,1)


if __name__=="__main__":

    #jacknife_emfa()
    #PJR=plot_jackknife_result('/sps/snovae/user/leget/CABALLO/SUGAR_validation/emfa_jacknife_with_sn_reject/','/sps/snovae/user/leget/CABALLO/SUGAR_validation/emfa_without_filter_CABALLO.pkl',Filter=False)
    PJR=plot_jackknife_result('/sps/snovae/user/leget/CABALLO/SUGAR_validation/emfa_jacknife_without_sn_reject/','/sps/snovae/user/leget/CABALLO/SUGAR_validation/emfa_CABALLO.pkl',Filter=True)
    PJR.Plot_result_error()
    #PJR.Plot_result()
                 
    
