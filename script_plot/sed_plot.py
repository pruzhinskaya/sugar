import matplotlib.pyplot as P
import numpy as N
import cPickle
import scipy.interpolate as inter
from ToolBox import MPL
from ToolBox import Astro
from ToolBox import Statistics 
import copy
from matplotlib.patches import Ellipse
from ToolBox.Signal import loess
from ToolBox.Plots import scatterPlot as SP
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from ToolBox import MPL
from matplotlib.widgets import Slider, Button, RadioButtons
import sys,os,optparse     
from scipy.stats import norm as NORMAL_LAW



class SUGAR_plot:

    def __init__(self,dico_hubble_fit):
                
        dico = cPickle.load(open(dico_hubble_fit))
        #self.key=dico['key']
        self.alpha=dico['alpha']
        self.M0=dico['m0']
        self.X=dico['X']
        self.data = dico['h']


    def plot_spectrophtometric_effec_time(self,comp=0):

        reorder = N.arange(190*19).reshape(190, 19).T.reshape(-1)
        X=self.X[reorder]
        M0=self.M0[reorder]
        ALPHA=self.alpha[:,comp][reorder]
     
        CST=N.mean(M0)
        fig,ax1=P.subplots(figsize=(7,8))
        P.subplots_adjust(left=0.1, right=0.85,bottom=0.1,top=0.99)
        Time=N.linspace(-12,42,19)
        Y2_label=[]
        Y2_pos=[]
        for i in range(19):
            
            if i%2==0:
                if (-12+(3*i))!=0:
                    Y2_label.append('%i days'%(Time[i]))
                else:
                    Y2_label.append('%i day'%(Time[i]))

                Y2_pos.append(M0[i*190:(i+1)*190][-1]-CST)

                ax1.plot(X[i*190:(i+1)*190],M0[i*190:(i+1)*190]-CST,'b')

                
                y_moins=M0[i*190:(i+1)*190]-ALPHA[i*190:(i+1)*190]*(N.mean(self.data[:,comp])+N.sqrt(N.var(self.data[:,comp])))
                y_plus=M0[i*190:(i+1)*190]+ALPHA[i*190:(i+1)*190]*(N.mean(self.data[:,comp])+N.sqrt(N.var(self.data[:,comp])))

                ax1.fill_between(X[i*190:(i+1)*190],M0[i*190:(i+1)*190]-CST,y_plus-CST,color='m',alpha=0.7 )
                ax1.fill_between(X[i*190:(i+1)*190],M0[i*190:(i+1)*190]-CST,y_moins-CST,color='g',alpha=0.7)

                if i==0:
                    CST-=2.2
                else:
                    CST-=1


        ax1.set_ylabel('$M_0(t,\lambda)$ + cst',fontsize=20)
        ax1.set_xlim(3300,8620)
        ax1.set_ylim(-1,14.8)
        ax1.set_xlabel('wavelength $[\AA]$',fontsize=20)
        p1 = P.Rectangle((0, 0), 1, 1, fc="magenta")
        p2 = P.Rectangle((0, 0), 1, 1, fc="green")
        ax1.legend([p1, p2], ['+1$\sigma_{q_%i}$'%(comp+1), '-1$\sigma_{q_%i}$'%(comp+1)],loc=4)
        P.gca().invert_yaxis()

        ax2=ax1.twinx()
        ax2.set_ylim(-1,14.8)
        ax2.yaxis.set_ticks(Y2_pos)
        ax2.yaxis.set_ticklabels(Y2_label,fontsize=15)
        ax2.set_ylim(ax2.get_ylim()[::-1])


if __name__=='__main__':

    SED=SUGAR_plot('../sugar/data_output/sugar_model.pkl')
    SED.plot_spectrophtometric_effec_time(comp=0)
    SED.plot_spectrophtometric_effec_time(comp=1)
    SED.plot_spectrophtometric_effec_time(comp=2)

