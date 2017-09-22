"""test multithread in python."""
from multiprocessing.pool import ThreadPool as Pool
from test_sed_fitting import generate_fake_sed
from scipy.sparse import block_diag
from scipy.sparse import coo_matrix
import numpy as np
import time
import copy
import sugar
from scipy import linalg

class comp_chi2:

    def __init__(self,proc=190):

        self.chi2 = np.zeros(50)
        self.loop = np.linspace(0,49,50).astype(int)
        self.pool = Pool(processes=proc)
        self.test2 = []

    def comp_chi2(self):
        NN=3000
        def _comp_chi2(integer):
            A = np.dot(np.ones(NN),np.dot(np.ones((NN,NN)),np.matrix(np.ones(NN)).T))
            self.chi2[integer] = A[0,0]

        for j in range(20):
            print j
            A = time.time()
            self.pool.map(_comp_chi2, self.loop)
            B=time.time()
            self.test2.append(B-A)

class m_step_block_diag:

    def __init__(self,proc=20):

        self.nsn = 105
        self.y, cy, x, x_err, wave, alpha_truth, grey_truth = generate_fake_sed(self.nsn,plot=False,bloc=True,size_bloc=19,nbin=190*19)
        self.size_bloc = 19 
        self.nbin = 190*19
        self.A = np.zeros((self.nbin,len(x[0])+2))
        self.h = np.ones((self.nsn,len(x[0])+2))
        self.h[:,2:] = x
        self.filter_grey = np.array([True,False,True,True,True])
        self.fit_grey=True
        self.sparse=True
        self.nslopes=4

        self.pool = Pool(processes=proc)
        self.loop = np.linspace(0,(self.nbin/self.size_bloc)-1,(self.nbin/self.size_bloc)).astype(int)
        
        self.wy = []
        self.wwy = []
        for sn in range(self.nsn):
            w = []
            ww = []
            for i in range(self.nbin/self.size_bloc):
                a = np.linalg.inv(sugar.sed_fitting.extract_block_diag(cy[sn],self.size_bloc,i))
                w.append(coo_matrix(a))
                ww.append(a)
            self.wy.append(block_diag(w))
            self.wwy.append(ww)
        #self.loop = np.linspace(0,49,50).astype(int)
        #self.pool = Pool(processes=proc)
        #self.test2 = []

    def _m_step_bloc_diagonal(self):

        if self.fit_grey:
            Y = copy.deepcopy(self.y) - (self.h[:,1]*np.ones(np.shape(self.y)).T).T
        else:
            Y = copy.deepcopy(self.y)

        new_slopes = np.zeros(np.shape(self.A[:,self.filter_grey]))

        h_ht_dict = {}
        
        for i in range(self.nbin/self.size_bloc):
            for sn in range(self.nsn):
                h_ht=np.dot(np.matrix(self.h[sn,self.filter_grey]).T,np.matrix(self.h[sn,self.filter_grey]))
                
                if self.sparse:
                    W_sub=sugar.sed_fitting.extract_block_diag(self.wy[sn],self.size_bloc,i)
                else:
                    W_sub = self.wy[sn][i*self.size_bloc:(i+1)*self.size_bloc, i*self.size_bloc:(i+1)*self.size_bloc]
                
                hh_kron_W_sub = linalg.kron(h_ht, W_sub)
                WYh = np.dot(W_sub, np.dot(np.matrix(Y[sn,i*self.size_bloc:(i+1)*self.size_bloc]).T,
                                           np.matrix(self.h[sn,self.filter_grey])))

                if sn == 0:
                    hh_kron_W_sum = np.copy(hh_kron_W_sub)
                    WYh_sum = np.copy(WYh)
                else:
                    hh_kron_W_sum += hh_kron_W_sub
                    WYh_sum += WYh

            h_ht_dict[i] = [hh_kron_W_sum, WYh_sum]
                    
        for wl in h_ht_dict.keys():
            hh_kron_W_sum, W_sum = h_ht_dict[wl]
            sum_WYh_vector = np.zeros(self.size_bloc*self.nslopes)
            for i in xrange(self.nslopes):   
                sum_WYh_vector[i*self.size_bloc:][:self.size_bloc]=W_sum[:,i].ravel()

            X_cho = linalg.cho_factor(hh_kron_W_sum)
            slopes_solve = linalg.cho_solve(X_cho, sum_WYh_vector)
            for i in xrange(self.nslopes):
                new_slopes[wl*self.size_bloc:(wl+1)*self.size_bloc,i] = slopes_solve[i*self.size_bloc:(i+1)*self.size_bloc]
            
        self.A[:,self.filter_grey]=new_slopes
        return new_slopes


    def _m_step_bloc_diagonal_mult(self):

        if self.fit_grey:
            Y = copy.deepcopy(self.y) - (self.h[:,1]*np.ones(np.shape(self.y)).T).T
        else:
            Y = copy.deepcopy(self.y)

        new_slopes = np.zeros(np.shape(self.A[:,self.filter_grey]))

        h_ht_dict = {}

        hh_kron_W_sum = np.zeros((self.nbin/self.size_bloc,self.size_bloc*self.nslopes,self.size_bloc*self.nslopes))
        WYh_sum = np.zeros((self.nbin/self.size_bloc,self.size_bloc,self.nslopes))

        for i in range(self.nbin/self.size_bloc):
            for sn in range(self.nsn):
                h_ht=np.dot(np.matrix(self.h[sn,self.filter_grey]).T,np.matrix(self.h[sn,self.filter_grey]))
                
                if self.sparse:
                    W_sub = self.wwy[sn][i]
                else:
                    W_sub = self.wy[sn][i*self.size_bloc:(i+1)*self.size_bloc, i*self.size_bloc:(i+1)*self.size_bloc]
                
                hh_kron_W_sum[i] += linalg.kron(h_ht, W_sub)
                WYh_sum[i] += np.dot(W_sub, np.dot(np.matrix(Y[sn,i*self.size_bloc:(i+1)*self.size_bloc]).T,
                                                   np.matrix(self.h[sn,self.filter_grey])))

            sum_WYh_vector = np.zeros(self.size_bloc*self.nslopes)
            for j in xrange(self.nslopes):   
                sum_WYh_vector[j*self.size_bloc:][:self.size_bloc]=WYh_sum[i][:,j].ravel()

            X_cho = linalg.cho_factor(hh_kron_W_sum[i])
            slopes_solve = linalg.cho_solve(X_cho, sum_WYh_vector)
            for j in xrange(self.nslopes):
                new_slopes[i*self.size_bloc:(i+1)*self.size_bloc,j] = slopes_solve[j*self.size_bloc:(j+1)*self.size_bloc]
                
        self.A[:,self.filter_grey]=new_slopes

        return new_slopes


if __name__ == '__main__':

    M=m_step_block_diag(proc=20)
    #A = M._m_step_bloc_diagonal()
    #B = M._m_step_bloc_diagonal_mult()

    test1=[]
    test2=[]
    for i in range(5):
        print i
        start = time.time()
        A = M._m_step_bloc_diagonal()
        end = time.time()
        test1.append(end-start)
    for i in range(5):
        print i
        start = time.time()
        B = M._m_step_bloc_diagonal_mult()
        end = time.time()
        test2.append(end-start)
