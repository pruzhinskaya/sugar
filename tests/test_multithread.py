#from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
import time


class comp_chi2:

    def __init__(self):

        self.chi2 = np.zeros(50)
        self.loop = np.linspace(0,49,50).astype(int)
        self.pool = Pool(processes=20)
        self.test1 = []
        self.test2 = []


    def comp_chi2(self):
        NN=3000
        def _comp_chi2(integer):
            A = np.dot(np.ones(NN),np.dot(np.ones((NN,NN)),np.matrix(np.ones(NN)).T))
            self.chi2[integer] = A[0,0]


        for j in range(100):
            print j 
            A = time.time()
            for i in self.loop:
                _comp_chi2(i)
            B=time.time()
            self.test1.append(B-A)
            
        for j in range(100):
            print j
            A = time.time()
            self.pool.map(_comp_chi2, self.loop)
            B=time.time()
            self.test2.append(B-A)



def doubler(number):
    A = np.dot(np.ones(1000),np.ones((1000,1000)))
    return 0

if __name__ == '__main__':

    numbers = np.linspace(0,5,100) 

    cc = comp_chi2()
    cc.comp_chi2()

    #pool = Pool(processes=20)

