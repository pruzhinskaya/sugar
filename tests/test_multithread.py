"""test multithread in python."""
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
import time


class comp_chi2:

    def __init__(self,proc=20):

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



def doubler(number):
    A = np.dot(np.ones(1000),np.ones((1000,1000)))
    return 0

if __name__ == '__main__':

    numbers = np.linspace(0,5,100) 

    test = []
    cc = comp_chi2(proc = 10)
    cc.comp_chi2()
    test.append(cc.test2)

    cc = comp_chi2(proc = 20)
    cc.comp_chi2()
    test.append(cc.test2)

    cc = comp_chi2(proc = 30)
    cc.comp_chi2()
    test.append(cc.test2)


