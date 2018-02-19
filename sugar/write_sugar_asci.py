import cPickle
import os
import numpy as N
import sugar

path = os.path.dirname(sugar.__file__)
sed = path + '/data_output/sugar_model.pkl'
max_light = path + '/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_save_before_PCA.pkl'

dic = cPickle.load(open(sed))
dic_at_max = cPickle.load(open(max_light))
fichier = open(path + '/data_output/SUGAR_model_v1.asci','w') 

Time = N.linspace(-12,42,19)

for Bin in range(len(dic['m0'])):
    fichier.write('%.5f    %.5f    %.5f    %.5f    %.5f    %.5f    %.5f    %.5f \n'%((Time[Bin%19],
                                                                                      dic['X'][Bin],
                                                                                      dic['m0'][Bin],
                                                                                      dic['alpha'][Bin,0],
                                                                                      dic['alpha'][Bin,1],
                                                                                      dic['alpha'][Bin,2],
                                                                                      sugar.extinctionLaw(dic['X'][Bin],Rv=dic_at_max['RV']),
                                                                                      1)))


fichier.close()
