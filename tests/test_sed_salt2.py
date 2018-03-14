import numpy as np
import sncosmo
import pylab as plt
import cPickle
import copy

source_salt24 = sncosmo.SALT2Source(modeldir='../../2-4-0/data/salt2-4/')
model = sncosmo.Model(source=source_salt24)



for_pf_bis = cPickle.load(open('../sugar/data_input/spectra_snia.pkl'))
#for_pf_bis = cPickle.load(open('../sugar/data_input/File_for_PF.pkl'))
meta = cPickle.load(open('../sugar/data_input/SNF-0203-CABALLOv2/META.pkl'))

for sn in for_pf_bis.keys():
    #sn = 'SN2006cj'
    #phase='0'
    print sn 
    model.set(z=meta[sn]['host.zhelio'], t0=0.,
              x0=meta[sn]['salt2.X0'],
              x1=meta[sn]['salt2.X1'],
              c=meta[sn]['salt2.Color'])
    wave = copy.deepcopy(for_pf_bis[sn]['0']['X'])*(1+meta[sn]['host.zhelio'])
    for phase in for_pf_bis[sn].keys():
        #for_pf_bis[sn][phase].update({'PF_flux':model.flux(for_pf_bis[sn][phase]['phase_salt2'],wave)*(1+meta[sn]['host.zhelio'])**3})
        for_pf_bis[sn][phase].update({'PF_flux':model.flux(for_pf_bis[sn][phase]['phase_salt2'],wave)*(1+meta[sn]['host.zhelio'])**2})
        

File = open('../sugar/data_input/file_pf_bis.pkl','w')
cPickle.dump(for_pf_bis,File)
File.close()

#plt.plot(for_pf_bis[sn][phase]['X'],for_pf_bis[sn][phase]['Y_flux_without_cosmology'],'b',linewidth=3)
#plt.plot(for_pf_bis[sn][phase]['X'],model.flux(for_pf_bis[sn][phase]['phase_salt2'],wave)*(1+meta[sn]['host.zhelio'])**2,'r',linewidth=3)


