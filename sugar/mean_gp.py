import numpy as np
import sugar
from scipy.signal import savgol_filter

def comp_mean(phase_min=-12, phase_max=48, draw=False):

    slds = sugar.load_data_sugar()
    slds.load_spectra()

    sn_name = slds.spectra.keys()

    wave = slds.spectra_wavelength[sn_name[0]]['0']
    number_bin_wavelength = len(slds.spectra_wavelength[sn_name[0]]['0'])

    spectra_bin_phase = np.linspace(phase_min,phase_max,int(phase_max-phase_min)/2+1)
    number_bin_phases = len(spectra_bin_phase)

    hist_spectra = np.zeros((len(sn_name),number_bin_wavelength*number_bin_phases))
    hist_spectra_weights = np.zeros((len(sn_name),number_bin_wavelength*number_bin_phases))
    wavelength = np.zeros(number_bin_wavelength*number_bin_phases)
    phases = np.zeros(number_bin_wavelength*number_bin_phases)

    reorder_spec_to_lc = np.arange(number_bin_wavelength*number_bin_phases).reshape(number_bin_phases, number_bin_wavelength).T.reshape(-1)
    reorder_lc_to_spec = np.arange(number_bin_wavelength*number_bin_phases).reshape(number_bin_wavelength, number_bin_phases).T.reshape(-1)
    
    for sn in range(len(sn_name)):
        print sn_name[sn], '%i/%i'%((sn+1,len(sn_name)))
        for t in range(len(spectra_bin_phase)):
            for key in slds.spectra_phases[sn_name[sn]].keys():
                if abs(spectra_bin_phase[t] - slds.spectra_phases[sn_name[sn]][key]) < 1:
                    hist_spectra[sn,number_bin_wavelength*t:number_bin_wavelength*(t+1)] = slds.spectra[sn_name[sn]][key]
                    hist_spectra_weights[sn,number_bin_wavelength*t:number_bin_wavelength*(t+1)] = 1./slds.spectra_variance[sn_name[sn]][key]
            wavelength[number_bin_wavelength*t:number_bin_wavelength*(t+1)] = wave
            phases[number_bin_wavelength*t:number_bin_wavelength*(t+1)] = spectra_bin_phase[t]


    average_function_spectra = np.average(hist_spectra,weights=hist_spectra_weights,axis=0)
    average_function_light_curve = average_function_spectra[reorder_spec_to_lc]
    wavelength = wavelength[reorder_spec_to_lc]
    phases = phases[reorder_spec_to_lc]
    
    average_function_light_curve_smooth = np.zeros_like(average_function_light_curve)
    for i in range(number_bin_wavelength):
        print i
        average_function_light_curve_smooth[number_bin_phases*i:number_bin_phases*(i+1)] = savgol_filter(average_function_light_curve[number_bin_phases*i:number_bin_phases*(i+1)], 15, 2)
        
    average_function_spectra_smooth = average_function_light_curve_smooth[reorder_lc_to_spec]

    fichier=open('average_for_gp.dat','w')
    for i in range(len(wavelength)):
            fichier.write('%.5f    %.5f    %.5f \n'%((phases[i],wavelength[i],average_function_light_curve_smooth[i])))
    fichier.close()  

    if draw:    
        import pylab as plt
        plt.figure()
        plt.imshow(hist_spectra,interpolation='nearest',aspect='auto',cmap=plt.cm.Greys)
        plt.figure()
        plt.imshow(hist_spectra_weights,interpolation='nearest',aspect='auto',cmap=plt.cm.Greys_r)
        plt.figure()
        
        plt.plot(average_function_light_curve)
        plt.plot(average_function_light_curve_smooth)
        plt.plot(average_function_spectra+5)
        plt.plot(average_function_spectra_smooth+5)
        plt.gca().invert_yaxis()
        plt.show()

if __name__ == '__main__':

    comp_mean(phase_min=-12,phase_max=48,draw=True)
