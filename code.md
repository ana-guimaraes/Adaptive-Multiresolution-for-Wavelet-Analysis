#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Ana Luiza Bastos Barbosa Guimarães da Silva
         Odylio Denys de Aguiar
         Riccardo Sturani
"""


import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data
from scipy import signal
import matplotlib.mlab as mlab
from scipy.interpolate import interp1d
import math
import matplotlib as mpl
import lal
import lalsimulation as lalsim
from scipy.signal import butter, filtfilt


# LOADING DATA

data=np.loadtxt('GW150914-4096-H1.txt')   # Loading the data

# GPS
gps_ev=1126259462.4       

tdata=8.       # Sampling Interval 
rate=4096.     # Sampling rate= nsamples/interval
dt=1./rate     # Time resolution 

gps_startdata= 1126257415           # When the data starts
gps_enddatacheck= 1126261511        # When the data ends


# Chopping data
tstart=6
gps_st=gps_ev-tstart         # New GPS starts  tstart before the event
gps_en=gps_st+tdata          # New GPS ends tdata after the event

gps_startpsd=gps_st-100  # GPS start for the PSD
gps_endpsd=gps_startpsd+32    # GPS end for the PSD


gps_enddata=gps_startdata+(float(len(data)))*dt  # The data has a total of 4096 seconds

tall=np.linspace(gps_startdata,gps_enddata,len(data))  # Establishing the time line 

print('GPS end:',gps_enddata, gps_enddatacheck) # They must be the same!

################# SAVING THE CHOPPED DATA
data1=data[np.where((tall>=gps_st) & (tall<=gps_en))]
np.savetxt('data.txt', data1)
#################

################# SAVING THE DATA for the PSD
datapsd=data[np.where((tall>=gps_startpsd) & (tall<=gps_endpsd))]
np.savetxt('datapsd.txt', datapsd)
#################



# NOISE SPECTRAL DENSITY

# parameters
nsamples= rate*tdata            # Number of samples in 8s of signal
df= 1/(nsamples*dt)             # Frequency Resolution, df= 1/interval 
npoints= int(rate/2./df)+1      # Number of samples used in frequency domain
frequency = np.linspace(0,rate/2, npoints, endpoint=True) # Frequency goes from [0,Fnyq] 
NFFT=4*int(rate)                # Number of seconds for the fast Fourier transform        

# NSD calculation + interpolation
psd, freqs = mlab.psd(datapsd, Fs = rate, NFFT = NFFT)     
iPSD_H1 = interp1d(freqs, psd)  # PSD interpolation 
dPSD_H1= iPSD_H1(frequency)


# NSD plot
plt.loglog(freqs,psd)
plt.title('Power Spectral Density of the Noise')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power $[Hz]^{-1}$')
plt.grid(which='both', axis='both')
plt.savefig('nsd')
plt.show()


#In case NSD is half of the fft_data size, you can double it using this:

#dPSD_H1 = np.zeros(int((rate/df)))
#for idx in range(len(frequency)):
#    dPSD_H1[idx]=iPSD_H1(frequency[idx])
#for idx in range(len(frequency)-2):
#    dPSD_H1[len(frequency)+idx]=dPSD_H1[len(frequency)-idx-1]



# CHOPPED DATA (TIME DOMAIN)

# parameters
npoints= int(rate/2./df)+1   # Number of samples in frequency domain
# Time goes from gps_st to gps_en and we use the convention t(event)=0
t = np.arange(gps_st-gps_ev,gps_en-gps_ev,dt)    


# Windowing the data

'''
We use Tukey Window!

signal.windows.tukey(N,a)

where N= number of samples
a=[0,1], if a=0: Retangular Window (box window),
         if a=1: Hann Window 
'''

dataw= data1*signal.windows.tukey(int(nsamples),0.2) 
#dataw= data1


# Chopped data in TIME domain plot
plt.plot(t, dataw) 
plt.title('GW150914 - Windowed time-domain data')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.savefig('windowedtimeseriesGW150914')
plt.show()

print('dt=',dt)
print('len(dataw)=',len(dataw))




# Plotting different Tukey Windows
tw1=signal.windows.tukey(int(nsamples),0.0)
tw2=signal.windows.tukey(int(nsamples),1)
tw3=signal.windows.tukey(int(nsamples),0.3)

# Chopped TW in TIME domain plot
plt.plot(t, tw1, label='Tukey Window (p= 0.0)')
plt.plot(t, tw3, label='Tukey Window (p= 0.3)')
plt.plot(t, tw2, label='Tukey Window (p= 1.0)')


plt.title('Tukey Windows with different shape parameters (p)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.savefig('windows')
plt.show()

tw1_data=data1*signal.windows.tukey(int(nsamples),0.1)
tw2_data=data1*signal.windows.tukey(int(nsamples),0.3)

# Chopped windowed data in TIME domain plot
plt.plot(t, tw1_data, label='Tukey Window (p= 0.1)')
plt.plot(t, tw2_data, label='Tukey Window (p= 0.3)')

plt.title('Data + Tukey Window')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.savefig('windows+data')
plt.show()



# CHOPPED DATA (FREQUENCY DOMAIN)

fft_data = np.fft.rfft(dataw)*dt       # Fast Fourier Transform of the chopped data
amplitude = 2./npoints * np.abs(fft_data[0:int(rate/2/df)+1])


print(len(frequency))
print(len(amplitude))
print(len(dataw))
print(len(fft_data))
print(len(dPSD_H1))

plt.loglog(frequency, amplitude) # Frequency domain plot
plt.title('Frequency-domain data')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid(which='both', axis='both')
plt.show()



# Parseval Theorem for Chopped Data

norm_freq=2.*np.sum(np.abs(fft_data*np.conj(fft_data)))-np.abs(fft_data[0]*np.conj(fft_data[0]))-np.abs(fft_data[-1]*np.conj(fft_data[-1]))
norm_freq=np.sqrt(norm_freq*df)
norm_time=np.sqrt(np.sum(dataw*dataw)*dt)
print("Perseval Theorem Check:")
print('norm time domain=', norm_time,",    norm freq domain= ",norm_freq)
print('norm time domain/ norm freq domain=',norm_time/norm_freq)



# WHITENING

# parameters
fnyq=rate/2  # Nyquist Frequency

'''
Whitening:
m_data= (signal in Fourier Domain)/ Amplitude Spectral Density
w_data: comes back to time domain
'''

m_data= fft_data/(np.sqrt(dPSD_H1)*rate/2)
w_data= np.fft.irfft(m_data)/dt               # Whitened data


# Apllying a bandpass filter
bpc1, bpc2 = butter(4, [20./fnyq, 1000./fnyq], btype='band') 
w_databp = filtfilt(bpc1, bpc2, w_data)    # Whitened and bandpassed data 


################# SAVING THE WHITENED DATA
np.savetxt('w_data.txt', w_data)
np.savetxt('w_databp.txt', w_databp)
#################


# PLOTS

# Whitened data in TIME domain
print('len(w_data)=', len(w_data))


plt.plot(t,w_data, label='w_data')
plt.title('Whitened data - Time-domain')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.savefig('wdata32-tw=0.3.jpg')
plt.show()


# Whitened and bandpassed data in TIME domain

print('len(w_databp)=', len(w_databp))
plt.plot(t,w_data, label='w_data')
plt.plot(t,w_databp, label='w_databp')
#plt.plot(t,(w_data-w_databp), label='diff')
plt.title('Whitened and bandpassed data - Time-domain')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.savefig('wdatabp')
plt.show()
print('Diff=',np.sum((w_data-w_databp)**2))

# Whitened and bandpassed data in FREQUENCY domain
fft_wdata = np.fft.rfft(w_data)*dt       
amplitude_wdata = 2./npoints * np.abs(fft_wdata[0:int(rate/2/df)+1])

fft_wdatabp = np.fft.rfft(w_databp)*dt       
amplitude_wdatabp = 2./npoints * np.abs(fft_wdatabp[0:int(rate/2/df)+1])



plt.loglog(frequency,amplitude_wdata, label='w_data')
plt.loglog(frequency,amplitude_wdatabp, label='w_databp')
plt.title('Whitened and bandpassed data - Frequency-domain')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.savefig('wdata-wdatabp')
plt.show()
print('Diff=',np.sum((amplitude_wdata-amplitude_wdatabp)**2))


# Parseval Theorem for Whitened Data
normwtil=2.*np.sum(np.abs(m_data*np.conj(m_data)))-np.abs(m_data[0]*np.conj(m_data[0]))-np.abs(m_data[-1]*np.conj(m_data[-1]))
normwtil=np.sqrt(normwtil*df)
normw=np.sqrt(np.sum(w_data*w_data)*dt)
print("Parseval Theorem:")
print('norm whitened time domain=', normw, ",    norm whitened freq domain= ", normwtil)
print('norm whitened time domain/ norm whitened freq domain=', normw/normwtil)


# In[ ]:


# DEFINITIONS


# Shannon's Entropy
def entropy(coef):
    norma = np.sqrt(np.sum(np.square(coef)))
    a = coef/norma
    #print(len(a))
    E=0.
    for i in range(len(a)):
        if abs(a[i])>0.:
            E += -np.sum((a[i]**2)*np.log(a[i]**2))
    return E


# Storing level
def store_level(N,ifreq,level):
    dt=(2**level)/rate
    T=float(N)*dt
    tbins=np.arange(0.,T+dt,dt)
    df=rate/2./pow(2,level)
    fbins=np.zeros(2)
    fbins[0]=float(ifreq)*df
    fbins[1]=fbins[0]+df
    return tbins, fbins


# Gaussian Function
def gaussian(t,t0,c): 
    return np.exp(-np.power((t-t0)/c, 2.)/2.)


# GW f(t) frequency
def max_after_xval(f,x,val=0.):
    idx=0
    N=len(x)
    while ( (idx<N-1) and (x[idx]<val) ):
        idx+=1
    while ( (idx<N-1) and (f[idx]<f[idx+1]) ): 
        idx+=1
    return idx,f[idx]


# Joining coeffs
def join(x1,x2):
    out=np.zeros(len(x1)+len(x2))
    out[:len(x1)]=x1
    out[len(x1):]=x2
    return out

# Quantile 
def quantile(inp,a):
    q= np.quantile(abs(inp),a)
    print('quantile=', q)
    return q 


# Adaptive Filtering
def filtering(inp,a):
    out=inp
    b=quantile(out,a)
    for indx in range(len(inp)):
        if abs(out[indx]) < b: 
            out[indx] = 0.
    print('filtered!')        
    return out


# Inverse Discrete Wavelet Transform

def idwt(coeffs,idxf,idxl,Ntdata,wl):
    nc=len(coeffs)  # number of coefficients
    if (nc*pow(2,idxl)!=Ntdata):
        print("Error in IDWT",Ntdata," vs. ",nc*pow(2,idxl))  
        return -1
    out=coeffs
    idxfi=idxf
    for idxi in range(idxl):
        if (idxfi%2==0):   
            out=pywt.idwt(out,np.zeros(len(out)),wl)
            #print('DONE 1')
        else:
            out=pywt.idwt(np.zeros(len(out)),out,wl)
        idxfi//=2
        #print('DONE 2')
    return out



# Discrete Wavelet Transform

def dwt(data,wavelet):
    
    N=len(data)
    maxl=pywt.dwt_max_level(len(data), wavelet)     
    max_level=int(np.log(N)/np.log(2))
    #max_level=3
    print('The maximum level of decomposition is', max_level)
    print('Mother wavelet:', wavelet)
    
    coeff_new=np.ones((1,len(data)))
    coeff_new*=data
    ntd_new=len(coeff_new[0])
    nfbin_new=1
    level=0
    go_new=np.ones(1,dtype=int)
    out=[]
    
    ctotal=[]

    coeffs_rec=np.zeros(len(data))
    coeffsfilter_rec=np.zeros(len(data))

    while ((level<max_level)and(np.sum(go_new)>0)):
        coeff_old=coeff_new
        nfbin_old=nfbin_new
        ntd_old=ntd_new
        go_old=go_new
        level+=1
        nfbin_new=nfbin_old*2
        ntd_new=int(ntd_old/2)
        coeff_new=np.zeros((nfbin_new,ntd_new))
        go_new=np.zeros(2*len(go_old),dtype=int)

        print("############# Level: ",level);
        coeff_new=np.zeros((nfbin_new,ntd_new))
        for i in range(nfbin_old):
            print("  ",i+1,"/",nfbin_old," go:",go_old[i])
            if go_old[i]==1:
                print(" DWT for an object of length:",len(coeff_old[i]))
                clow, chigh = pywt.dwt(coeff_old[i],wavelet)
                
                E0=entropy(coeff_old[i])
                E1=entropy(join(clow,chigh))
                #print('clow=',clow,',   chigh=',chigh)
                #print('** lowest clow=',min(clow), ', highest clow', max(clow))
                #print('## lowest chigh=',min(chigh), ', highest chigh', max(chigh))
                #print('++++++',coeff_old[i])
                print(" The previous entropy was:  ",E0," The new entropy is:  ",E1)

                if (E1>E0):
                    print("   Saved for later\n")
                    tbins, fbins=store_level(len(coeff_old[i]),i,level-1)
                    
                    coeffs_rec+=idwt(coeff_old[i],i,level-1,len(data),wavelet)
                    
                    coeffs=filtering(coeff_old[i],.0)
                    coeffsfilter_rec+=idwt(coeffs,i,level-1,len(data),wavelet)


                    #print('+++++ COEFFS:',coeffs)
                    #print('***** COEFFS REC:',coeffs_rec)
                    
                    ctotal.extend(coeff_old[i])  

                    out.append([tbins,fbins,coeffs])
                    #print('//// COEFFS negative:',coeffs[np.where(coeffs<0)])
                    #print('//// COEFFS positive:',coeffs[np.where(coeffs>0)])
                    
                else:
                    
                    if level==max_level:
                        #clow
                        tbins, fbins=store_level(len(clow),2*i,level)
                                
                        ctotal.extend(clow)
    
                            
                        clow_f=filtering(clow,.2)
                        out.append([tbins,fbins,clow_f])
                        

                        
                        coeffs_rec+=idwt(clow, 2*i,level, len(data),wavelet)
                        coeffsfilter_rec+=idwt(clow_f,2*i,level,len(data),wavelet)

                        #chigh
                        tbins, fbins=store_level(len(chigh),2*i+1,level)
                            
                        ctotal.extend(chigh)
    
                        chigh_f=filtering(chigh,.2)
                        out.append([tbins,fbins,chigh_f])
                        
                    
                        coeffs_rec+=idwt(chigh, 2*i+1,level, len(data),wavelet)
                        coeffsfilter_rec+=idwt(chigh_f,2*i+1,level,len(data),wavelet)

                        #print('**** COEFFS REC:',coeffs_rec)
                       
                        #print('+++++++++',clow_f,chigh_f)

                    else:
                        
                        print("   Go on i f=",i," MAX e=",np.log(float(ntd_old)))
                        coeff_new[2*i]  =clow
                        coeff_new[2*i+1]=chigh
                        go_new[2*i]=1
                        go_new[2*i+1]=1
                        

                                          
                       
    
    if (np.sum(go_new)):
        print(level," levels of decompositions were made!")
    else:
        print("Stopped at ",level-1," decompositions!")
    return out, coeffs_rec, coeffsfilter_rec, ctotal




# Mother Wavelet

#print(pywt.wavelist(kind='discrete'))



# DWT DECOMPOSITIONS

out_wdata, coeffs_rec, coeffsfilter_rec, ctotal = dwt(w_databp,'db1')



print('They must all be the same:')
print(len(w_data))
print(len(w_databp))
print(len(coeffs_rec))
print(len(ctotal))


# Quantile line
q95=np.quantile(np.abs(ctotal),.95)
q90=np.quantile(np.abs(ctotal),.90)

plt.axvline(q90,color='g', label='Quantile (0.90)')
plt.axvline(q95,color='r', label='Quantile (0.95)')

# Histogram plot
ctotal_abs=np.abs(ctotal)

plt.hist(ctotal_abs, bins=len(ctotal), color='#0504aa', range=(min(np.abs(ctotal_abs)), 
                                                               max(np.abs(ctotal_abs))))

plt.grid(axis='y')
plt.gcf().set_size_inches(40, 20) # alterar tamanho
plt.xlabel('Module of the Coefficients',fontsize=30)
plt.ylabel('Number of Coefficients',fontsize=30)
plt.title('Histogram',fontsize=35)
plt.yticks(range(0, 46),fontsize=15) # mudar escala do eixo Y + tamanho da fonte
plt.xticks(fontsize=30)
plt.legend(fontsize=30)
plt.xlim([0.1,0.2])

plt.savefig('histogram-GW150914-xlim.jpg')
plt.show()

print(max(ctotal))
print(len(np.abs(ctotal)))
#print(bins)

# Set a clean upper y-axis limit. , color='#0504aa', 
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)



# IDWT Plot 

# Data reconstruction plot

plt.plot(t, w_databp, label='w_databp')
plt.plot(t, coeffs_rec, label='coeffs_rec')
plt.plot(t, (w_databp-coeffs_rec) ,label='diff')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Whitened data bp and Recovered coefficients - GW150914 (H1)')
plt.legend()
plt.grid()
plt.savefig('idwt-zoom')
plt.show()
print('Diff=', np.sum((w_databp-coeffs_rec)**2))

# Filtered data reconstruction plot

plt.plot(t, w_databp, label='w_databp')
plt.plot(t, coeffsfilter_rec,label='coeffsfilter_rec')
plt.plot(t, (w_databp-coeffsfilter_rec),label='diff')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Whitened data bp and Recovered filtered coefficients - GW150914 (H1)')
plt.legend()
plt.grid()
plt.savefig('idwt-filter')
plt.show()
print('Diff=',np.sum((w_databp-coeffsfilter_rec)**2))



#### MULTI-RESOLUTION REPRESENTATION ####

plt.figure("Multi-resolution")

### Plotting GW signal ###

Mtot=28
#q=1.
#m1 = q*Mtot/(1.+q)
m1=36
#m2 = Mtot/(1.+q)
m2=29
s1 = [0.,0.,0.]
s2 = [0.,0.,0.]
rate=4096.
deltaT = 1./rate
f_min = 10.
f_ref = f_min
phi_ref = 0.
distance = 410*pow(10,6)*lal.PC_SI
inclination = 0 
pars = lal.CreateDict()
model = lalsim.SEOBNRv4PHM

lalsim.SimInspiralWaveformParamsInsertEOBEllMaxForNyquistCheck(pars,2)
hp, hc=lalsim.SimInspiralChooseTDWaveform(m1*lal.MSUN_SI,m2*lal.MSUN_SI,s1[0],s1[1],
                                          s1[2],s2[0],s2[1],s2[2],
                                          distance,inclination,phi_ref,0.,0.,0.,
                                          deltaT,f_min,
                                          f_ref,pars,lalsim.SEOBNRv4PHM)


hpd=hp.data.data
hcd=hc.data.data
dhp=np.concatenate((np.diff(hpd),np.zeros(1)))/deltaT
dhc=np.concatenate((np.diff(hcd),np.zeros(1)))/deltaT
tm=np.arange(0,len(hpd))*deltaT+(hp.epoch.gpsSeconds+1.e-9*hp.epoch.gpsNanoSeconds)
f22GW=(hpd*dhc-hcd*dhp)/((hpd*hpd+hcd*hcd)*2*np.pi)
print(len(tm),len(hp.data.data),len(hcd),len(dhp),len(dhc),len(f22GW))

imax,_=max_after_xval(f22GW,tm)
idxs2=np.where(tm<=tm[imax])

hpd2=hpd[idxs2]
tm2=tm[idxs2]
f22GW2=f22GW[idxs2]
print(len(hpd2),len(tm2),len(f22GW2))


######### Wavelet Representation ########## 

zmin=1000000
zmax=-1000000

for i in range(len(out_wdata)):
    zmin=min(zmin,min(out_wdata[i][2]))
    zmax=max(zmax,max(out_wdata[i][2]))
if zmax<abs(zmin):
    zmax=abs(zmin)
else:
    zmin=-zmax

        
for i in range(len(out_wdata)):
    X=out_wdata[i][0]-tstart
    Y=out_wdata[i][1]
    Z=np.ones((1,len(out_wdata[i][2])))
    Z*=abs(out_wdata[i][2])
    #Z[np.where(abs(Z)>0.2)]*=0
    

    Yy,Xx=np.meshgrid(X,Y)
    
    cmap = mpl.cm.RdBu_r
  
    plt.pcolormesh(Yy,Xx,Z, cmap=cmap,vmin=zmin, vmax=zmax)
plt.plot(tm2,f22GW2)    
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('GW150914 (H1) - Filter 1')
plt.colorbar(label='Arbitrary Units')
plt.show()
print('zmin:',zmin,'zmax:',zmax)



### Plotting GW signal check ###

plt.figure()
plt.plot(tm2,f22GW2)
plt.show()



