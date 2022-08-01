from numpy.random import normal
import numpy as np

def cNoise(beta,shape=(1024,),std=0.001, maxCorrections=10,maxAvgError=0.01, eta=0.6):
    '''
       Wrote by: Rubens Andreas Sautter (2021)
       
       An parameter of correction has been used (s):
       	FFT(f(w)) = gauss(0,std) * (1/w^beta)^(beta*s/2) 
       
        Frequencies are measured in multidimensional space by the frequency euclidian distance.
        
       =====================================================================================
       beta (float) - the colored noise decay (0-white noise, 1-pink noise, 2- red noise)
       shape (tuple) - the output shape
       std  (float) standard deviation of the spectrum gaussian function (see reference)
       maxCorrections (int) - maximum number of iteractions of the process of decay correction 
       maxAvgError (float) - target error of the spectrum decay
       eta (float) - optimization parameter, large eta can sometimes not reach the minimum
       		small eta is slower (like the gradient descent eta)
       		
       		* For beta = [0,2], eta>=0.6 seems to converge
       		      beta - 3, eta <= 0.6 seems to converge
       
       =====================================================================================
       Inspired by:
      http://articles.adsabs.harvard.edu//full/1995A%26A...300..707T/0000707.000.html
    '''
    dimension = []
    for index,dsize in enumerate(shape):
        dimension.append(np.fft.fftfreq(dsize).tolist())
    dimension = tuple(dimension)
    d = float(len(dimension))
    
    freqs = np.power(np.sum(np.array(np.meshgrid(*dimension,indexing='ij'))**2,axis=0),1/2)*np.sqrt(2)/4
    
    #Sampling gaussian with sandard deviation varying according to frequency
    ftSample = normal(loc=0,scale=std,size=shape) + 1j*normal(loc=0,scale=std,size=shape)
    
    # Setting the scale [0,2pi]
    freqs = np.pi*freqs
    not0Freq = (np.abs(freqs)>1e-15)
    positiveFreq = (freqs>1e-15)
    
    decayCorrectionL = []
    errorL = []
    
    # Building the first spectrum trial
    decayCorrection = np.sqrt(2)**(d-1)
    scaling = (freqs[not0Freq]+0j)**(-(beta*decayCorrection)/2)
    generatedSpectrum = ftSample.copy()
    generatedSpectrum[not0Freq] = (ftSample[not0Freq]*scaling)
    spsd = np.sum(np.abs(generatedSpectrum))
    out = np.fft.ifftn(generatedSpectrum).real
    # zero avg
    ftSample[0] = 0.0
    
    # one dimensional noise does not require corrections
    if len(dimension)==1:
        return out
    
    #measuring the average beta
    betas = []
    for i in range(len(out)):
        series = out[i,...]	
        # multidimensional slice
        if(len(dimension)>2):
            for j in range(len(dimension)-2):
                series = series[0]
        psd = np.fft.fft(series)
        psd = np.real(psd*np.conj(psd))
        lfreqs = np.fft.fftfreq(len(series))
        fPSD = psd[lfreqs>0.0]
        fFreqs = lfreqs[lfreqs>0.0]
        fit = np.polyfit(np.log(fFreqs),np.log(fPSD),deg=1)
        betas.append(-fit[0])
    

    	
    # measuring the error 
    smallCorrection = beta-np.average(betas)
    
    #including in the list
    decayCorrectionL.append(decayCorrection)
    errorL.append(smallCorrection)
    
    countCycles = 0
    # rebuilding the spectrum
    while np.abs(smallCorrection)>maxAvgError:
        decayCorrection += smallCorrection*eta
        scaling = (freqs[not0Freq]+0j)**(-(beta*decayCorrection)/2)
        generatedSpectrum = ftSample.copy()
        generatedSpectrum[not0Freq] = (ftSample[not0Freq]*scaling)
        spsd = np.sum(np.abs(generatedSpectrum))
        out = np.fft.ifftn(generatedSpectrum).real

        #measuring the average beta
        betas = []
        for i in range(len(out)):
            series = out[i,...]
            # multidimensional slice
            if(len(dimension)>2):
                for j in range(len(dimension)-2):
                    series = series[0]
            psd = np.fft.fft(series)
            psd = np.real(psd*np.conj(psd))
            lfreqs = np.fft.fftfreq(len(series))
            fPSD = psd[lfreqs>0.0]
            fFreqs = lfreqs[lfreqs>0.0]
            fit = np.polyfit(np.log(fFreqs),np.log(fPSD),deg=1)
            betas.append(-fit[0])
        	
        # measuring the error
        smallCorrection = beta-np.average(betas)
        
        print("Noise error - ", smallCorrection)
        decayCorrectionL.append(decayCorrection)
        errorL.append(smallCorrection)
    	
        countCycles = countCycles+1
        if countCycles>maxCorrections:
            break
            
    
    # resampling with the best decay
    errorL = np.abs(errorL)
    print("Best decay constant:", decayCorrectionL[np.argmin(errorL)]," Error: ",errorL[np.argmin(errorL)])
    decayCorrection = decayCorrectionL[np.argmin(errorL)]
    scaling = (freqs[not0Freq]+0j)**(-(beta*decayCorrection)/2)
    generatedSpectrum = ftSample.copy()
    generatedSpectrum[not0Freq] = (ftSample[not0Freq]*scaling)
    spsd = np.sum(np.abs(generatedSpectrum))
    out = np.fft.ifftn(generatedSpectrum).real

    # normalizing
    out = out / np.max(np.abs(out))
    return out
