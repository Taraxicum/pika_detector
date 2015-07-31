"""
Recreating HarmonicAnalysis.m process in python

Example:
    import pika as p
    import numpy as np

    (audio, freq, nBits) = p.load_audio(p.infile)
    left = [v[0] for v in audio] #left channel
    found = p.find_pika(left, freq)
    out = p.pika_audio(left, freq, found)
    p.write_wav(out, "test.wav")

"""
#from pylab import *
#import pickle
import numpy as np
import scipy
from matplotlib import animation
import scikits.audiolab
import scipy.signal as signal
import scipy.stats as stats
from scipy.signal import kaiserord, lfilter, firwin, freqz, fftconvolve
import find_peaks as peaks
#from FFT import *
#import random

infile = "May26-2014-BeaconRockSP.wav"

def load_audio(filepath):
  (snd, sampFreq, nBits) = scikits.audiolab.wavread(filepath)
  print "sample frequency of {}: {}".format(filepath, sampFreq)
  return (snd, sampFreq, nBits)


def find_pika(audio, freq):
    pika_found = []
    fft_size = 4096
    for i in range(0, len(audio), fft_size):
        f = np.absolute(np.fft.fft(audio[i:i+fft_size]))
        f = f[fft_size/32:len(f)/2]
        f = np.log10(f/sum(f))
        f_mean = np.mean(f)
        tf = [1.0 if x > f_mean + 1 else 0.0 for x in f]
        tf = np.convolve(tf,[0.125, 0.25, 0.4, 0.5, 0.9, 1, 1, 1, 0.9, 0.5, 0.4, 0.25, 0.125]);
        #signal.find_peaks_cwt(tf, min_length= 50)        
        locs = peaks.detect_peaks(tf, mpd=50)
        if len(locs) > 4:
            inter_peak_dist = np.convolve(locs, [1, -1])
            harmonic_freq =  np.median(inter_peak_dist[2:-1])
            if (harmonic_freq < 75) and (harmonic_freq > 55):
                pika_found.append(i*1.0/freq)
    found = [int(x) for x in pika_found] #integer values
    found = list(set(found)) #get unique values
    return found

def pika_audio(audio, freq, pika):
    X = []
    m = max(audio)
    #d = np.asarray(audio)
    #d = np.append(d, np.zeros(freq))
    for ind in pika:
        X.append(m)
        X.extend(audio[ind*freq:(ind+1)*freq])
    return X

def write_wav(audio, outfile='test.wav'):
    scikits.audiolab.wavwrite(np.asarray(audio), outfile, 44100)
    
