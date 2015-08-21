"""
Translation of Shankar Shivappa's matlab HarmonicAnalysis.m code to Python

Usage Example:
    import pika as p
    import numpy as np

    (audio, freq, nBits) = p.load_audio(p.infile)
    left = [v[0] for v in audio] #left channel
    found, peaks = p.find_pika(left, freq)
    out = p.pika_audio(left, freq, found)
    p.write_wav(out, "test.wav")

"""
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import scikits.audiolab
import find_peaks as peaks
import matplotlib.pyplot as plt

#infile = "May26-2014-BeaconRockSP4800.wav"
infile = "May26-2014-BeaconRockSP_shankars.wav"

def load_audio(filepath):
    (snd, sampFreq, nBits) = scikits.audiolab.wavread(filepath)
    print "sample frequency of {}: {}".format(filepath, sampFreq)
    #if sampFreq <> 48000:
        #print "File not sample frequency of 48000, please resample and try again"
        #return False     
    return (snd, sampFreq, nBits)


def find_pika(audio, freq):
    pika_found = []
    fft_size = 4096
    output = open("pika_all_locs.csv", "w")
    for i in range(0, len(audio) - fft_size, fft_size/2):
        f = np.absolute(np.fft.fft(audio[i:i+fft_size]))    #Magnitude of fft
        f = f[fft_size/32:len(f)/2]                         #Chop out some unneeded frequencies
        f = np.log10(f/sum(f))                              #Compress signal
        f_mean = np.mean(f)                                 
        tf = [1.0 if x > f_mean + 1 else 0.0 for x in f]
        tf = np.convolve(tf,[0.125, 0.25, 0.4, 0.5, 0.9, 1, 1, 1, 0.9, 0.5, 0.4, 0.25, 0.125]);
        locs = peaks.detect_peaks(tf, mpd=50)
        if len(locs) >= 4: 
            inter_peak_dist = np.convolve(locs, [1, -1])
            harmonic_freq =  np.median(inter_peak_dist[1:-1])
            if (harmonic_freq < 75) and (harmonic_freq > 55):
                pika_found.append(i*1.0/freq)
    found = [int(x) for x in pika_found] #integer values
    #TODO use call boundaries rather than 
    found = list(set(found)) #get unique values
    output.close()
    return found

def pika_audio(audio, freq, pika):
    X = []
    m = max(audio)
    for ind in pika:
        X.append(m)
        X.extend(audio[(ind)*freq:(ind+1)*freq])
    return X

def write_wav(audio, freq, outfile='test.wav'):
    scikits.audiolab.wavwrite(np.asarray(audio), outfile, freq)
    
