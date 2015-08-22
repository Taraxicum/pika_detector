"""
Adapted from Shankar Shivappa's matlab HarmonicAnalysis.m code to Python

Usage Example:
    import pika as p
    
    (audio, freq, nBits) = p.load_audio(p.infile)   #here p.infile could be any desired audio file
    #left = [v[0] for v in audio] #left channel if a stereo file not needed for mono
    parser = p.AudioParser(audio, freq)
    parser.pre_process()
    parser.find_pika_calls()
    parser.output_audio("pika_calls.wav")
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
    return (snd, sampFreq, nBits)

class AudioParser(object):
    """For taking chunks of audio then pre-processing, identifying and outputting pika calls.
    Long sections of audio are tough on memory usage, so pre-chopping longer audio into 10 second
    or so chunks is recommended.
    
    Example usage:
    parser = p.AudioParser(audio, freq)
    parser.pre_process()
    parser.find_pika_calls()
    parser.output_audio("pika_calls.wav")
    """
    def __init__(self, audio, frequency):
        """Audio should be a single channel of raw audio data.
        """
        self.audio = audio
        self.frequency = frequency
        self.processed_fft_frames = []
    
    def pre_process(self):
        """Transform (fft), filter, and normalize audio
        """
        pika_found = []
        fft_size = 4096
        for i in range(0, len(self.audio) - fft_size, fft_size/2):
            f = np.absolute(np.fft.fft(self.audio[i:i+fft_size]))    #Magnitude of fft
            f = f[fft_size/32:len(f)/2]                         #Chop out some unneeded frequencies
            f = np.log10(f/sum(f))                              #Compress signal
            f_mean = np.mean(f)                                 
            tf = [1.0 if x > f_mean + 1 else 0.0 for x in f]
            tf = np.convolve(tf,[0.125, 0.25, 0.4, 0.5, 0.9, 1, 1, 1, 0.9, 0.5, 0.4, 0.25, 0.125])
            self.processed_fft_frames.append(tf)

    def find_pika_calls(self):
        """pre_process() must have been called first.
        :puts list of integers identifying which seconds in the audio are thought to contain
        into self.found
        pika calls.
        """
        if len(self.processed_fft_frames) == 0:
            print "Can't find pika calls: No processed fft_frames, did you called pre_process() first?"
            return
        
        pika_found = []
        factor = 2048.0/self.frequency #mutliply i by factor to get time of the frame's location in seconds 
        for i, f in enumerate(self.processed_fft_frames):
            locs = peaks.detect_peaks(f, mpd=50)

            if len(locs) >= 4: 
                #print locs
                inter_peak_dist = np.convolve(locs, [1, -1])
                harmonic_freq =  np.median(inter_peak_dist[1:-1])
                if (harmonic_freq < 75) and (harmonic_freq > 55):
                    pika_found.append(i*factor)
        found = [int(x) for x in pika_found] #integer values
        #TODO use call boundaries rather than peaks
        self.found = list(set(found)) #get unique values

    def output_audio(self, file_name):
        """Must have called find_pika_calls first.
        No return value, but outputs pika audio to file_name
        """
        X = []
        m = max(self.audio)
        for ind in self.found:
            X.append(m)
            X.extend(self.audio[(ind)*self.frequency:(ind+1)*self.frequency])
        scikits.audiolab.wavwrite(np.asarray(X), file_name, self.frequency)

