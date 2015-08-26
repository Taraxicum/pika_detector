"""
Adapted from Shankar Shivappa's matlab HarmonicAnalysis.m code to Python
Further adaptations include pre-processing as discussed in "Acoustic Classification of Bird Species
 from syllabels: an Emprirical Study", Briggs, Fern, and Raich; Oregon State University

Usage Example:
    import pika as p
    
    (audio, freq, nBits) = p.load_audio(p.infile)   #here p.infile could be any desired audio file
    #left = [v[0] for v in audio] #left channel if a stereo file not needed for mono
    p.audio_segments(audio, freq, 10, "trial.wav") 

"""
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import scikits.audiolab
import find_peaks as peaks
import matplotlib.pyplot as plt
#from pylab import specgram

#infile = "May26-2014-BeaconRockSP4800.wav"
infile = "May26-2014-BeaconRockSP_shankars.wav"

def load_audio(filepath):
    (snd, sampFreq, nBits) = scikits.audiolab.wavread(filepath)
    print "sample frequency of {}: {}".format(filepath, sampFreq)
    return (snd, sampFreq, nBits)

def audio_segments(audio, freq, segment_length=10, 
        harmonic_file="output/harmonic.wav"):
    """Segments audio into segment_length (in seconds) chunks and runs the 
    algorithm on each chunk, joining the output together and outputting 
    to harmonic_file
    """
    harmonic_out = []
    total = 0

    for i in range(0, len(audio)/freq, segment_length):
        end = min(len(audio), (i+segment_length)*freq)
        parser = AudioParser(audio[i*freq:end], freq)
        parser.pre_process()
        count, out = parser.find_pika_from_harmonic(.1)
        harmonic_out.extend(out)
        total += count
    print "Total suspected calls: {}".format(total)
    scikits.audiolab.wavwrite(np.asarray(harmonic_out), harmonic_file, freq)

class AudioParser(object):
    """For taking chunks of audio then pre-processing, identifying and outputting pika calls.
    Long sections of audio are tough on memory usage, so pre-chopping longer audio into 10 second
    or so chunks is recommended.
    
    Example usage:
    parser = p.AudioParser(audio, freq)
    parser.pre_process()
    parser.output_audio("pika_calls.wav")
    """
    def __init__(self, audio, frequency):
        """Audio should be a single channel of raw audio data.
        """
        self.audio = audio
        self.frequency = frequency
    
    def plot_energy(self):
        factor = 2048.0/self.frequency
        plt.plot([i*factor for i in range(len(self.energy_envelope))], self.energy_envelope, 'ro')
        plt.show()
    
    def spectrogram(self):
        factor = 2048.0/self.frequency
        plt.subplot(1, 3, 1)
        plt.imshow(np.asarray(self.fft).T, origin='lower')
        plt.xticks(plt.xticks()[0], [str(int(t*factor)) for t in plt.xticks()[0]])
        plt.xlim(0, len(self.fft))
        plt.xlabel("Original chopped fft")
        x = [i*factor for i in range(len(self.fft))]
        plt.subplot(1, 3, 2)
        plt.imshow(np.asarray([f[0:500] for f in self.fft]).T, origin='lower')
        plt.xticks(plt.xticks()[0], [str(int(t*factor)) for t in plt.xticks()[0]])
        plt.xlim(0, len(self.fft))
        plt.xlabel("Further chopped fft")
        plt.subplot(1, 3, 3)
        plt.imshow(np.asarray([f[0:500] for f in self.processed_fft_frames]).T, origin='lower')
        plt.xticks(plt.xticks()[0], [str(int(t*factor)) for t in plt.xticks()[0]])
        plt.xlim(0, len(self.fft))
        plt.xlabel("Further chopped fft with log, threshold, and convolve")
        plt.show()

    
    def pre_process(self):
        """Transform (fft), filter, and normalize audio
        """
        fft_size = 4096
        first_dim = len(self.audio)/(fft_size/2) - 1
        second_dim = 500 # fft_size/2 - fft_size/32
        self.fft = np.zeros((first_dim, second_dim))
        self.processed_fft_frames = np.zeros((first_dim, second_dim+12)) #extra values from convolve (I think?)
        for i in range(0, len(self.audio) - fft_size, fft_size/2):
            f = np.absolute(np.fft.fft(self.audio[i:i+fft_size]))    #Magnitude of fft
            f = f[fft_size/32:fft_size/2]                         #Chop out some unneeded frequencies
            #f = np.log10(f/sum(f))
            self.fft[i*2/fft_size] = f[0:500]
        max_val = np.amax(self.fft)
        normed_fft = self.fft/max_val
        avg_fft = np.sum(normed_fft, axis=0)/len(normed_fft)
        nr_fft = np.zeros_like(self.fft)
        for i, frame in enumerate(normed_fft):
            nr_fft[i] = [max(frame[j] - avg_fft[j], 0) for j, v in enumerate(frame)] #noise-reduction
        
        f_mean = np.mean(nr_fft)
        #.15 threshold found through trial and error, could be adjusted for
        # slightly better results
        threshold = .15
        tf = [[1.0 if x > f_mean + threshold else 0.0 for x in f] for f in nr_fft] 
        self.processed_fft_frames = tf


        ###########TEMP###################
        ##Uncomment to view spectrograms of different filterings
        #factor = 2048.0/self.frequency
        #length = len(normed_fft)
        #plt.subplot(1, 4, 1)
        #plt.imshow(np.asarray([f[0:500] for f in normed_fft]).T, origin='lower')
        #plt.xticks(plt.xticks()[0], [str(int(t*factor)) for t in plt.xticks()[0]])
        #plt.xlim(0, length)
        #plt.xlabel("fft: chop, log, threshold, convolve")
        ##################################
        #plt.subplot(1, 4, 2)
        #plt.imshow(np.asarray([f[0:500] for f in normed_fft]).T, origin='lower')
        #plt.xticks(plt.xticks()[0], [str(int(t*factor)) for t in plt.xticks()[0]])
        #plt.xlim(0, length)
        #plt.xlabel("fft: and normed by max")
        ##################################
        #plt.subplot(1, 4, 3)
        #plt.imshow(np.asarray([f[0:500] for f in nr_fft]).T, origin='lower')
        #plt.xticks(plt.xticks()[0], [str(int(t*factor)) for t in plt.xticks()[0]])
        #plt.xlim(0, length)
        #plt.xlabel("fft: and subtract avg fft")
        ##################################
        #plt.subplot(1, 4, 4)
        #plt.imshow(np.asarray([f[0:500] for f in root_nr_fft]).T, origin='lower')
        #plt.xticks(plt.xticks()[0], [str(int(t*factor)) for t in plt.xticks()[0]])
        #plt.xlim(0, length)
        #plt.xlabel("fft: and square root")
        #plt.show()
        ##################################
        self.energy_envelope = np.convolve(np.sum([f[0:500] for f in nr_fft], axis=1), 
                [0.125, 0.25, 0.4, 0.5, 0.9, 1, 1, 1, 0.9, 0.5, 0.4, 0.25, 0.125])
        max_energy = np.max(self.energy_envelope)
        self.energy_envelope /= max_energy
        ##################################
        #plt.subplot(2, 1, 1)
        #plt.plot([i*factor for i in range(len(self.energy_envelope))], self.energy_envelope, 'b.')
        #plt.xlim(0, 60)
        #plt.xlabel("Energy")
        ##################################
        #e_mean = np.mean(self.energy_envelope)                                 
        #thresh_energy = [1.0 if x > 1.7*e_mean else 0.0 for x in self.energy_envelope]
        #plt.subplot(2, 1, 2)
        #plt.plot([i*factor for i in range(len(self.energy_envelope))], thresh_energy, 'g.')
        #plt.ylim(0, 1.2)
        #plt.xlim(0, 60)
        #plt.xlabel("Energy Threshold")
        #plt.show()
        ##################################

    def energy_segments(self):
        """Uses energy envelopes to attempt to identify locations of pika calls
        """
        energy_threshold = 1.7*np.average(self.energy_envelope)
        factor = 2048.0/self.frequency #mutliply i by factor to get time of the frame's location in seconds 
        self.energy_ridges = []
        current_ridge = None
        for i, e in enumerate(self.energy_envelope):
            if e > energy_threshold:
                if current_ridge is None:
                    current_ridge = i*factor
            else:
                if current_ridge is not None:
                    self.energy_ridges.append([current_ridge, i*factor])
                    current_ridge = None
        if current_ridge is not None:
            self.energy_ridges.append([current_ridge, len(self.energy_envelope)*factor])
            current_ridge = None
    
    def harmonic_frequency(self):
        """Uses harmonic frequencies to attempt to identify locations of pika calls
        """
        pika_found = []
        self.ridges = []
        factor = 2048.0/self.frequency #mutliply i by factor to get time of the frame's location in seconds 
        self.peaks = []
        current_ridge = None
        for i, f in enumerate(self.processed_fft_frames):
            locs = peaks.detect_peaks(f, mpd=50)
            self.peaks.append(locs)
            if len(locs) >= 4: 

                inter_peak_dist = np.convolve(locs, [1, -1])
                harmonic_freq =  np.median(inter_peak_dist[1:-1])
                if (harmonic_freq < 75) and (harmonic_freq > 55):
                    if current_ridge is None:
                        current_ridge = i*factor
                    pika_found.append(i*factor)
                elif current_ridge is not None:
                    self.ridges.append([current_ridge, i*factor])
                    current_ridge = None
            elif current_ridge is not None:
                self.ridges.append([current_ridge, i*factor])
                current_ridge = None
        if current_ridge is not None: 
            self.ridges.append([current_ridge, len(self.processed_fft_frames)*factor])
        return pika_found

    def find_pika_from_energy(self, filename='test.wav'):
        self.energy_segments()
        output = []
        m = max(self.audio)
        #print "Number of incidents being output: {}".format(len(self.energy_ridges))
        for r in self.energy_ridges:
            output.append(m)
            output.extend(self.audio[max(0, r[0]-.01)*self.frequency:
                min(r[1]+.01, len(self.energy_envelope))*self.frequency])
        return output
   
    def find_pika_from_harmonic(self, buffer_length=.01):
        """pre_process() must have been called first.
        :returns list containing segments of self.audio thought to contain pika calls
        with buffer_length (in seconds) space on each side of the pika call 
        (buffered with the signal in audio).
        """
        output = []
        m = max(self.audio)
        self.harmonic_frequency()
        print "Number of incidents being output: {}".format(len(self.ridges))
        for r in self.ridges:
            output.append(m)
            output.extend(self.audio[max(0, r[0] - buffer_length)*self.frequency:
                min(r[1] + buffer_length, len(self.energy_envelope))*self.frequency])
        return len(self.ridges), output
