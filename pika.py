"""
Adapted from Shankar Shivappa's matlab HarmonicAnalysis.m code to Python
Further adaptations include pre-processing as discussed in "Acoustic Classification of Bird Species
 from syllabels: an Emprirical Study", Briggs, Fern, and Raich; Oregon State University

Usage Example:
    import pika as p
    
    (audio, freq, nBits) = p.load_audio(p.infile)   #here p.infile could be any desired audio file
    p.audio_segments(audio, freq, 10, "trial.wav") 

If you want the output aligned with the original audio (useful for debugging purposes) instead of the 
last line in the example above use:
    p.audio_segments(audio, freq, 10, "trial.wav", True)


Example exploring results:
    import pika as p
    
    (audio, freq, nBits) = p.load_audio(p.infile)   #here p.infile could be any desired audio file
    parser = p.AudioParser(audio[10*freq:30*freq], freq) #loads in the audio from second 10 to 30
    parser.pre_process()
    parser.harmonic_frequency()
    parser.plot_pika_from_harmonic() #Will show plots of the predicted results
    #This may be useful to further optimize predictor, particularly in noisy scenarios
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
infile2 = "LarchMountain_interspeciesCalls.wav"

def load_audio(filepath):
    (snd, sampFreq, nBits) = scikits.audiolab.wavread(filepath)
    if len(snd[0]) == 2: #get left channel if a stereo file not needed for mono
        snd = [v[0] for v in snd] 
    print "sample frequency of {}: {}".format(filepath, sampFreq)
    return (snd, sampFreq, nBits)

def audio_segments(audio, freq, segment_length=10, 
        harmonic_file="output/harmonic.wav", original_placement=False):
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
        count, out = parser.find_pika_from_harmonic(.1, original_placement=original_placement)
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
    def __init__(self, audio, frequency, debug=False):
        """Audio should be a single channel of raw audio data.
        """
        self.audio = audio
        self.frequency = frequency
        self.debug = debug
    
    def plot_energy(self):
        factor = self.step_size*1.0/self.frequency
        plt.plot([i*factor for i in range(len(self.energy_envelope))], self.energy_envelope, 'ro')
        plt.show()
    
    def spectrogram(self):
        factor = self.step_size*1.0/self.frequency
        #plt.subplot(1, 2, 1)
        #plt.imshow(np.asarray(self.fft).T, origin='lower')
        #plt.xticks(plt.xticks()[0], [str(int(t*factor)) for t in plt.xticks()[0]])
        #plt.xlim(0, len(self.fft))
        #plt.xlabel("Chopped fft")
        #x = [i*factor for i in range(len(self.fft))]
        #plt.subplot(1, 3, 2)
        #plt.imshow(np.asarray([f[0:500] for f in self.fft]).T, origin='lower')
        #plt.xticks(plt.xticks()[0], [str(int(t*factor)) for t in plt.xticks()[0]])
        #plt.xlim(0, len(self.fft))
        #plt.xlabel("Further chopped fft")
        #plt.subplot(1, 2, 2)
        plt.imshow(np.asarray([f for f in self.processed_fft_frames]).T, origin='lower')
        plt.xticks(plt.xticks()[0], [str(int(t*factor)) for t in plt.xticks()[0]])
        plt.xlim(0, len(self.fft))
        plt.xlabel("Processed FFT")
        plt.show()

    
    def pre_process(self):
        """Transform (fft), filter, and normalize audio
        """
        fft_size = 4096
        self.step_size = fft_size/16
        first_dim = len(self.audio)/(self.step_size) - 1
        second_dim = 275 # fft_size/2 - fft_size/32
        self.fft = np.zeros((first_dim, second_dim))
        self.processed_fft_frames = np.zeros((first_dim, second_dim+12)) #extra values from convolve (I think?)
        for i in range(0, len(self.audio) - fft_size, self.step_size):
            f = np.absolute(np.fft.fft(self.audio[i:i+fft_size]))    #Magnitude of fft
            f = f[fft_size/32:fft_size/2]                         #Chop out some unneeded frequencies
            #f = np.log10(f/sum(f))
            self.fft[i/self.step_size] = f[150:425] #This will need to be changed if fft_size changes!
        max_val = np.amax(self.fft)
        normed_fft = self.fft/max_val
        avg_fft = np.sum(normed_fft, axis=0)/len(normed_fft)
        nr_fft = np.zeros_like(self.fft)
        for i, frame in enumerate(normed_fft):
            nr_fft[i] = [max(frame[j] - avg_fft[j], 0) for j, v in enumerate(frame)] #noise-reduction
            #if i%30 == 0:
            #    plt.subplot(1,2,1)
            #    plt.plot(self.fft[i])
            #    plt.subplot(1,2,2)
            #    plt.plot(nr_fft[i])
            #    plt.show()
            #    print "showing plot at time: {}".format((i*fft_size/2.0)/self.frequency)
        
        f_mean = np.mean(normed_fft)
        #.15 threshold found through trial and error, could be adjusted for
        # slightly better results
        threshold = .05
        tf = [[x if x > f_mean + threshold else 0.0 for x in f] for f in nr_fft] 
        self.processed_fft_frames = tf


        ###########TEMP###################
        ##Uncomment to view spectrograms of different filterings
        #factor = self.step_size*1.0/self.frequency
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
        self.energy_envelope = np.convolve(np.sum([f[0:500] for f in self.processed_fft_frames], axis=1), 
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
        factor = self.step_size*1.0/self.frequency #mutliply i by factor to get time of the frame's location in seconds 
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
        self.ridges = []
        factor = self.step_size*1.0/self.frequency #mutliply i by factor to get time of the frame's location in seconds 
        self.peaks = []
        current_ridge = None
        for i, f in enumerate(self.processed_fft_frames):
            locs = peaks.detect_peaks(f, mpd=50)
            self.peaks.append(locs)
            if self.debug and len(locs) > 2 and len(locs) < 4:
                inter_peak_dist = np.convolve(locs, [1, -1])
                print "locs: {}".format(locs)
                print "FEW PKS, ipd: {}, i: {}, time: {}".format(inter_peak_dist, i, i*factor)
            
            if len(locs) >= 4 and locs[0] > 30: 
                inter_peak_dist = np.convolve(locs, [1, -1])
                harmonic_freq =  np.median(inter_peak_dist[1:-1])
                if (harmonic_freq < 75) and (harmonic_freq > 55):
                    if self.debug:
                        print "YES, ipd: {}, i: {}, time: {}".format(inter_peak_dist, i, i*factor)
                    if current_ridge is None:
                        current_ridge = i*factor
                elif current_ridge is not None:
                    self.ridges.append([current_ridge, i*factor])
                    current_ridge = None
                elif self.debug:
                    print "WRONG H-FREQS, ipd: {}, i: {}, time: {}".format(inter_peak_dist, i, i*factor)
            elif current_ridge is not None:
                self.ridges.append([current_ridge, i*factor])
                current_ridge = None
        if current_ridge is not None: 
            self.ridges.append([current_ridge, len(self.processed_fft_frames)*factor])

    def filter_short_ridges(self, threshold=.07):
        """Detected calls should be in self.ridges (as done in self.harmonic_frequency).
        This function will go through self.ridges and eliminate all ridges shorter than
        threshold (in seconds). The pika calls I have seen so far have been around .2 seconds.
        self.consolidate_ridges should be called before this function so that a ridge that
        was broken by a false negative frame will be less likely to get falsely eliminated here.
        """
        self.ridges[:] = [r for r in self.ridges if r[1] - r[0] > threshold]

    
    def consolidate_ridges(self, threshold=.07):
        """Detected calls should be in self.ridges (as done in self.harmonic_frequency).
        This function will go through the ridges and if the end of one ridge is within
        threshold seconds of the next it will combine the ridges.
        This is useful for when the detector has a false negative in the middle of a call 
        (perhaps due to microphone noise or other issue) and so ends the detected call,
        but immediately starts up the next call when it begins detecting it again.
        """
        new_ridges = []
        ridge_count = len(self.ridges)
        if ridge_count == 0:
            return #Nothing to do - no detected pika calls to consolidate
        
        current_ridge = self.ridges[0]
        for i, r in enumerate(self.ridges):
            if i + 1 < ridge_count:
                if current_ridge[1] + threshold >= self.ridges[i+1][0]:
                    current_ridge[1] = self.ridges[i+1][1]
                else:
                    new_ridges.append(current_ridge)
                    current_ridge = self.ridges[i+1]
            else:
                new_ridges.append(current_ridge)
        self.ridges = new_ridges

    
    def find_pika_from_energy(self, filename='test.wav'):
        self.energy_segments()
        output = []
        m = max(self.audio)
        for r in self.energy_ridges:
            output.append(m)
            output.extend(self.audio[max(0, r[0]-.01)*self.frequency:
                min(r[1]+.01, len(self.energy_envelope))*self.frequency])
        return output
   
    def plot_pika_from_harmonic(self):
        """Uses audio predicted to be pika calls found in self.ridges (populated with self.harmonic_frequency())
         for each predicted call, plots self.fft, self.processed_fft_frames, and the spectrogram from
         self.processed_fft_frames with a surrounding buffer of 30 additional frames on each side of the 
         predicted call.
        """
        factor = self.step_size*1.0/self.frequency 
        for r in self.ridges:
            mid = (r[1] + r[0])/2
            mid_frame = int(mid/factor)
            first = r[0]
            first_frame = int(first/factor)
            end_frame = int(r[1]/factor)
            print "showing plot at frame {}, time: {}".format(first_frame, first)
            
            plt.subplot(1,3,1)
            plt.plot(self.fft[mid_frame])
            plt.subplot(1,3,2)
            plt.plot(self.processed_fft_frames[mid_frame])
            
            plt.subplot(1,3,3)
            plt.imshow(np.asarray(self.processed_fft_frames[first_frame-30:end_frame+30]).T, origin='lower')
            plt.xlabel("pika?")
            plt.show()
    
    
    def find_pika_from_harmonic(self, buffer_length=.01, original_placement=False):
        """pre_process() must have been called first.
        :returns list containing segments of self.audio thought to contain pika calls
        with buffer_length (in seconds) space on each side of the pika call 
        (buffered with the signal in audio).
        :original_placement if True outputs detected calls in same spacing as in original 
        signal with 0s padding between calls.  This may be useful for comparing output to 
        original to get better insight into false positives/negatives
        """
        self.harmonic_frequency()
        self.consolidate_ridges()
        self.filter_short_ridges()

        if len(self.ridges) == 0:
            output = np.zeros(len(self.audio))
            return 0, output
        else:
            m = max(self.audio)
            print "Number of incidents being output: {}".format(len(self.ridges))
            last_endpoint = 0
            output = []
            for i, r in enumerate(self.ridges):
                if original_placement:
                    output.extend(np.zeros((r[0] - last_endpoint)*self.frequency))
                    output.extend(self.audio[int(r[0]*self.frequency): int(r[1]*self.frequency)])
                    last_endpoint = r[1]
                else:
                    output.append(m)
                    output.extend(self.audio[int(max(0, r[0] - buffer_length)*self.frequency):
                        int(min(r[1] + buffer_length, len(self.processed_fft_frames))*self.frequency)])
            if original_placement:
                output.extend(np.zeros(len(self.audio) - last_endpoint*self.frequency))

            return len(self.ridges), output
