"""
Adapted from Shankar Shivappa's matlab HarmonicAnalysis.m code to Python
Further adaptations include pre-processing as discussed in "Acoustic Classification of Bird Species
 from syllabels: an Emprirical Study", Briggs, Fern, and Raich; Oregon State University

Usage Example:
    import pika as p
    
    (audio, freq, nBits) = p.load_audio("input.wav")
    p.audio_segments(audio, freq, 10, "output.wav") 

If you want the output aligned with the original audio (useful for debugging purposes) instead of the 
last line in the example above use:
    p.audio_segments(audio, freq, 10, "output.wav", True)


Example exploring results:
#This may be useful to further optimize predictor, particularly in noisy scenarios
    import pika as p
    
    (audio, freq, nBits) = p.load_audio("input.wav")
    parser = p.AudioParser(audio, freq)
    parser.pre_process()
    parser.harmonic_frequency()
    parser.plot_pika_from_harmonic() #Will show plots of the predicted results
"""
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import scikits.audiolab
import find_peaks as peaks
import pika_db_models as db
import matplotlib.pyplot as plt
import time
import os
import glob
#from pylab import specgram

#infile = "May26-2014-BeaconRockSP4800.wav"
infile = "May26-2014-BeaconRockSP_shankars.wav"
#infile = "./input/July26-2014-LarchMountain/..."
infile2 = "LarchMountain_interspeciesCalls.wav"

def load_and_parse(recording):
    """Takes Recording object which should have been already preprocessed and further processes
    the results of the preprocessing looking for pika calls.
    :recording: Recording db object
    """
    path = os.path.dirname(recording.filename) + "\\recording{}\\".format(recording.id)
    wav_files = glob.glob(path + "offset_*.wav") #should be of the form e.g. 'offset_3.0.wav'
    for f in wav_files:
        just_file = os.path.basename(f)
        offset = float(just_file[7:just_file.find(".w")])
        (snd, freq, bits) = load_audio(f)
        audio_segments(snd, freq, offset, recording, 10)

def load_audio(filepath):
    (snd, sampFreq, nBits) = scikits.audiolab.wavread(filepath)
    if snd.ndim == 2: #get left channel if a stereo file not needed for mono
        snd = [v[0] for v in snd] 
    print "sample frequency of {}: {}".format(filepath, sampFreq)
    return (snd, sampFreq, nBits)

def audio_segments(audio, freq, offset, recording, segment_length=10):
    """Segments audio into segment_length (in seconds) chunks and runs the 
    algorithm on each chunk.  Creates new Call db object and a file for each
    identified call.  The wav file containing the call will have its filename
    in the database and should be in the path associated with recording
    :audio: the audio file to be processed
    :freq: frequency of the audio file
    :offset: offset of audio being processed from beginning of original file
    :recording: db object of recording containing audio being processed
    :segment_length: length in seconds of the segments to run the algorithm on

    :recording: db object of the recording being processed - should have been
    preprocessed before this point.
    """
    harmonic_out = []
    total = 0
    start_time = time.time()
    path = os.path.dirname(recording.filename) + "\\recording{}\\".format(recording.id)

    for i in range(0, len(audio)/freq, segment_length):
        end = min(len(audio), (i+segment_length)*freq)
        parser = AudioParser(audio[i*freq:end], freq)
        parser.pre_process()
        count, c_offsets, out = parser.find_pika_from_harmonic(.1)
        harmonic_out.append(out)
        total += count
    print "Total suspected calls: {}, total processing time (in seconds): {}".format(total,
            time.time() - start_time)
    if total > 0:
        for i, call in enumerate(out):
            c_offset = float(offset + c_offsets[i])
            c_duration = float(len(call)*1.0/freq)
            c = db.Call(recording=recording, offset=c_offset, duration=c_duration, filename="temp")
            c.filename = path + "call{}.wav".format(c.id)
            scikits.audiolab.wavwrite(np.asarray(call), c.filename, freq)

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
        self.fft_size = 4096
        self.step_size = self.fft_size/16
        self.factor = self.step_size*1.0/self.frequency
    
    def plot_energy(self):
        plt.plot([i*self.factor for i in range(len(self.energy_envelope))], self.energy_envelope, 'ro')
        plt.show()
    
    def spectrogram(self):
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
        plt.xticks(plt.xticks()[0], [str(int(t*self.factor)) for t in plt.xticks()[0]])
        plt.xlim(0, len(self.fft))
        plt.xlabel("Processed FFT")
        plt.show()

    
    def pre_process(self):
        """Transform (fft), filter, and normalize audio
        """
        first_dim = len(self.audio)/(self.step_size) - 1
        second_dim = 275 # fft_size/2 - fft_size/32
        self.fft = np.zeros((first_dim, second_dim))
        self.processed_fft_frames = np.zeros((first_dim, second_dim+12)) #extra values from convolve (I think?)
        for i in range(0, len(self.audio) - self.fft_size, self.step_size):
            f = np.absolute(np.fft.fft(self.audio[i:i+self.fft_size]))    #Magnitude of fft
            f = f[self.fft_size/32:self.fft_size/2]                         #Chop out some unneeded frequencies
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
        #length = len(normed_fft)
        #plt.subplot(1, 4, 1)
        #plt.imshow(np.asarray([f[0:500] for f in normed_fft]).T, origin='lower')
        #plt.xticks(plt.xticks()[0], [str(int(t*self.factor)) for t in plt.xticks()[0]])
        #plt.xlim(0, length)
        #plt.xlabel("fft: chop, log, threshold, convolve")
        ##################################
        #plt.subplot(1, 4, 2)
        #plt.imshow(np.asarray([f[0:500] for f in normed_fft]).T, origin='lower')
        #plt.xticks(plt.xticks()[0], [str(int(t*self.factor)) for t in plt.xticks()[0]])
        #plt.xlim(0, length)
        #plt.xlabel("fft: and normed by max")
        ##################################
        #plt.subplot(1, 4, 3)
        #plt.imshow(np.asarray([f[0:500] for f in nr_fft]).T, origin='lower')
        #plt.xticks(plt.xticks()[0], [str(int(t*self.factor)) for t in plt.xticks()[0]])
        #plt.xlim(0, length)
        #plt.xlabel("fft: and subtract avg fft")
        ##################################
        #plt.subplot(1, 4, 4)
        #plt.imshow(np.asarray([f[0:500] for f in root_nr_fft]).T, origin='lower')
        #plt.xticks(plt.xticks()[0], [str(int(t*self.factor)) for t in plt.xticks()[0]])
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
        #plt.plot([i*self.factor for i in range(len(self.energy_envelope))], self.energy_envelope, 'b.')
        #plt.xlim(0, 60)
        #plt.xlabel("Energy")
        ##################################
        #e_mean = np.mean(self.energy_envelope)                                 
        #thresh_energy = [1.0 if x > 1.7*e_mean else 0.0 for x in self.energy_envelope]
        #plt.subplot(2, 1, 2)
        #plt.plot([i*self.factor for i in range(len(self.energy_envelope))], thresh_energy, 'g.')
        #plt.ylim(0, 1.2)
        #plt.xlim(0, 60)
        #plt.xlabel("Energy Threshold")
        #plt.show()
        ##################################

    def energy_segments(self):
        """Uses energy envelopes to attempt to identify locations of pika calls
        """
        energy_threshold = 1.7*np.average(self.energy_envelope)
        self.energy_ridges = []
        current_ridge = None
        for i, e in enumerate(self.energy_envelope):
            if e > energy_threshold:
                if current_ridge is None:
                    current_ridge = i*self.factor
            else:
                if current_ridge is not None:
                    self.energy_ridges.append([current_ridge, i*self.factor])
                    current_ridge = None
        if current_ridge is not None:
            self.energy_ridges.append([current_ridge, len(self.energy_envelope)*self.factor])
            current_ridge = None
    
    def harmonic_frequency(self):
        """Uses harmonic frequencies to attempt to identify locations of pika calls
        """
        self.ridges = []
        self.peaks = []
        current_ridge = None
        for i, f in enumerate(self.processed_fft_frames):
            locs = peaks.detect_peaks(f, mpd=50)
            self.peaks.append(locs)
            if self.debug and len(locs) > 2 and len(locs) < 3:
                inter_peak_dist = np.convolve(locs, [1, -1])
                print "locs: {}".format(locs)
                print "FEW PKS, ipd: {}, i: {}, time: {}".format(inter_peak_dist, i, i*self.factor)
            
            if len(locs) >= 3 and locs[0] > 30 and locs[0] < 60: 
                inter_peak_dist = np.convolve(locs, [1, -1])
                harmonic_freq =  np.median(inter_peak_dist[1:-1])
                if all((((ipd < 75) and (ipd > 55)) or
                    ((ipd < 135) and (ipd > 110))) for ipd in inter_peak_dist[1:-1]):
                    if self.debug:
                        print "YES, ipd: {}, i: {}, time: {}".format(inter_peak_dist, i, i*self.factor)
                    if current_ridge is None:
                        current_ridge = i*self.factor
                elif current_ridge is not None:
                    self.ridges.append([current_ridge, i*self.factor])
                    current_ridge = None
                elif self.debug:
                    print "WRONG H-FREQS, ipd: {}, i: {}, time: {}".format(inter_peak_dist, i, i*self.factor)
            elif current_ridge is not None:
                self.ridges.append([current_ridge, i*self.factor])
                current_ridge = None
        if current_ridge is not None: 
            self.ridges.append([current_ridge, len(self.processed_fft_frames)*self.factor])

    def filter_short_ridges(self, threshold=.10):
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
        for r in self.ridges:
            mid = (r[1] + r[0])/2
            mid_frame = int(mid/self.factor)
            first = r[0]
            first_frame = int(first/self.factor)
            end_frame = int(r[1]/self.factor)
            print "showing plot at frame {}, time: {}".format(first_frame, first)
            
            plt.subplot(1,3,1)
            plt.plot(self.fft[mid_frame])
            plt.subplot(1,3,2)
            plt.plot(self.processed_fft_frames[mid_frame])
            
            plt.subplot(1,3,3)
            plt.imshow(np.asarray(self.processed_fft_frames[first_frame-30:end_frame+30]).T, origin='lower')
            plt.xlabel("pika?")
            plt.show()
    
    
    def find_pika_from_harmonic(self, buffer_length=.01):
        """pre_process() must have been called first.
        :returns list containing segments of self.audio thought to contain pika calls
        with buffer_length (in seconds) space on each side of the pika call 
        (buffered with the signal in audio).
        """
        self.harmonic_frequency()
        if self.debug:
            print "Ridges before consolidation: {}".format(self.ridges)
        self.consolidate_ridges()
        if self.debug:
            print "Ridges after consolidation: {}".format(self.ridges)
        self.filter_short_ridges()
        if self.debug:
            print "Ridges after filtering out short ridges: {}".format(self.ridges)

        if len(self.ridges) == 0:
            return 0, [], []
        else:
            print "Number of incidents being output: {}".format(len(self.ridges))
            last_endpoint = 0
            output = []
            offsets = []
            for i, r in enumerate(self.ridges):
                offsets.append(max(0, r[0] - buffer_length))
                start = offsets[-1]*self.frequency
                end = min(r[1] + buffer_length, len(self.processed_fft_frames))*self.frequency
                output.append(self.audio[int(start):int(end)])
            return len(self.ridges), offsets, output
