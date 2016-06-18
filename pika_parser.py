import utility as u
import find_peaks as peaks
import numpy as np
import matplotlib.pyplot as plt
import scikits.audiolab
import processing as p
import os


class PikaParser(object):
    """In general this class should probably not be used directly.  See
    pika.py for functions intended for direct use which make use of
    this class.

    This class is For taking chunks of audio then changing to frequency domain
    and performing tasks.  The main task is filtering, identifying
    and outputting pika calls.  Another task is providing an interface for
    verification of previously identify pika calls.

    Long sections of audio are tough on memory usage, so longer audio is
    pre-chopped into 10 second or shorter chunks.  This can cause issues if
    a pika call is on a bounder between chunks.  If necessary the code could
    be adjusted to deal with that situation, but for now it will be ignored
    in favor of getting things working otherwise.
    """
    #*Constructor*#
    def __init__(self, recording, audio_file, database, debug=False,
            mpd=None, ipd_filters=None):
        """Audio should be a single channel of raw audio data.
        """
        self.recording = recording
        self.full_audio = self.load_audio(audio_file)
        self.db = database
        self.frequency = self.recording.bitrate
        self.debug = debug
        self.fft_size = 4096
        self.step_size = self.fft_size/64
        self.factor = self.step_size*1.0/self.frequency
        
        if mpd is None:
        #minimum peak distance for calculating harmonic frequencies
            self.mpd = 40 
        else:
            self.mpd = mpd
        
        if ipd_filters is None:
        #each ipd must fall within one of the ipd_filter ranges to be 
        #considered a successful candidate for a pika call
            self.ipd_filters = [[54, 70], [110, 135]] 
            #self.ipd_filters = [[54, 70]] 
        else:
            self.ipd_filters = ipd_filters
    
    #*Public Methods*#
    def identify_and_write_calls(self):
        for chunk, offset in p.segment_audio(self.full_audio, self.frequency):
            fft = self.filtered_fft(chunk)
            frame_scores = self.score_fft(fft)
            good_intervals = self.find_passing_intervals(frame_scores)
            self.write_calls(chunk, offset, good_intervals)
        return
    
    def verify_call(self, call, with_audio=True):
        plt.ion()
        fft = self.filtered_fft(self.full_audio)
        self.spectrogram(fft, call.filename)
        response = u.get_verification(call, with_audio)
        plt.close()
        return response
    
    #*Private Methods*#
    def load_audio(self, audio_file):
        (audio, bitrate, nBits) = scikits.audiolab.wavread(audio_file)
        if audio.ndim == 2: 
            #get left channel if a stereo file not needed for mono
            audio = [v[0] for v in audio] 
        return audio
    
    def filtered_fft(self, audio):
        first_dim=np.ceil(1.0*(len(audio))/(self.step_size))
        second_dim = 275
        fft = np.zeros((first_dim, second_dim))
        for i in xrange(0, len(audio), self.step_size):
            f = np.absolute(np.fft.fft(audio[i:i+self.fft_size], self.fft_size))
            f = f[self.fft_size/32:self.fft_size/2][150:425] #This will need
                                        #to be changed if fft_size changes!
            fft[i/self.step_size] = f 
        
        #normalize
        max_val = np.amax(fft)
        fft = fft/max_val
        
        #noise-reduction
        avg_fft = np.sum(fft, axis=0)/len(fft)
        for i, frame in enumerate(fft):
            fft[i] = [max(frame[j]-avg_fft[j], 0) for j, v in enumerate(frame)] 
        
        #filter out quiet parts
        f_mean = np.mean(fft)
        threshold = .05
        return [[x if x > f_mean + threshold else 0.0 for x in f] for f in fft] 
    
    def score_fft(self, fft):
        """Scores frames for how likely they seem to be part of a pika call
        :fft: numpy 2d array fft
        :debug: if True outputs statements to help debug scoring process
        :returns 1d array of likeliness scores corresponding to the frames
        """
        scores = []
        for i, frame in enumerate(fft):
            score = 0.0
            locs = peaks.detect_peaks(frame, mpd=self.mpd)
            if self.debug:
                if len(locs) != 0:
                    ipd = np.convolve(locs, [1, -1])
                    print "t: {:.3f}, ipd {}".format(i*self.factor, ipd)
            
            if len(locs) >= 3 and locs[0] < 75: 
                ipd = np.convolve(locs, [1, -1])
                amount = 3.0/(len(locs) - 2)
                for x in ipd:
                    if any((x >= bot) and (x <= top) 
                            for bot, top in self.ipd_filters):
                        score += amount

            scores.append(score)
        return scores
    
    def find_passing_intervals(self, frame_scores):
        """
        :frame_scores: pika call likelihood scores of fft frames.
        :returns list of intervals that are likely to contain a pika call
        """
        scores = np.convolve(frame_scores,
                [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5], mode='same')
        ridges = []
        current_ridge = None
        threshold = 10.5
        for i, s in enumerate(scores):
            if current_ridge is not None:
                if s < threshold:
                    ridges.append([current_ridge, (i -1)*self.factor])
                    current_ridge = None
            else:
                if s > threshold:
                    current_ridge = i*self.factor
        if current_ridge is not None:
            ridges.append([current_ridge, len(scores)*self.factor])
        min_ridge_length = .1
        ridges[:] = [r for r in ridges if r[1] - r[0] > min_ridge_length]
        return ridges
    
    def write_calls(self, audio, offset, intervals):
        """
        :offset: offset of audio with respect to the original recording
        :intervals: list of intervals in seconds (e.g. [[2.4,2.71], [5.9,6.18]])
        where audio contains identified pika calls

        Creates Call objects for each of the intervals and writes the call audio
        as a wav file into a call subdirectory of the recording directory with
        filename callN.wav where N is the call id in the database.
        """

        output_path = self.recording.output_folder() + "calls/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for interval in intervals:
            c_offset = float(offset + interval[0])
            c_duration = float(interval[1] - interval[0])
            c = self.db.Call(recording=self.recording, offset=c_offset,
                    duration=c_duration, filename="temp")
            c.filename = output_path + "call{}.wav".format(c.id)
            scikits.audiolab.wavwrite( np.asarray(audio[
                        int(interval[0]*self.frequency)
                        :int(interval[1]*self.frequency)
                        ]), c.filename, self.frequency)
        
    def spectrogram(self, fft, label=None):
        plt.imshow(np.asarray([f for f in fft]).T,
                origin='lower')
        plt.xticks(plt.xticks()[0], ["{0:.2f}".format(t*self.factor) 
            for t in plt.xticks()[0]])
        plt.xlim(0, len(fft))
        if label is None:
            plt.xlabel("Processed FFT")
        else:
            plt.xlabel(label)
        plt.show(block=False)
