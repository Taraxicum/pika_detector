import numpy as np
import find_peaks as peaks
import matplotlib.pyplot as plt

class AudioParser(object):
    """For taking chunks of audio then pre-processing, identifying and outputting pika calls.
    Long sections of audio are tough on memory usage, so pre-chopping longer audio into 10 second
    or so chunks is recommended.
    
    Example usage:
    parser = p.AudioParser(audio, freq)
    parser.pre_process()
    """
    def __init__(self, audio, frequency, debug=False, mpd=None, ipd_filters=None):
        """Audio should be a single channel of raw audio data.
        """
        self.audio = audio
        self.frequency = frequency
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
            #self.ipd_filters = [[51, 75], [110, 135]] 
            self.ipd_filters = [[54, 70]] 
        else:
            self.ipd_filters = ipd_filters
    
    def spectrogram(self, label=None):
        plt.imshow(np.asarray([f for f in self.processed_fft_frames]).T,
                origin='lower')
        plt.xticks(plt.xticks()[0], ["{0:.2f}".format(t*self.factor) 
            for t in plt.xticks()[0]])
        plt.xlim(0, len(self.fft))
        if label is None:
            plt.xlabel("Processed FFT")
        else:
            plt.xlabel(label)
        plt.show(block=False)

    
    def pre_process(self, do_norm=True, do_filter=True, do_nr = True):
        """Transform (fft), filter, and normalize audio
        """
        first_dim=np.ceil(1.0*(len(self.audio))/(self.step_size))
        #second_dim = self.fft_size/2-self.fft_size/32
        second_dim = 275
        self.fft = np.zeros((first_dim, second_dim))
        #self.processed_fft_frames = np.zeros((first_dim, second_dim))
        #for i in xrange(0, len(self.audio) - self.fft_size, self.step_size):
        for i in xrange(0, len(self.audio), self.step_size):
            f = np.absolute(np.fft.fft(self.audio[i:i+self.fft_size],
                self.fft_size))
            f = f[self.fft_size/32:self.fft_size/2][150:425] #This will need
                                        #to be changed if fft_size changes!
            self.fft[i/self.step_size] = f 
        if do_norm:
            max_val = np.amax(self.fft)
            self.fft = self.fft/max_val
        
        if do_nr: #noise-reduction
            avg_fft = np.sum(self.fft, axis=0)/len(self.fft)
            nr_fft = np.zeros_like(self.fft)
            for i, frame in enumerate(self.fft):
                self.fft[i] = [max(frame[j] - avg_fft[j], 0)
                        for j, v in enumerate(frame)] 
        
        if do_filter:
            f_mean = np.mean(self.fft)
            threshold = .05
            self.fft = [[x if x > f_mean + threshold else 0.0 for x in f]
                    for f in self.fft] 
        self.processed_fft_frames = self.fft

    def score_frame(self, frame, i):
        """Scores a frame for how likely it seems to be part of a pika call
        :frame: numpy 1d array fft frame
        :returns likeliness score
        """
        score = 0.0
        locs = peaks.detect_peaks(frame, mpd=self.mpd)
        if self.debug:
            if len(locs) == 0:
                1+2 == 3
                #print "no peaks detected at {:.3f}".format(i*self.factor)
            else:
                ipd = np.convolve(locs, [1, -1])
                print "t: {:.3f}, ipd {}".format(i*self.factor, ipd)
        
        if len(locs) >= 3 and locs[0] < 75: 
            ipd = np.convolve(locs, [1, -1])
            #harmonic_freq =  np.median(ipd[1:-1])
            amount = 3.0/(len(locs) - 2)
            for x in ipd:
                if any((x >= bot) and (x <= top) for bot, top in self.ipd_filters):
                    score += amount

            
            #if all(any((ipd >= bot) and (ipd <= top) for bot, top in self.ipd_filters)
            #        for ipd in inter_peak_dist[1:-1]):
        return score
    
    def harmonic_frequency(self):
        """
        Uses harmonic frequencies to attempt to identify locations of pika calls
        """
        frame_scores = []
        for i, f in enumerate(self.processed_fft_frames):
            frame_scores.append(self.score_frame(f, i))
        if len(frame_scores) > 0:
            frame_scores = np.convolve(frame_scores,
                    [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5], mode='same')
        if self.debug:
            print "Frame scores\n{}".format(frame_scores)
        self.ridges = self.find_ridges(frame_scores)
        #TODO rather than manually doing ridges in the frame scoring
        #process, try convolving along the frame scores and filter
        #on a particular threshold to find ridges of likely calls
    
    def find_ridges(self, scores):
        #scores = [1 if x > threshold else 0 for x in scores]
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
        return ridges

    def filter_short_ridges(self, threshold=.10):
        """
        Detected calls should be in self.ridges (as done in
          self.harmonic_frequency).
        This function will go through self.ridges and eliminate all ridges 
          shorter than threshold (in seconds).  The pika calls I have seen
          so far have been around .2 seconds. self.consolidate_ridges should
          be called before this function so that a ridge that was broken by
          a false negative frame will be less likely to get falsely 
          eliminated here.
        """
        self.ridges[:] = [r for r in self.ridges if r[1] - r[0] > threshold]

    
    def consolidate_ridges(self, threshold=.02):
        """
        Detected calls should be in self.ridges (as done in 
          self.harmonic_frequency).
        This function will go through the ridges and if the end of one ridge 
          is within threshold seconds of the next it will combine the ridges.
          This is useful for when the detector has a false negative in the 
          middle of a call (perhaps due to microphone noise or other issue)
          and so ends the detected call, but immediately starts up the next
          call when it begins detecting it again.
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

    
    def plot_pika_from_harmonic(self):
        """
        Uses audio predicted to be pika calls found in self.ridges 
          (populated with self.harmonic_frequency()) for each predicted call,
          plots self.fft, self.processed_fft_frames, and the spectrogram from
          self.processed_fft_frames with a surrounding buffer of 30 additional
          frames on each side of the predicted call.
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
    
    
    def find_pika_from_harmonic(self, buffer_length=.10):
        """pre_process() must have been called first.
        :returns list containing segments of self.audio thought to contain 
          pika calls with buffer_length (in seconds) space on each side of the 
          pika call (buffered with the signal in audio).
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

