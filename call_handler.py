import scikits.audiolab
import numpy as np
import itertools
import abc
import time

class CallHandler(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def handle_call(self, offset, audio):
        """Get offset of identified call and the audio data of the call
        and do something with them."""
        return

    @abc.abstractmethod
    def __enter__(self):
        return

    @abc.abstractmethod
    def __exit__(self, exception_type, exception_val, trace):
        return


class ToDB(CallHandler):
    def __init__(self, recording, db, frequency):
        self.recording = recording
        self.db = db
        self.frequency = frequency
        self.output_path = self.recording.output_folder() + "calls/"
    
    def handle_call(self, offset, audio):
        duration = len(audio)*1.0/frequency
        call = self.db.Call(recording=self.recording, offset=offset,
                duration = duration, filename="temp")
        scikits.audiolab.wavwrite(np.asarray(audio), c.filename, self.frequency)
        call.filename = self.output_path + "call{}.wav".format(call.id)
    
    def __enter__(self):
        return

    def __exit__(self, exception_type, exception_val, trace):
        return

class CallCounter(CallHandler):
    def __init__(self):
        self.count = 0

    def __enter__(self):
        self.count = 0
        self.start_time = time.time()
        return self

    def __exit__(self, exception_type, exception_val, trace):
        print "Total calls identified: {}".format(self.count)
        print "Total elapsed time: {}".format(time.time() - self.start_time)

    def handle_call(self, offset, audio):
        self.count += 1

class ToFile(CallHandler):
    def __init__(self, out_file, frequency):
        self.out_file = out_file
        self.frequency = frequency
        self.current_end = 0.0

    def __enter__(self):
        self.output = []
        self.start_time = time.time()
        return self

    def handle_call(self, offset, audio):
        #print "offset: {}, audio length {}, current_end {}".format(
                #offset, len(audio)*1.0/self.frequency, self.current_end)
        self.output.append(np.zeros(
            max(int((offset - self.current_end)*self.frequency), 0)))
        self.output.append(audio)
        self.current_end = offset + len(audio)*1.0/self.frequency

    def __exit__(self, exception_type, exception_val, trace):
        wav_data = list(itertools.chain.from_iterable(self.output))
        scikits.audiolab.wavwrite(np.asarray(wav_data), self.out_file, 
                self.frequency)
        print "Total elapsed time: {}".format(time.time() - self.start_time)
