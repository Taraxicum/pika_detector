from typing import Iterable, Sized

import errno

from numpy.core.multiarray import ndarray

from pika2 import parse_audio
import call_handler as ch
import sys
import os
from time import time
#import scikits.audiolab
#import wave
import soundfile
from soundfile import SoundFile
import numpy as np

if __name__== '__main__':
    #Got this setup from:
    #https://www.stavros.io/posts/standalone-django-scripts-definitive-guide/
    proj_path = "D:/Workspace/pika/pika_project/"
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pika_project.settings")
    sys.path.append(proj_path)
    os.chdir(proj_path)

    from django.core.wsgi import get_wsgi_application
    application = get_wsgi_application()

from pika_app.models import Recording, Call

def main(argv=None):

    recordings = Recording.objects.filter(processed=False)
    #TODO before processing, have interface which states number of files
    #to be processed and total length of recordings (and maybe estimate
    # of how long it will take?) and verify user wants to proceed.
    num_recordings = len(recordings)
    print("Will be processing {} recordings".format(num_recordings))
    
    start_time = time()
    for recording in recordings:
        rec_time = time()
        rec_soundfile = SoundFile(recording.recording_file)
        # TODO Here or elsewhere, automatically parse datetime from filename and/or details
        recording.sample_frequency = rec_soundfile.samplerate
        handler = ToDB(recording, recording.sample_frequency)
        #handler = ch.CallCounter()
        parse_audio(recording.recording_file, handler)
        recording.processed = True
        recording.save()
        print("processed this recording in {} seconds".format(time() - rec_time))
    print("finished processing batch in {} seconds".format(time() - start_time))


class ToDB(ch.CallHandler):
    def __init__(self, recording, frequency):
        # type: (Recording, int) -> None
        self.recording = recording
        self.frequency = frequency
        
        self.output_path = os.path.join(self.recording.output_folder(),
                "calls{}".format(os.sep))
        if not os.path.exists(self.output_path):
            try:
                os.makedirs(self.output_path)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
            
    
    def handle_call(self, offset, audio):
        # type: (float, ndarray) -> None
        duration = len(audio)*1.0/self.frequency
        #import pdb; pdb.set_trace()
        call = Call(recording=self.recording, offset=offset,
                    duration=duration, filename="temp")
        call.save()

        call.filename = self.output_path + "call{}.wav".format(call.id)
        call.save()
        soundfile.write(call.filename, np.asarray(audio), self.frequency)

    def __enter__(self):
        # type: () -> ch.CallHandler
        return self

    def __exit__(self, exception_type, exception_val, trace):
        # type: (...) -> None
        return



if __name__ == "__main__": main()    
