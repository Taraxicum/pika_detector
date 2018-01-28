from typing import Iterable, Sized

import errno

from numpy.core.multiarray import ndarray

from pika2 import parse_audio
import call_handler as ch
import sys
import os
#import scikits.audiolab
#import wave
import soundfile
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
    print("Will be processing {} recordings".format(len(recordings)))

    for recording in recordings:
        recording.sample_frequency = 44100 #FIXME Need to automatically update when creating the recording object
        handler = ToDB(recording, recording.sample_frequency)
        #handler = ch.CallCounter()
        parse_audio(recording.recording_file, handler)
        recording.processed = True
        recording.save()

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
