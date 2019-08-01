import sys
import os
from time import time

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

    recordings = Recording.objects.filter(processed=True)
    num_recordings = len(recordings)
    print("Will be attempting to label {} recordings".format(num_recordings))
    
    for recording in recordings:
        label_recording(recording, recording.label_file)

def label_recording(recording, outfile):
    calls = recording.calls.filter(verified=True)
    timestamps = []
    for call in calls:
        timestamps.append([call.start_time, call.end_time])

    if not timestamps:
        return

    print("Found {} calls to label in {}".format(len(calls), recording.label_file))
    with open(outfile, 'w') as file:
        for timestamp in timestamps:
            file.write("{}\t{}\tpika\n".format(timestamp[0], timestamp[1]))





if __name__ == "__main__": main()    
