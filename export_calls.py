import os
from pytz import timezone
import shutil
import sys

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

def call_time(call):
    # type: (Call) -> datetime
    result = call.call_time.astimezone(timezone("US/Pacific"))
    return result

def main(argv=None):
    recordings = Recording.objects.filter(processed=True)
    #TODO before processing, have interface which states number of files
    #to be processed and total length of recordings (and maybe estimate
    # of how long it will take?) and verify user wants to proceed.

    destination_folder = "d:\\Workspace\\pika\\good_calls"
    for recording in recordings:
        for call in recording.calls.filter(verified=True).order_by('id'):
            print_row(recording, call)
            copy_call(call, destination_folder)

def print_row(record, call):
    # type: (Recording, Call) -> None
    """
    """
    print("{}".format(call_time(call).strftime("%H:%M:%S %B %d, %Y")))

def copy_call(call, destination_folder):
    # type: (Call, str) -> None
    call_output_filename = "{}-{}.wav".format(call.id, call_time(call).strftime("%Y_%m_%d-%H_%M_%S"))
    assert os.path.isdir(destination_folder)
    to_file = os.path.join(destination_folder, call_output_filename)

    shutil.copy2(call.filename, to_file)

if __name__ == "__main__": main()
