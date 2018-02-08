import os
import glob
import sys

import re
from datetime import datetime

import shutil
from soundfile import SoundFile

if __name__== '__main__':
    #Got this setup from:
    #https://www.stavros.io/posts/standalone-django-scripts-definitive-guide/
    proj_path = "D:/Workspace/pika/pika_project/"
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pika_project.settings")
    sys.path.append(proj_path)
    os.chdir(proj_path)

    from django.core.wsgi import get_wsgi_application
    application = get_wsgi_application()

from pika_app.models import Collection, Recording, Observer, recording_path_by_collection


def main():
    assert len(sys.argv) == 2, "There should be one argument - the folder containing the collection"
    directory = sys.argv[1]
    assert isinstance(directory, str), "The argument should be a string"
    assert os.path.isdir(directory), "The argument should be a directory"
    files = glob.glob(os.path.join(directory,  "*.wav"))
    d = parse_filename_for_date(files[0])
    collection = build_collection(directory)

    for file in files:
        build_recording(file, collection)

def build_collection(directory):
    while True:
        observer = get_observer()
        description = get_description()
        collection = Collection(
            observer=observer,
            description=description,
            notes="",
        )
        confirmation = input("{}; {} is the desired observer and description? (Y/n)".format(collection.observer.name,
                                                                                            collection.description))
        if confirmation.upper() == "Y":
            collection.save()
            return collection

def get_observer():
    observers = Observer.objects.all()
    while True:
        for observer in observers:
            print("{}) {}; {}".format(observer.pk, observer.name, observer.institution))
        print("Enter number corresponding to desired observer or enter name if new observer")
        name = input("Observer:")
        try:
            index = int(name)
        except ValueError:
            index = None
        if index is not None:
            observer = Observer.objects.get(pk=index)
            name = observer.name
            break
        else:
            confirmation = input("{} is the desired observer? (Y/n)".format(name))
            if confirmation.upper() == "Y":
                if index is None:
                    observer = Observer.objects.create(name=name)
                    observer.save()
                break
    return observer

def get_description():
    description = input("One-line description of collection:")
    return description

def build_recording(filename, collection):
    soundfile = SoundFile(filename)
    start_time = parse_filename_for_date(filename)

    to_file = recording_path_by_collection(collection, os.path.basename(filename))
    base_path, file = os.path.split(to_file)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    shutil.copy2(filename, to_file)

    recording = Recording(
        collection=collection,
        recording_file=to_file,
        start_time=start_time,
        duration=len(soundfile)/soundfile.samplerate,
        sample_frequency=soundfile.samplerate,
        notes="",
    )
    recording.save()

def parse_filename_for_date(filename):
    name = os.path.basename(filename)
    pattern = re.compile(r".*_(?P<year>20\d{2})(?P<month>\d{2})(?P<day>\d{2})_"
                         r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})\.wav")
    m = pattern.match(name)
    return datetime(year=int(m.group("year")),
                    month=int(m.group("month")),
                    day=int(m.group("day")),
                    hour=int(m.group("hour")),
                    second=int(m.group("second")),
                    )

if __name__ == "__main__": main()
