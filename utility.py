"""Utility functions for pika.py

Mostly UI helper functions

"""
import glob
import datetime
import os
import pika_db_models as db

def set_observations(collection):
    mp3files = glob.glob(collection.folder + "\\*.mp3")
    while len(mp3files) > 0:
        print("\n".join(["{}: {}".format(i, f) for i, f in enumerate(mp3files)]))
        file_numbers = raw_input("Which files should be part of the first observation?")
        file_numbers = [int(v) for v in re.split(', |\s|,', file_numbers)]
        print("The following files were selected:\n{}".format("\n".join([mp3files[f]
            for f in file_numbers])))
        description = u.get_text("Short description of observation: ")
        notes = u.get_text("More detailed notes about observation: ")
        count_estimate = u.get_count_estimate()
        latitude = u.get_latitude()
        longitude = u.get_longitude()
        observation = db.Observation(collection=collection, description=description,
                notes=notes, count_estimate=count_estimate, 
                latitude=latitude, longitude=longitude)
        for f in [mp3files[n] for n in file_numbers]:
            set_recording(observation, f)
            mp3files.remove(f)

def set_recording(observation, mp3):
    #Going to assume an existing recording object for the given mp3 does not already exist
    # would be more defensive to check for it, but since the collection object it descends from
    # is already checked for prior existence, the recording *should* be safe as well
    start = datetime.datetime.fromtimestamp(os.path.getmtime(mp3))
    print("For the recording {}, start time found as {}".format(mp3, start))
    if not confirm("Is that the correct start time?"):
        start = u.get_start_time()
    info = mutagen.mp3.MP3(mp3).info
    recording = db.Recording(filename=mp3, observation=observation, start_time=start, 
            duration=info.length, bitrate=info.bitrate)
    return


def confirm(prompt):
    while True:
        value = raw_input(prompt + " (y/n)").lower()
        if value == "n":
            return False
        elif value == "y":
            return True
        else:
            print "invalid input, should be 'y' or 'n' but got {}".format(value)

def get_count_estimate():
    estimate = None
    while estimate is None:
        value = raw_input("Give estimate for number of pika in this observation" 
                " (put 'none' if you don't have an estimate): ")
        if value.lower() == "none":
            return None
        try:
            estimate = int(value)
        except ValueError as e:
            print "Value error - estimate should be an integer value. {}".format(e)
    return estimate

def get_latitude():
    lat = None
    while lat is None:
        value = raw_input("latitude: ")
        try:
            lat = float(value)
        except ValueError as e:
            print "Value error: latitude should be a decimal value between -90, 90. {}".format(e)
            continue
        if abs(lat) >= 90:
            print("Given value was outside appropriate range for latitude:"
            " should be between -90 and 90, got {}".format(lat))
            lat = None
    return lat

def get_longitude():
    longitude = None
    while longitude is None:
        value = raw_input("longitude: ")
        try:
            longitude = float(value)
        except ValueError as e:
            print "Value error: longitude should be a decimal value between -180, 180. {}".format(e)
            continue
        if abs(longitude) >= 180:
            print("Given value was outside appropriate range for longitude:"
            " should be between -180 and 180, got {}".format(longitude))
            longitude = None
    return longitude


def get_text(prompt):
    return raw_input(prompt)

def get_start_date():
    return get_date("Start date")
        
def get_end_date(start_date):
    if confirm("Use same date as start date ({})?".format(start_date.strftime("%m/%d/%Y"))):
        return None
    return get_date("End date")

def get_date(prompt):
    date = None
    while date is None:
        value = raw_input("{} (use mm/dd/yy format): ".format(prompt))
        try:
            date = datetime.datetime.strptime(value, "%m/%d/%y")
            print "{} of {}".format(prompt, date.strftime("%B %d, %Y"))
        except ValueError as e:
            print "Error getting date: {}".format(e)
    return date

def get_start_time():
    start = None
    while start is None:
        value = raw_input("Enter start time in MM/DD/YY HH:MM:SS format (24 hour time): ")
        try:
            start = datetime.datetime.strptime(value, "%m/%d/%y %H:%M:%S")
            print "Start time of {}".format(start.strftime("%B %d, %Y %I:%M:%S %p"))
        except ValueError as e:
            print "Error getting start time: {}".format(e)
    return start

def get_observer():
    observer = None
    while observer is None:
        observers = db.Observer.select()
        print "Observers:\n {}".format("\n".join(["{}: {}".format(o.id, o.name) for o in observers]))
        try:
            choice = raw_input("Choose observer id:")
            observer = db.Observer.get(int(choice))
        except SQLObjectNotFound:
            print "Your choice {} doesn't correspond to one of the observers.".format(choice)
            continue
        print "You chose observer: {}".format(observer.name)
    return observer
    
def get_collection_folder():
    mp3files = []
    while len(mp3files) == 0:
        folder = raw_input("Folder for collection?\n")
        if not os.path.exists(folder):
            print("Folder doesn't exist in attempt to create collection: {}".format(folder))
        else:
            collection = db.Collection.selectBy(folder=folder)
            if collection.count() > 0:
                print("Collection record for folder {} already exists (with id: {}, description: {})\n"
                        "Here's another chance!".format(input_folder, collection[0].id, collection[0].description))
            else: #new collection, get files that may be in collection
                mp3files = glob.glob(folder + "\\*.mp3")
                if len(mp3files) == 0:
                    print("No mp3 files were found in {}, there must be mp3 files to be processed for"
                            "the collection initialization to proceed.".format(folder))
                print "Collection folder contains {} mp3 files".format(len(mp3files))
    return folder