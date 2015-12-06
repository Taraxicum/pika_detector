"""
Adapted from Shankar Shivappa's matlab HarmonicAnalysis.m code to Python
Further adaptations include pre-processing as discussed in "Acoustic Classification
of Bird Species from syllabels: an Emprirical Study", Briggs, Fern, and Raich;
Oregon State University

Usage Example: 
    For initializing a collection you should have the mp3s in an accessible folder and be
    ready with whatever notes/gps coordinates/etc. you have for the collection, 
    then in ipython console:
    
        import pika as p
        p.init_collection_and_associated_recordings()

    The script will prompt you for the needed information.
    
"""
#TODO provide usage examples that work with the updated code

import pika_parser as pp
import ui_utility as u
import processing as p
import pika_db_models as db
import os

def init_collection_and_associated_recordings():
    db.init_db()

    folder = u.get_collection_folder()    
    observer = u.get_observer()
    start_date = u.get_start_date()
    end_date = u.get_end_date(start_date)
    description = u.get_text("Short description of collection: ")
    notes = u.get_text("More in depth notes about collection: ")
    collection = db.Collection(observer=observer, folder=folder,
            start_date=start_date, end_date=end_date,
            description=description, notes=notes)

    u.set_observations(collection)
    return collection

def preprocess_collection(collection):
    """
    Processes collection of recordings to get into manageable sized pieces and do 
        initial filter to remove sections of audio that don't have interesting sound in
        import pika call spectrum.
    :collection: collection of observations -> recordings to be processed
    modifies: creates files in recordingN subfolder of collection's folder where N is the
    id of the recording object in the database so there will be a subfolder for each 
    recording in collection.  Within the subfolder will be audio files named offset_nn.n.wav
    where nn.n is the offset in seconds of where the file was found in the original recording.
    """
    for observation in collection.observations:
        for recording in observation.recordings:
            output_path = recording.output_folder()
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            for chunk, offset in chunk_recording(recording):
                p.write_active_segments(chunk, output_path, offset)

def identify_calls(collection):
    for observation in collection.observations:
        for recording in observation.recordings:
            for f in recording.chunked_files:
                identify_and_write_calls(recording, f)

def identify_and_write_calls(recording, audio_file):
    parser = pp.PikaParser(recording, audio_file)
    parser.identify_and_write_calls()

def verify_calls(recording):
    for call in recording.get_unverified_calls():
        parser = pp.PikaParser(recording, call.filename)
        if not parser.verify_call(call):
            return
