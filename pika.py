"""
Adapted from Shankar Shivappa's matlab HarmonicAnalysis.m code to Python
Further adaptations include pre-processing as discussed in "Acoustic Classification
of Bird Species from syllabels: an Emprirical Study", Briggs, Fern, and Raich;
Oregon State University

Usage Example: 
    For initializing a collection you should have the mp3s in an accessible folder and be
    ready with whatever notes/gps coordinates/etc. you have for the collection, 
    then in ipython console the following code will take you through creation of the collection
    all the way to verifying identified calls:
    
        import pika as p
        collection = p.init_collection_and_associated_recordings()
        p.preprocess_collection(collection)
        p.identify_and_write_calls(collection)
        p.verify_calls(collection)

    The script will prompt you for the needed information.
    
"""
import pika_parser as pp
import utility as util
import processing as processing
import pika_db_models as db
import os

def init_collection_and_associated_recordings():
    db.init_db()

    folder = util.get_collection_folder()
    observer = util.get_observer()
    start_date = util.get_start_date()
    end_date = util.get_end_date(start_date)
    description = util.get_text("Short description of collection: ")
    notes = util.get_text("More in depth notes about collection: ")
    collection = db.Collection(observer=observer, folder=folder,
            start_date=start_date, end_date=end_date,
            description=description, notes=notes)

    util.set_observations(collection)
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
            
            for chunk, offset in processing.chunk_recording(recording):
                processing.write_active_segments(chunk, output_path, offset)

def identify_and_write_calls(collection):
    for observation in collection.observations:
        for recording in observation.recordings:
            for f in recording.chunked_files():
                parser = pp.PikaParser(recording, f, db)
                parser.identify_and_write_calls()

def verify_calls(collection):
    for observation in collection.observations:
        for recording in observation.recordings:
            for call in recording.get_unverified_calls():
                parser = pp.PikaParser(recording, call.filename, db)
                if not parser.verify_call(call):
                    return
