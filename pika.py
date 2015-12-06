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
import find_peaks as peaks
import numpy as np
import scikits.audiolab
import pika_db_models as db
import matplotlib.pyplot as plt
import time
import os
import glob
import subprocess
from pylab import specgram
import datetime
import re
import mutagen.mp3

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
            rootpath = os.path.dirname(recording.filename)
            output_path = rootpath + "/recording{}/".format(recording.id)
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
        parser.verify_call(call)


###############Old code below, refactored code above################
#def examine_calls(calls=None, with_audio=True, debug=False, mpd=None):
#    """Go through all verified calls, display spectogram,
#    and optionally play audio or debugging information.
#    """
#    if calls is None:
#        calls = db.Call.select(db.Call.q.verified==True)
#    i = 1
#    for call in calls:
#        (audio, freq, bits) = load_audio(call.filename)
#        parser = AudioParser(audio, freq, debug=debug, mpd=mpd)
#        parser.pre_process(do_nr=False, do_filter=False, do_norm=False)
#        count, offsets, output = parser.find_pika_from_harmonic()
#        parser.spectrogram("{} {:.1f}".format(call.filename, call.offset+offsets[0]))
#        if with_audio:
#            play_audio(call.filename,vol_mult=40);
#        raw_input("Displayed call {} of {}.  Press enter to continue".format(i, 
#            calls.count()))
#        plt.close()
#        i += 1
#
#
#def verify_calls(with_audio=True, debug=False):
#    """Go through unverified calls, display spectogram and get input verifying
#    whether or not it is a pika call."""
#    calls = db.Call.select(db.Call.q.verified==None)
#    for call in calls:
#        (audio, freq, bits) = load_audio(call.filename)
#        parser = AudioParser(audio, freq, debug=debug)
#        parser.pre_process(do_nr=False, do_filter=False, do_norm=False)
#        count, offsets, output = parser.find_pika_from_harmonic()
#        parser.spectrogram(call.filename + " {}".format(call.offset))
#        if with_audio:
#            play_audio(call.filename);
#        if not get_verification(call):
#            plt.close()
#            return
#        plt.close()
#
#def get_verification(call):
#    """Return false if quit otherwise return True when valid response given"""
#    volume_mult = 20
#    while True:
#        print "Verify as pika call?"
#        r = raw_input("(Y)es/(N)o/(S)kip/(R)eplay/(L)ouder/(Q)uit (then press enter)")
#        r = r.lower()
#        if r == "q":
#            return False
#        if r == "y":
#            call.verified = True
#            return True
#        elif r == "n":
#            call.verified = False
#            return True
#        elif r == "s":
#            return True
#        elif r == "l":
#            volume_mult += 5
#            play_audio(call.filename, volume_mult)
#        elif r == "r":
#            play_audio(call.filename, volume_mult)
#            print "Call has been replayed"
#
#
#def play_audio(audio, vol_mult=20):
#    subprocess.call(["ffplay", "-nodisp", "-autoexit",
#        "-loglevel", "0", "-af", "volume={}".format(vol_mult), audio])
#
#
#def examine_results(recording, debug=True):
#    """For debugging the results on the audio from a recording"""
#    path = os.path.dirname(recording.filename) + "/recording{}/".format(recording.id)
#    wav_files = np.array(glob.glob(path + "offset_*.wav")) #should be of the form e.g. 'offset_3.0.wav'
#    
#    #ipd_trial = [[50, 75], [99, 135]]
#    #mpd_trial = 40
#    
#    #Sorting files based on numeric offset rather than lexicographically
#    #that way we get e.g. offset_37.0.wav before offset_201.0.wav
#    #where is the order would be reversed in the lexicographic sort
#    just_files = np.array([os.path.basename(f) for f in wav_files])
#    audio_offsets = np.array([float(f[7:f.find(".w")]) for f in just_files])
#    wav_files = wav_files[audio_offsets.argsort()]
#    audio_offsets = audio_offsets[audio_offsets.argsort()]
#
#    for i, f in enumerate(wav_files):
#        (snd, freq, bits) = load_audio(f)
#        for j, chunk in enumerate(segment_audio(snd, freq, 10)): 
#            parser = AudioParser(chunk, freq, debug=debug)#, mpd=mpd_trial, ipd_filters=ipd_trial)
#            parser.pre_process()
#            count, offsets, output = parser.find_pika_from_harmonic()
#            offsets[:] = ["{0:.2f}".format(o) for o in offsets]
#            print "examing results on {}".format(f)
#            parser.spectrogram(label="found {} calls at {} base file offset {}".format(count, 
#                offsets, audio_offsets[i] + j*10))
#            raw_input("Press enter to continue")
#            plt.close()
#
#
#def load_and_parse(recording):
#    """Takes Recording object which should have been already preprocessed and 
#    further processes the results of the preprocessing looking for pika calls.
#    :recording: Recording db object
#    """
#    path = os.path.dirname(recording.filename) + "/recording{}/".format(recording.id)
#    wav_files = glob.glob(path + "offset_*.wav") #should be of the form e.g. 'offset_3.0.wav'
#    for f in wav_files:
#        just_file = os.path.basename(f)
#        offset = float(just_file[7:just_file.find(".w")])
#        (snd, freq, bits) = load_audio(f)
#        audio_segments(snd, freq, offset, recording, 10)
#
#def load_audio(filepath):
#    (snd, sampFreq, nBits) = scikits.audiolab.wavread(filepath)
#    if snd.ndim == 2: #get left channel if a stereo file not needed for mono
#        snd = [v[0] for v in snd] 
#    #print "sample frequency of {}: {}".format(filepath, sampFreq)
#    return (snd, sampFreq, nBits)
#
#def segment_audio(audio, freq, segment_length=10):
#    """iterator: iterates through audio segment_length seconds at a time
#    yielding those segment_length seconds of the audio.
#    """
#    offset = 0
#    step_size = segment_length*freq
#    while offset < len(audio):
#        next_offset = offset + step_size
#        end = min(len(audio), next_offset)
#        yield audio[offset:end]
#        offset = next_offset
#
#def audio_segments(audio, freq, file_offset, recording, segment_length=10):
#    """Segments audio into segment_length (in seconds) chunks and runs the 
#    algorithm on each chunk.  Creates new Call db object and a file for each
#    identified call.  The wav file containing the call will have its filename
#    in the database and should be in the path associated with recording
#    :audio: the audio file to be processed
#    :freq: frequency of the audio file
#    :offset: offset of audio being processed from beginning of original file
#    :recording: db object of recording containing audio being processed
#    :segment_length: length in seconds of the segments to run the algorithm on
#
#    :recording: db object of the recording being processed - should have been
#    preprocessed before this point.
#    """
#    harmonic_out = []
#    total = 0
#    start_time = time.time()
#    path = os.path.dirname(recording.filename) + "/recording{}/".format(recording.id)
#
#    for i, chunk in enumerate(segment_audio(audio, freq, segment_length)): 
#        parser = AudioParser(chunk, freq)
#        parser.pre_process()
#        count, call_offsets, out = parser.find_pika_from_harmonic(.2)
#        harmonic_out.append(out)
#        total += count
#        chunk_offset = i*segment_length
#        for j, call in enumerate(out):
#            c_offset = float(file_offset + chunk_offset + call_offsets[j])
#            c_duration = float(len(call)*1.0/freq)
#            c = db.Call(recording=recording, offset=c_offset, duration=c_duration, filename="temp")
#            c.filename = path + "call{}.wav".format(c.id)
#            scikits.audiolab.wavwrite(np.asarray(call), c.filename, freq)
#    
#    print "Finished processing {}\nTotal suspected calls: {}, total processing time (in seconds): {}". \
#        format(recording.filename, total, time.time() - start_time)
#
