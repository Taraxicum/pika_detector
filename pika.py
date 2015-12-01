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

from pika_parser import AudioParser
import utility as u
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

#infile = "May26-2014-BeaconRockSP4800.wav"
infile = "May26-2014-BeaconRockSP_shankars.wav"
#infile = "./input/July26-2014-LarchMountain/..."
infile2 = "LarchMountain_interspeciesCalls.wav"

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

    set_observations(collection)
    return collection

def preprocess_collection(collection):
    for observation in collection.observations:
        for recording in observation.recordings:
            rootpath = os.path.dirname(recording.filename)
            output_path = rootpath + "\\recording{}\\".format(recording.id)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            for chunk, offset in chunk_recording(recording):
                write_active_segments(chunk, output_path, offset)

def write_active_segments(filename, path, offset):
    """
    Processes audio file to find parts of the file that are active - i.e. the parts that aren't 
    just background noise.  Outputs files to path folder with file named offset_{}.wav where the
    {} is the offset position calculated by the offset parameter passed in + the offset of the
    segment relative to the audio in filename.

    :filename: name of the audio file to be processed.
    :path: path of folder to write output to
    :offset: base offset for filename; in general it is expected that the original file will be
        broken into smaller chunks initially and this function will be applied to those chunks,
        so the offset would be the offset of the chunk in the recording
    """
    intervals = find_active_segments(filename)
    if len(intervals) == 0:
        print "No active segments found in {}".format(filename)
        return
    for i, interval in enumerate(intervals):
        outfile = path + "offset_{}.wav".format(offset + interval[0])
        try:
            with open(os.devnull, 'w') as f:
                subprocess.check_call(["ffmpeg", "-loglevel", "0", '-channel_layout', 'stereo', "-i", filename,
                    "-ss", str(interval[0]), "-t", str(interval[1]-interval[0]), outfile])
        except Exception as inst:
            print "There was an exception writing temp file {} in write_active_segments:".format(outfile)
            print type(inst)
            print inst.args
            print inst

def find_active_segments(filename, verbose=0):
    """Returns array of intervals of audio that have magnitude above a threshold
    background noise.  If a segment of the audio is more than .5 seconds from
    a sound over the threshold it will not be included in the output
    :filename: audio file to be used - should be wav file
    :returns: array of intervals e.g. [[5, 157], [990, 1105]] of time in seconds
    (floor of start value, ceiling of end value) corresponding to louder sections
    of the audio
    """
    if verbose > 0:
        start_time = time.time()
    (snd, freq, nbits) = scikits.audiolab.wavread(filename)
    if snd.ndim == 2:
        snd = [v[0] for v in snd]
    fft_size = 4096
    step_size = fft_size/2
    first_dim = len(snd)/(step_size) - 1
    second_dim = 275
    fft = np.zeros((first_dim, second_dim))
    threshold_distance = .5*freq/step_size #seconds worth of steps
    factor = step_size*1.0/freq

    for i in range(0, len(snd) - fft_size, step_size):
        f = np.absolute(np.fft.fft(snd[i:i+fft_size]))
        fft[i/step_size] = f[fft_size/32:fft_size/2][150:425]
    max_val = np.amax(fft)
    fft /= max_val
    
    avg_magnitude = np.mean(fft)
    max_to_mean = np.divide(np.max(fft, axis=1), np.mean(fft, axis=1))
    threshold = 4.5
    above_threshold = [1 if x > threshold else 0 for x in max_to_mean]
    above_threshold = np.convolve(above_threshold, np.ones(threshold_distance), 'same')
    
    intervals = get_intervals(above_threshold, factor)
    if verbose > 0: 
        print "Total length kept {}".format(total_segment_length(intervals))
        print "time taken {}".format(time.time() - start_time)
    return intervals    

def get_intervals(segments, factor):
    """Looks for positive intervals in list of segments and returns list of 
    the intervals scaled to where they are located in seconds.
    :segments: list of values under consideration.  Generally here they will be
    some value derived from the fft of audio data.
    :factor: value to convert between position in the segments list and
    position in the audio file in seconds.
    :returns list of intervals (with endpoints rounded to integer values)
    corresponding to the times in the audio file where the values in segments 
    are non-zero.  Before being returned calls reduce_intervals to combine
    segments with overlap that result from the rounding of the endpoints.
    """
    intervals = []
    interval_start = None
    for i, val in enumerate(segments):
        if interval_start is None:
            if val > 0:
                interval_start = np.floor(i*factor)
        elif val == 0:
            intervals.append([interval_start, np.ceil(i*factor)])
            interval_start = None
    if interval_start is not None:
        intervals.append([interval_start, np.ceil(factor*len(segments))])
    #print intervals
    return reduce_intervals(intervals)
    
def reduce_intervals(intervals):
    """
    :intervals: a list of intervals that may have overlapping endpoints but
    where if [a, b] occurs before [c, d] in the list then a <= c.
    :returns list of intervals with overlaps combined so that now if
    [a, b] occurs before [c, d] in the returned list, then b < c.
    """
    reduced_intervals = []
    interval_start = None
    for s in intervals:
        if interval_start is not None:
            if s[0] <= interval_end:
                #print "Overlap old {}, new {}".format(current_segment_end, s)
                interval_end = s[1]
            else:
                #print "NO overlap old {}, new {}".format(current_segment_end, s)
                reduced_intervals.append([interval_start, interval_end])
                interval_start = s[0]
                interval_end = s[1]
        else:
            interval_start = s[0]
            interval_end = s[1]
    if interval_start is not None:
        reduced_intervals.append([interval_start, intervals[-1][1]])
    return reduced_intervals


def chunk_recording(recording, segment_length=300):
    """iterator: iterates through recording segment_length seconds at a time
    using ffmpeg to create a temp file of the audio chunk
    yields the temp filename, offset
    """
    offset = 0
    step_size = int(segment_length*recording.bitrate)
    outfile = "temp/temp.wav"
    
    while offset < recording.duration:
        next_offset = offset + step_size
        end = min(int(recording.duration), next_offset)
        length = end - offset
        subprocess.check_output(["ffmpeg", "-loglevel", "0", "-channel_layout", "stereo",
            "-i", recording.filename, "-ss", str(offset), "-t", str(length), outfile])
        
        yield outfile, offset
        offset = next_offset

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

###############Old code below, refactored code above################
def examine_calls(calls=None, with_audio=True, debug=False, mpd=None):
    """Go through all verified calls, display spectogram,
    and optionally play audio or debugging information.
    """
    if calls is None:
        calls = db.Call.select(db.Call.q.verified==True)
    i = 1
    for call in calls:
        (audio, freq, bits) = load_audio(call.filename)
        parser = AudioParser(audio, freq, debug=debug, mpd=mpd)
        parser.pre_process(do_nr=False, do_filter=False, do_norm=False)
        count, offsets, output = parser.find_pika_from_harmonic()
        parser.spectrogram("{} {:.1f}".format(call.filename, call.offset+offsets[0]))
        if with_audio:
            play_audio(call.filename,vol_mult=40);
        raw_input("Displayed call {} of {}.  Press enter to continue".format(i, 
            calls.count()))
        plt.close()
        i += 1


def verify_calls(with_audio=True, debug=False):
    """Go through unverified calls, display spectogram and get input verifying
    whether or not it is a pika call."""
    calls = db.Call.select(db.Call.q.verified==None)
    for call in calls:
        (audio, freq, bits) = load_audio(call.filename)
        parser = AudioParser(audio, freq, debug=debug)
        parser.pre_process(do_nr=False, do_filter=False, do_norm=False)
        count, offsets, output = parser.find_pika_from_harmonic()
        parser.spectrogram(call.filename + " {}".format(call.offset))
        if with_audio:
            play_audio(call.filename);
        if not get_verification(call):
            plt.close()
            return
        plt.close()

def get_verification(call):
    """Return false if quit otherwise return True when valid response given"""
    volume_mult = 20
    while True:
        print "Verify as pika call?"
        r = raw_input("(Y)es/(N)o/(S)kip/(R)eplay/(L)ouder/(Q)uit (then press enter)")
        r = r.lower()
        if r == "q":
            return False
        if r == "y":
            call.verified = True
            return True
        elif r == "n":
            call.verified = False
            return True
        elif r == "s":
            return True
        elif r == "l":
            volume_mult += 5
            play_audio(call.filename, volume_mult)
        elif r == "r":
            play_audio(call.filename, volume_mult)
            print "Call has been replayed"


def play_audio(audio, vol_mult=20):
    subprocess.call(["ffplay", "-nodisp", "-autoexit",
        "-loglevel", "0", "-af", "volume={}".format(vol_mult), audio])


def examine_results(recording, debug=True):
    """For debugging the results on the audio from a recording"""
    path = os.path.dirname(recording.filename) + "\\recording{}\\".format(recording.id)
    wav_files = np.array(glob.glob(path + "offset_*.wav")) #should be of the form e.g. 'offset_3.0.wav'
    
    #ipd_trial = [[50, 75], [99, 135]]
    #mpd_trial = 40
    
    #Sorting files based on numeric offset rather than lexicographically
    #that way we get e.g. offset_37.0.wav before offset_201.0.wav
    #where is the order would be reversed in the lexicographic sort
    just_files = np.array([os.path.basename(f) for f in wav_files])
    audio_offsets = np.array([float(f[7:f.find(".w")]) for f in just_files])
    wav_files = wav_files[audio_offsets.argsort()]
    audio_offsets = audio_offsets[audio_offsets.argsort()]

    for i, f in enumerate(wav_files):
        (snd, freq, bits) = load_audio(f)
        for j, chunk in enumerate(segment_audio(snd, freq, 10)): 
            parser = AudioParser(chunk, freq, debug=debug)#, mpd=mpd_trial, ipd_filters=ipd_trial)
            parser.pre_process()
            count, offsets, output = parser.find_pika_from_harmonic()
            offsets[:] = ["{0:.2f}".format(o) for o in offsets]
            print "examing results on {}".format(f)
            parser.spectrogram(label="found {} calls at {} base file offset {}".format(count, 
                offsets, audio_offsets[i] + j*10))
            raw_input("Press enter to continue")
            plt.close()


def load_and_parse(recording):
    """Takes Recording object which should have been already preprocessed and 
    further processes the results of the preprocessing looking for pika calls.
    :recording: Recording db object
    """
    path = os.path.dirname(recording.filename) + "\\recording{}\\".format(recording.id)
    wav_files = glob.glob(path + "offset_*.wav") #should be of the form e.g. 'offset_3.0.wav'
    for f in wav_files:
        just_file = os.path.basename(f)
        offset = float(just_file[7:just_file.find(".w")])
        (snd, freq, bits) = load_audio(f)
        audio_segments(snd, freq, offset, recording, 10)

def load_audio(filepath):
    (snd, sampFreq, nBits) = scikits.audiolab.wavread(filepath)
    if snd.ndim == 2: #get left channel if a stereo file not needed for mono
        snd = [v[0] for v in snd] 
    #print "sample frequency of {}: {}".format(filepath, sampFreq)
    return (snd, sampFreq, nBits)

def segment_audio(audio, freq, segment_length=10):
    """iterator: iterates through audio segment_length seconds at a time
    yielding those segment_length seconds of the audio.
    """
    offset = 0
    step_size = segment_length*freq
    while offset < len(audio):
        next_offset = offset + step_size
        end = min(len(audio), next_offset)
        yield audio[offset:end]
        offset = next_offset

def audio_segments(audio, freq, file_offset, recording, segment_length=10):
    """Segments audio into segment_length (in seconds) chunks and runs the 
    algorithm on each chunk.  Creates new Call db object and a file for each
    identified call.  The wav file containing the call will have its filename
    in the database and should be in the path associated with recording
    :audio: the audio file to be processed
    :freq: frequency of the audio file
    :offset: offset of audio being processed from beginning of original file
    :recording: db object of recording containing audio being processed
    :segment_length: length in seconds of the segments to run the algorithm on

    :recording: db object of the recording being processed - should have been
    preprocessed before this point.
    """
    harmonic_out = []
    total = 0
    start_time = time.time()
    path = os.path.dirname(recording.filename) + "\\recording{}\\".format(recording.id)

    for i, chunk in enumerate(segment_audio(audio, freq, segment_length)): 
        parser = AudioParser(chunk, freq)
        parser.pre_process()
        count, call_offsets, out = parser.find_pika_from_harmonic(.2)
        harmonic_out.append(out)
        total += count
        chunk_offset = i*segment_length
        for j, call in enumerate(out):
            c_offset = float(file_offset + chunk_offset + call_offsets[j])
            c_duration = float(len(call)*1.0/freq)
            c = db.Call(recording=recording, offset=c_offset, duration=c_duration, filename="temp")
            c.filename = path + "call{}.wav".format(c.id)
            scikits.audiolab.wavwrite(np.asarray(call), c.filename, freq)
    
    print "Finished processing {}\nTotal suspected calls: {}, total processing time (in seconds): {}". \
        format(recording.filename, total, time.time() - start_time)

