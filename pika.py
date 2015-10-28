"""
Adapted from Shankar Shivappa's matlab HarmonicAnalysis.m code to Python
Further adaptations include pre-processing as discussed in "Acoustic Classification of Bird Species
 from syllabels: an Emprirical Study", Briggs, Fern, and Raich; Oregon State University

Usage Example:
    import pika as p
    
    (audio, freq, nBits) = p.load_audio("input.wav")
    p.audio_segments(audio, freq, 10, "output.wav") 

If you want the output aligned with the original audio (useful for debugging purposes) instead of the 
last line in the example above use:
    p.audio_segments(audio, freq, 10, "output.wav", True)


Example exploring results:
#This may be useful to further optimize predictor, particularly in noisy scenarios
    import pika as p
    
    (audio, freq, nBits) = p.load_audio("input.wav")
    parser = p.AudioParser(audio, freq)
    parser.pre_process()
    parser.harmonic_frequency()
    parser.plot_pika_from_harmonic() #Will show plots of the predicted results
"""
from pika_parser import AudioParser
import numpy as np
#import pandas as pd
#import scipy
#from scipy import signal
import scikits.audiolab
import pika_db_models as db
import matplotlib.pyplot as plt
import time
import os
import glob
import subprocess
from pylab import specgram

#infile = "May26-2014-BeaconRockSP4800.wav"
infile = "May26-2014-BeaconRockSP_shankars.wav"
#infile = "./input/July26-2014-LarchMountain/..."
infile2 = "LarchMountain_interspeciesCalls.wav"

def verify_calls(with_audio=True):
    """Go through unverified calls, display spectogram and get input verifying
    whether or not it is a pika call."""
    calls = db.Call.select(db.Call.q.verified==None)
    for call in calls[54:calls.count()]:
        (audio, freq, bits) = load_audio(call.filename)
        parser = AudioParser(audio, freq, debug=True)#, mpd=mpd_trial, ipd_filters=ipd_trial)
        parser.pre_process()
        count, offsets, output = parser.find_pika_from_harmonic()
        specgram(audio, 1028, freq, noverlap=600)
        plt.title("specgram of {}".format(call.filename))
        plt.show(block=False)
        if with_audio:
            play_audio(call.filename);
        raw_input("press enter to continue")
        plt.close()

def play_audio(audio):
    subprocess.call(["ffplay", "-nodisp", "-autoexit",
        "-loglevel", "0", "-af", "volume=20", audio])


def examine_results(recording):
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
            parser = AudioParser(chunk, freq, debug=True)#, mpd=mpd_trial, ipd_filters=ipd_trial)
            parser.pre_process()
            count, offsets, output = parser.find_pika_from_harmonic()
            offsets[:] = ["{0:.2f}".format(o) for o in offsets]
            print "examing results on {}".format(f)
            parser.spectrogram(label="found {} calls at {} base file offset {}".format(count, 
                offsets, audio_offsets[i] + j*10))


def load_and_parse(recording):
    """Takes Recording object which should have been already preprocessed and further processes
    the results of the preprocessing looking for pika calls.
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
    offset = 0
    step_size = segment_length*freq
    while offset < len(audio):
        next_offset = offset + step_size
        end = min(len(audio), next_offset)
        yield audio[offset:end]
        offset = next_offset

def audio_segments(audio, freq, offset, recording, segment_length=10):
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
        count, call_offsets, out = parser.find_pika_from_harmonic(.1)
        harmonic_out.append(out)
        total += count
        chunk_offset = i*freq
        for j, call in enumerate(out):
            c_offset = float(offset + chunk_offset + call_offsets[j])
            c_duration = float(len(call)*1.0/freq)
            c = db.Call(recording=recording, offset=c_offset, duration=c_duration, filename="temp")
            c.filename = path + "call{}.wav".format(c.id)
            scikits.audiolab.wavwrite(np.asarray(call), c.filename, freq)
    
    print "Finished processing {}\nTotal suspected calls: {}, total processing time (in seconds): {}". \
        format(recording.filename, total, time.time() - start_time)

