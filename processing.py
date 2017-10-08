"""
Processing functions mainly in support of pika.py
"""
import numpy as np
#import scikits.audiolab
import wave
import time
import os
import glob
import subprocess
import mutagen.mp3


def segment_mp3(filename, segment_length=300, output_frequency=44100):
    """
    Parses mp3 into .wav files and yields audio and offset of the segments to be iterated over 
    :filename: path of mp3 to returns segments of
    :segment_length: in seconds the length of the segments (last segment will
    probably be less than segment_length
    """
    offset = 0
    step_size = int(segment_length) #in seconds
    outfile = "temp.wav"
    info = mutagen.mp3.MP3(filename).info
    if output_frequency is not None and info.sample_rate != output_frequency:
        resample = True
    else:
        resample = False

    while offset < info.length:
        next_offset = offset + step_size
        end = min(int(info.length), next_offset)
        length = end - offset
        try:
            if resample:
                subprocess.check_output(["ffmpeg", "-loglevel", "0", "-channel_layout", "stereo",
                    "-i", filename, "-ac", "1",
                    "-ar", str(output_frequency),
                    "-ss", str(offset), "-t", str(length), outfile])
            else:
                subprocess.check_output(["ffmpeg", "-loglevel", "0", "-channel_layout", "stereo",
                    "-i", filename, "-ss", str(offset), "-t", str(length), outfile])
            
            yield outfile, offset
        finally:
            os.remove(outfile)
        offset = next_offset


def write_active_segments(filename, path, offset, frequency=44100):
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
    :frequency to write segments at.  Previosly defaulted to same frequency as input file, but 
    since frequency effects the frequencies in the fft it might be better to default to a fixed
    rate.  I chose 44100 here because that is a common sample frequency and I suspect that we
    will mostly be using recordings at that sample frequency or higher.
    """
    intervals, fft = find_active_segments(filename)
    if len(intervals) == 0:
        print("No active segments found in {}".format(filename))
        return
    for i, interval in enumerate(intervals):
        print("i: {}, interval: {}".format(i, interval))
        outfile = path + "offset_{}.wav".format(offset + interval[0])
        pkl_outfile = path + "offset_{}.pkl".format(offset + interval[0])
        try:
            fft.serialize_interval(interval[0], interval[1], pkl_outfile)
            with open(os.devnull, 'w') as f:
                subprocess.check_call(["ffmpeg", "-loglevel", "0", '-channel_layout', 'stereo', "-i", filename,
                    "-ar", str(frequency), "-ss", str(interval[0]), "-t", str(interval[1]-interval[0]), outfile])
        except Exception as inst:
            print("There was an exception writing temp file {} in write_active_segments:".format(outfile))
            print(type(inst))
            print(inst.args)
            print(inst)

def load_wav(filename):
    #(snd, freq, nbits) = scikits.audiolab.wavread(filename)
    wave_read = wave.open(filename, 'rb')
    freq = wave_read.getframerate()
    import pdb; pdb.set_trace()
    if snd.ndim == 2:
        snd = [v[0] for v in snd]
    return snd, freq
    
def find_active_segments(filename, verbose=0, fft=None, audio=None, freq=None):
    """Returns array of intervals of audio that have magnitude above a threshold
    background noise.  If a segment of the audio is more than .5 seconds from
    a sound over the threshold it will not be included in the output
    :filename: audio file to be used - should be wav file
    :returns: array of intervals e.g. [[5, 157], [990, 1105]] of time in seconds
    (floor of start value, ceiling of end value) corresponding to louder sections
    of the audio, as well as the windowed, filtered fft used to find the 
    active segments (so it can be written out for use when finding the calls).
    """
    if verbose > 0:
        start_time = time.time()
    
    if audio is None or freq is None:
        snd, freq = load_wav(filename)
    else:
        snd = audio

    
    fft_size = 4096
    step_size = fft_size/2
    window = [fft_size/32 + 150]
    window.append(window[0] + 275)
    
    if fft is None:
        fft = pfft.ProcessedFFT(snd, window, fft_size, step_size, freq)
        fft.process_fft()

    threshold_distance = 10 #.5*freq/step_size #seconds worth of steps
    factor = step_size*1.0/freq

    avg_magnitude = np.mean(fft.fft)
    max_to_mean = np.divide(np.max(fft.fft, axis=1), np.mean(fft.fft, axis=1))
    threshold = 4.5
    above_threshold = [1 if x > threshold else 0 for x in max_to_mean]
    above_threshold = np.convolve(above_threshold, np.ones(threshold_distance), 'same')
    
    intervals = get_intervals(above_threshold, factor)
    if verbose > 0: 
        print("Total length kept {}".format(total_segment_length(intervals)))
        print("time taken {}".format(time.time() - start_time))
    return intervals, fft

def total_segment_length(intervals):
    length = 0
    for interval in intervals:
        length += interval[1] - interval[0]
    return length

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
                interval_start = int(np.floor(i*factor))
        elif val == 0:
            intervals.append([interval_start, int(np.ceil(i*factor))])
            interval_start = None
    if interval_start is not None:
        intervals.append([interval_start, int(np.ceil(factor*len(segments)))])
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

def segment_audio(audio, freq, segment_length=10):
    """returns iterator: iterates through audio segment_length seconds at a time
    yielding those segment_length seconds of the audio and the offset.
    Note the important difference between this and chunk_recording which
    performs a similar function:  segment_audio iterates through a loaded
    audio array and yields it up in segments.  chunk_recording works through
    an existing (unloaded) audio file, writes chunks of that audio to a
    temporary file and yields that temporary file and the offset (in seconds)
    of it within the original file.
    """
    offset = 0
    step_size = int(segment_length*freq) #in samples
    total_frames = len(audio)
    while offset < total_frames: #len(audio):
        next_offset = offset + step_size
        end = min(total_frames, next_offset)
        #yield audio.readframes(step_size), offset*1.0/freq
        yield audio[offset:end], offset*1.0/freq
        offset = next_offset

def get_split_wavs(recording):
    #path = recording.output_folder()
    path = "collections/test6/HermanCreek/split/" #FIXME! Should not be hard coded path
    wav_files = np.array(glob.glob(path + "output*.wav"))
    base_files = [os.path.basename(f) for f in wav_files]
    audio_offsets = np.array([int(f[6:f.find(".w")]) for f in base_files])
    wav_files = wav_files[audio_offsets.argsort()]
    audio_offsets.sort()
    for i in range(len(wav_files)):
        yield wav_files[i], 120*audio_offsets[i]
    
def chunk_recording(recording, segment_length=300, output_frequency=44100):
    """iterator: iterates through recording segment_length seconds at a time
    using ffmpeg to create a temp file of the audio chunk
    yields the temp filename, offset
    Note the important difference between this and segment_audio which
    performs a similar function:  segment_audio iterates through a loaded
    audio array and yields it up in segments.  chunk_recording works through
    an existing (unloaded) audio file, writes chunks of that audio to a
    temporary file and yields that temporary file.
    
    :recording the pika_db object corresponding to the audio file to be chunked.
    :segment_length (in seconds) length of chunks
    :output_frequency set to go with original frequency of the audio file, 
    otherwise saves temp file at given frequency
    """
    offset = 0
    step_size = int(segment_length) #in seconds
    outfile = "temp/temp.wav"
    info = mutagen.mp3.MP3(recording.filename).info
    if output_frequency is not None and info.sample_rate != output_frequency:
        resample = True
    else:
        resample = False

    while offset < recording.duration:
        next_offset = offset + step_size
        end = min(int(recording.duration), next_offset)
        length = end - offset
        try:
            if resample:
                subprocess.check_output(["ffmpeg", "-loglevel", "0", "-channel_layout", "stereo",
                    "-i", recording.filename, "-ar", str(output_frequency),
                    "-ss", str(offset), "-t", str(length), outfile])
            else:
                subprocess.check_output(["ffmpeg", "-loglevel", "0", "-channel_layout", "stereo",
                    "-i", recording.filename, "-ss", str(offset), "-t", str(length), outfile])
            
            yield outfile, offset
        finally:
            os.remove(outfile)
        offset = next_offset

