"""
Processing functions mainly in support of by pika.py
"""
import numpy as np
import scikits.audiolab
import time
import os
import subprocess

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

def segment_audio(audio, freq, segment_length=10):
    """iterator: iterates through audio segment_length seconds at a time
    yielding those segment_length seconds of the audio and the offset.
    Note the important difference between this and chunk_recording which
    performs a similar function:  chunk_recording iterates through a loaded
    audio array and yields it up in segments.  chunk_recording works through
    an existing (unloaded) audio file, writes chunks of that audio to a
    temporary file and yields that temporary file.
    """
    offset = 0
    step_size = segment_length*freq
    while offset < len(audio):
        next_offset = offset + step_size
        end = min(len(audio), next_offset)
        yield audio[offset:end], offset
        offset = next_offset

def chunk_recording(recording, segment_length=300):
    """iterator: iterates through recording segment_length seconds at a time
    using ffmpeg to create a temp file of the audio chunk
    yields the temp filename, offset
    Note the important difference between this and segment_audio which
    performs a similar function:  segment_audio iterates through a loaded
    audio array and yields it up in segments.  chunk_recording works through
    an existing (unloaded) audio file, writes chunks of that audio to a
    temporary file and yields that temporary file.
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

