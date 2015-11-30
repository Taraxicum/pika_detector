"""
Many audio files in the pika project may be long with many segments of silence
(since audio recorder may be left in the field for hours/days at a time).

In order to make the audio more practical to deal with, tools are provided here to process the audio into smaller chunks and remove long sections where there is no audio activity (beyond background noise).

#TODO usage example:

"""

import mutagen.mp3
import scikits.audiolab
import subprocess
import os
import glob
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import pika_db_models as db

db.init_db()
def example_work_flow():
    #input_folder = ".\collections\initial_trials"
    input_folder = ".\collections\July26-2014-LarchMountain"
    start_date = datetime.date(2014, 7, 26) #"July 26, 2014"
    description = "Larch Mountain"
    observer = db.Observer.get(2) #Shankar

    collection = create_collection(input_folder, observer, start_date, description)
    process_files_in_collection(collection)
    
def create_collection(foldername, observer, start_date, description):
    if not os.path.exists(foldername):
        raise Exception("Folder doesn't exist in attempt to create collection: {}".format(foldername))
        return
    
    collection = db.Collection.selectBy(folder=foldername)
    if collection.count() > 0:
        if collection.count() > 1:
            print "More than one collection record for folder {}.  Please look into!" \
                    "\nUsing first collection object found, id {}, description {}".format(input_folder, 
                            collection[0].id, collection[0].description)
        collection = collection[0]
    else:
        collection = db.Collection(observer=observer, folder=foldername, 
                start_date=start_date, description="Doing more testing")
    return collection


def process_files_in_collection(collection):
    #For now assume each collection consists of a single observation object.  TODO implement interface
    #  so we don't have to go with that assumption, also so we can put in values for the observation fields
    observation = db.Observation(collection=collection, notes="Test Run - need to update fields if want to use")
    
    mp3files = glob.glob(collection.folder + "\\*.mp3")
    
    for f in mp3files:
        recording = db.Recording.selectBy(filename=f)
        if recording.count() > 0:
            if recording.count() > 1:
                print "More than one recording record for file {}.  Please look into!" \
                        "\nUsing first recording object found, id {}, observation id {}".format(
                                f, recording[0].id, recording[0].collection.id)
            recording = recording[0]
        else:
            start = datetime.datetime.fromtimestamp(os.path.getmtime(f))
            info = mutagen.mp3.MP3(f).info
            recording = db.Recording(filename=f, observation=observation, start_time=start, 
                    duration=info.length, bitrate=info.bitrate)
    
        process_mp3_in_chunks(recording)



def process_mp3_in_chunks(recording, chunklength=300):
    """Splits mp3 referred to by recording into separate chunks and calls processing function
    after each chunk is temporarily output to temp_output.
    chunklength defaults to 300 seconds (5 minutes) per chunk
    """
    if recording.filename is None:
        raise Exception("No filename given for process_mp3_in_chunks")
    
    temp_output='tmp{}.wav'     
    #make subdirectory for recording to store chunks in as well as preprocessed wav files
    rootpath = os.path.dirname(recording.filename)
    output_path = rootpath + "\\recording{}\\".format(recording.id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for count in range(0, int(recording.duration), chunklength):
        outfilename = output_path + temp_output.format(count)
        if count + chunklength < recording.duration:
            length = chunklength
        else:
            length = recording.duration - count
        try:
            subprocess.check_output(["ffmpeg", "-loglevel", "0", "-channel_layout", "stereo",
                "-i", recording.filename, "-ss", str(count), "-t", str(length), outfilename])
        except Exception as inst:
            print "There was an exception:"
            print type(inst)
            print inst.args
            print inst
        write_active_segments(outfilename, output_path, count)

        os.remove(outfilename)
        for f in glob.glob("output\\*"):
            os.remove(f)

def write_active_segments(filename, path, offset):
    """
    Processes audio file to find parts of the file that are active - i.e. the parts that aren't 
    just background noise.  Outputs to a file of the same name in the joined_output folder
    :filename: name of the audio file to be processed.
    """
    
    print "Processing {} at offset {}".format(filename, offset)

    intervals = find_active_segments(filename)
    output_file = "joined_output\\{}".format(filename)
    if len(intervals) == 0:
        print "No active segments found in {}".format(filename)
        return
    for i, interval in enumerate(intervals):
        temp = path + "offset_{}.wav".format(offset + interval[0])
        try:
            with open(os.devnull, 'w') as f:
                subprocess.check_call(["ffmpeg", "-loglevel", "0", '-channel_layout', 'stereo', "-i", filename,
                    "-ss", str(interval[0]), "-t", str(interval[1]-interval[0]), temp])
        except Exception as inst:
            print "There was an exception writing temp file {} in write_active_segments:".format(temp)
            print type(inst)
            print inst.args
            print inst


def find_active_segments(filename, show_plot=False):
    """Returns array of intervals of audio that have magnitude above a threshold
    background noise.  If a segment of the audio is more than .5 seconds from
    a sound over the threshold it will not be included in the output
    :filename: audio file to be used - should be wav file
    :returns: array of intervals e.g. [[5, 157], [990, 1105]] of time in seconds
    (floor of start value, ceiling of end value) corresponding to louder sections
    of the audio
    """
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
    
    #Subtract off average frame for noise reduction:
    #avg_fft = np.sum(fft, axis=0)/len(fft)
    #for i, frame in enumerate(fft):
        #fft[i] = [max(frame[j] - avg_fft[j], 0) for j, v in enumerate(frame)] #noise-reduction
    
    avg_magnitude = np.mean(fft)
    #print "Average magnitude {}".format(avg_magnitude)
    #threshold = .2 #avg_magnitude + .12 #May want to make this threshold adjustable
    #since whether the segment of audio has a lot happening or is mostly quiet
    #could effect the average based thresholding
    #above_threshold = [1 if x > threshold else 0 for x in np.max(fft, axis=1)]
    #above_threshold = np.convolve(above_threshold, np.ones(threshold_distance), 'same')
    #Compare mean magnitude to max magnitude with theory that noisy parts will tend to
    #have low max compared to high mean while good signal will tend to have higher
    #max to mean ratio
    max_to_mean = np.divide(np.max(fft, axis=1), np.mean(fft, axis=1))
    threshold = 4.5
    above_threshold = [1 if x > threshold else 0 for x in max_to_mean]
    above_threshold = np.convolve(above_threshold, np.ones(threshold_distance), 'same')
    
    #print "threshold length {}".format(len(above_threshold))
    #print "remove_silence: file {}, length {}, freq {}, nbits {}".format(filename,
            #len(snd), freq, nbits)
    intervals = get_intervals(above_threshold, factor)
    #print intervals
    print "Total length kept {}".format(total_segment_length(intervals))
    print "time taken {}".format(time.time() - start_time)
    
    if show_plot:
        plt.subplot(4, 1, 1)
        plt.imshow(np.asarray([f for f in fft]).T, origin='lower')
        plt.xticks(plt.xticks()[0], [str(int(t*factor)) for t in plt.xticks()[0]])
        plt.xlim(0, len(fft))
        plt.xlabel("Processed FFT")
        plt.subplot(4, 1, 2)
        plt.xlabel("mean of fft frame")
        plt.plot(np.mean(fft, axis=1))
        plt.xticks(plt.xticks()[0], [str(int(t*factor)) for t in plt.xticks()[0]])
        plt.xlim(0, len(fft))
        plt.subplot(4, 1, 3)
        plt.xlabel("max of fft frame")
        plt.plot(np.max(fft, axis=1))
        plt.xticks(plt.xticks()[0], [str(int(t*factor)) for t in plt.xticks()[0]])
        plt.xlim(0, len(fft))
        plt.subplot(4, 1, 4)
        plt.xlabel("filtered max to mean ration of fft")
        plt.plot([x if x > 4 else 0 for x in max_to_mean]) #above_threshold)
        plt.xticks(plt.xticks()[0], [str(int(t*factor)) for t in plt.xticks()[0]])
        plt.xlim(0, len(fft))
        plt.tight_layout()
        plt.show()
    #return above_threshold    
    return intervals    

def total_segment_length(segments):
    """Takes a list of segments and returns the total length of those segments.
    example:
        input: [[5, 19], [20, 31], [50, 56]]
        output: 31
    """
    return np.sum([x[1] - x[0] for x in segments])

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
