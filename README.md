# pika_detector 
Code to detect pika (*Ochonta princeps*) calls in audio file.

There are two versions of code: the original matlab code created by Shankar and the Python version I have been adapting from it.

#### Basic usage example
The following will process the input.wav in 10 second chunks, then combine the results into output.wav.  There may be detection issues for a call if it happens to fall over the boundary between segments.
    
    import pika as p
    
    (audio, freq, nBits) = p.load_audio("input.wav")
    p.audio_segments(audio, freq, 10, "output.wav") 

If you want the output aligned with the original audio (useful for debugging purposes) instead of the 
last line in the example above use:

    p.audio_segments(audio, freq, 10, "output.wav", True)


#### Example exploring results visually
The following will show plots of predicted pika calls found in input.wav

    import pika as p
    
    (audio, freq, nBits) = p.load_audio("input.wav")
    parser = p.AudioParser(audio, freq)
    parser.pre_process()
    parser.harmonic_frequency()
    parser.plot_pika_from_harmonic()
