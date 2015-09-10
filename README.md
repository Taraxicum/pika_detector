# pika_detector 
Code to detect pika (*Ochonta princeps*) calls in audio file.

There are two versions of code: the original matlab code created by Shankar and the Python version I have been adapting from it.

Usage Examples for the Python code:
    
    import pika as p
    
    (audio, freq, nBits) = p.load_audio(p.infile)   #here p.infile could be any desired audio file
    #left = [v[0] for v in audio] #left channel if a stereo file not needed for mono
    p.audio_segments(audio, freq, 10, "trial.wav") 

If you want the output aligned with the original audio (useful for debugging purposes) instead of the 
last line in the example above use:

    p.audio_segments(audio, freq, 10, "trial.wav", True)


Example exploring results visually:

    import pika as p
    
    (audio, freq, nBits) = p.load_audio(p.infile)   #here p.infile could be any desired audio file
    parser = p.AudioParser(audio[10*freq:30*freq], freq) #loads in the audio from second 10 to 30
    parser.pre_process()
    parser.harmonic_frequency()
    parser.plot_pika_from_harmonic() #Will show plots of the predicted results
