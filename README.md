# pika_detector 
Code to detect pika (*Ochonta princeps*) calls in audio file.

There are two versions of code: the original matlab code created by Shankar and the Python version I have been adapting from it.

Example usage for the Python code:
    
    import pika as p
    
    (audio, freq, nBits) = p.load_audio("my_audio_file.wav")   
    #if you are using a stereo audio file you could uncomment the line
    #below to get a single channel from the stereo audio
    #left = [v[0] for v in audio] 
    
    p.audio_segments(audio, freq, 10, "trial.wav") 

