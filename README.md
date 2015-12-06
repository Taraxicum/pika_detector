# pika_detector 
This code is setup to create database entries and files around identifying pika (*Ochonta princeps*) calls in collections of audio files.  It will collect/ask for meta data including who collected the audio, locations, descriptions, etc.

Some external libraries and tools are required - see setup_process.txt for notes about what else needs to be installed.

#### Basic usage example
For initializing a collection you should have the mp3s in an accessible folder and be ready with whatever notes/gps coordinates/etc. you have for the collection, then in an ipython console the following code will take you through creation of the collection all the way to verifying the calls that were identified (optional):

    import pika as p
    collection = p.init_collection_and_associated_recordings()
    p.preprocess_collection(collection)
    p.identify_and_write_calls(collection)
    p.verify_calls(collection)

