# pika_detector 
This code is setup to create database entries and files around identifying pika (*Ochonta princeps*) calls in collections of audio files.  It uses django for the bulk of the user interface, but the analysis part is run on the command line (see below for details).

Some external libraries and tools are required - see setup_process.txt for notes about what else needs to be installed.

#### Initial setup
-Follow the steps in setup_process.txt to ensure helping programs are in place.
-in the pika_project directory create database setup by running:  
--python manage.py makemigrations
--python manage.py migrate
-Create an admin user for django (see tutorial: https://docs.djangoproject.com/en/1.9/intro/tutorial02/)
-start server by running: python manage.py runserver
-Open a browser to http://127.0.0.1:8000/admin/ and login as the admin user, create other users if desired


#### Creating Observers, Collections and Recordings records
For a new install - some steps may be skipped if previously completed (e.g. creating an observer)
-Start up a server for django (if not already running) by running the following in the pika_project directory: python manage.py runserver
-In a browser go to http://127.0.0.1:8000/admin/pika_app/
-Create an observer by clicking the add link next to Observers, then save when finished
-Create a collection by clicking the add link enxt to Collections
--Fill out the form for the collections details including folder which must be a subfolder of the collections folder in pika_project (collections folder may need to be created before use).  This folder should contain the recordings to be analyzed.
--Scroll down to the bottom of the collection creation form and you can create a recording (or multiple recordings) record for the collection.  This interface will need to be updated, currently the filename drop down just gives you the option of any mp3 in the collections folder tree (rather than narrowing to the folder given in the collection form above).
--Save when finished editing

#### Analyzing unprocessed records
First, process_records.py may need to be updated to match your system - set the proj_path variable within the file to the directory where you have the project installed.
Then, from the pika_project folder in the console run:
  python process_records.py
This may take awhile to run, particularly on very large files.  On my system it takes 10 or 20 seconds to process a 3 minute recording.

For initializing a collection you should have the mp3s in an accessible folder and be ready with whatever notes/gps coordinates/etc. you have for the collection, then in an ipython console the following code will take you through creation of the collection all the way to verifying the calls that were identified (optional):

    import pika as p
    collection = p.init_collection_and_associated_recordings()
    p.preprocess_collection(collection)
    p.identify_and_write_calls(collection)
    p.verify_calls(collection)

