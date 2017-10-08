# pika_detector 
This code is setup to create database entries and files around identifying pika (*Ochonta princeps*) calls in collections of audio files.  It uses django for the bulk of the user interface, but the analysis part is run on the command line (see below for details).

Some external libraries and tools are required - see setup_process.txt for notes about what else needs to be installed.

#### Initial setup
- Follow the steps in setup_process.txt to ensure helping programs are in place.
- In the pika_project directory create database setup by running:  
- - python manage.py makemigrations
- - python manage.py migrate
- Create an admin user for django (see tutorial: https://docs.djangoproject.com/en/1.9/intro/tutorial02/)
- Start server by running: python manage.py runserver
- Open a browser to http://127.0.0.1:8000/admin/ and login as the admin user, create other users if desired


#### Creating Observers, Collections and Recordings records
For a new install (some steps may be skipped if previously completed, e.g. creating an observer)
- Start up a server for django (if not already running) by running the following in the pika_project directory: python manage.py runserver
- In a browser go to http://127.0.0.1:8000/admin/pika_app/
- Create an observer by clicking the add link next to Observers, then save when finished
- Create a collection by clicking the add link next to Collections
- - Fill out the form for the collections details including folder which must be a subfolder of the collections folder in pika_project (collections folder may need to be created before use, also may need to update MEDIA_ROOT in piks_project/settings.py).  This folder should contain the recordings to be analyzed.
- - Scroll down to the bottom of the collection creation form and you can create a recording (or multiple recordings) record for the collection.  This interface will need to be updated, currently the filename drop down just gives you the option of any mp3 in the collections folder tree (rather than narrowing to the folder given in the collection form above).
- - Save when finished editing

#### Analyzing unprocessed records
First, process_records.py may need to be updated to match your system:  set the proj_path variable within the file to the directory where you have the project installed.

Then, from the pika_project folder in the console run:

    python process_records.py

This may take awhile to run, particularly on very large files.  On my system it takes 10 or 20 seconds to process a 3 minute recording.


#### Verifying identified calls
In the same way as with process_records.py, verify_calls.py may need to be updated to match your system before this will work.

Then from the pika_project folder in the console run:

    python verify_calls.py

This will lead you through a dialog for each unverified call, playing the audio and displaying the spectrogram of the hypothesized call and asking for verification as to whether it actually is a pika call.

If you know the recording id the database uses for the file you want analyzed you can limit your verification to that recording (there is not currently a particularly nice way of finding the recording id out before running the analysis - hopefully we will make this process more user friendly in the future).  For instance if the recording id is 7 you can run:

    python verify_calls.py 7

To see a basic analysis of the results after the verification process is complete you can run (again assuming you are interested in recording 7):

    python verify_calls.py a 7

This will give total count, count of true positives and of false positives.  It will also list out the times (in minutes:seconds) of the verified calls.
