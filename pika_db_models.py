"""
SQLObject based database interface of models for pika calls/observation/etc.
    objects

#See sqlobject documentation for usage examples:
http://sqlobject.org/SQLObject.html

"""

from sqlobject import *
import sys

def init_db():
    #db_filename = os.path.abspath('pika.db')
    #if os.path.exists(db_filename):
        #os.unlink(db_filename)
    connection_string = "sqlite:/D:/Workspace/pika/pika.db"
    connection = connectionForURI(connection_string)
    sqlhub.processConnection = connection

def init_tables():
    """NOTE: this should only work if the tables don't already exist.
    If you want to reset the database you could delete the database manually
    then run this function to initialize the tables.
    """
    if sure and really_sure:
        Observer.createTable()
        Observation.createTable()
        Collection.createTable()
        recording.createTable()
        call.createTable()
    

class Observer(SQLObject):
    name = StringCol()
    institution = StringCol(default=None)

class Collection(SQLObject):
    """Collections may include multiple observations, observers and
    recordings all of which should originate from the same general area
    and trip and be stored in the same input folder.  Although it is possible
    to have multiple observers in one collection, I expect that it will
    mostly be just one observer per collection.

    example: Bob, Alice and Jane all go on a trip to unicorn falls and
    observe pika at several talus slopes along the trail.  All recordings
    made on this trip can be in the same collection.
    """
    folder = StringCol()
    start_date = DateCol()
    end_date = DateCol(default=None)
    description = StringCol(default=None)

class Observation(SQLObject):
    """Observations should be from a single specific location with a
    single observer on a single trip, but may include multiple recordings.
    
    example: Bob makes several recordings at the third talus slope along the 
    trail to unicorn falls.  Each of these recordings can be considered part
    of the same observation.
    
    Bob continues on and also records at the fifth talus slope.  The recordings
    there should be considered a separate observation.

    Jane and Alice both make recordings at the fourth talus slope along the 
    trail.  Since they are separate observers their observations should be
    catalogued separately even though they are at the same location on the
    same trip.
    """
    collection = ForeignKey("Collection")
    observer = ForeignKey("Observer")
    latitude = FloatCol(default=None)
    longitude = FloatCol(default=None)
    datum = StringCol(default=None) #Should probably always be WGS84 or NAD83 (I believe are equivalent)
    count_estimate = IntCol(default=None)
    notes = StringCol(default=None)

class Recording(SQLObject):
    """Individual recordings.  There should be one record for every file collected."""
    
    observation = ForeignKey("Observation")
    filename = StringCol()
    start_time = DateTimeCol()
    duration = FloatCol()
    bitrate = FloatCol()
    device = StringCol(default=None)

class Call(SQLObject):
    """Identified pika calls.  One record for each pika call identified during processing."""
    recording = ForeignKey("Recording")
    verified = BoolCol(default=None)
    offset = FloatCol()
    duration = FloatCol()
    filename = StringCol()
