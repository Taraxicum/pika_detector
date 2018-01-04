from __future__ import unicode_literals

from django.db import models
from django.conf import settings
import os
import mutagen.mp3

import pika2

# Create your models here.
def recording_path(instance, filename):
    ##EA 8/1/16
     # CONSIDER move elsewhere and import
     # CONSIDER expand to provide layout for all pathing for app?
    rec_path = "{}/Collections/{}/collection_{}/" .format(
            settings.MEDIA_ROOT,
            instance.collection.observer.name,
            instance.collection.id)
    return '{0}/{1}'.format(rec_path, filename)


class Observer(models.Model):
    name = models.CharField(max_length=200)
    institution = models.CharField(max_length=200, blank=True)

    def __str__(self):
        return "{}; {}".format(self.name, self.institution)

    class Meta:
        app_label = "pika_app"

class Collection(models.Model):
    observer = models.ForeignKey(Observer, related_name='collections')
    #folder = models.FilePathField(path=settings.MEDIA_ROOT,
    #        allow_folders=True, allow_files=False, recursive=True) #May need to adjust
    latitude = models.DecimalField(max_digits=9, decimal_places=6,
            default=None, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6,
            default=None, null=True, blank=True)
    datum = models.CharField(max_length = 100, default=None,
            null=True, blank=True)
    count_estimate = models.IntegerField(default=None, null=True, blank=True)
    description = models.TextField(default=None, blank=True, max_length=200)
    notes = models.TextField(default=None, blank=True)

    def __str__(self):
        return "{}: {}".format(self.observer.name, self.description)

    class Meta:
        app_label = "pika_app"

class Recording(models.Model):
    collection = models.ForeignKey(Collection,
            related_name='recordings',
            on_delete=models.CASCADE)
    #filename = models.FilePathField(self.observation.collection.folder) #May need to adjust
    start_time = models.DateTimeField()
    #filename = models.FilePathField(path=settings.MEDIA_ROOT,
    #        match="\.mp3$", recursive=True, default=None, blank=True)
    ## EA 8/1/16 replaced with recording_file
    recording_file = models.FileField(default="", upload_to=recording_path)
    duration = models.FloatField(default=None, null=True, blank=True)
    sample_frequency = models.FloatField(default=None, null=True, blank=True)
    device = models.CharField(max_length=100, default=None,
            null=True, blank=True)
    notes = models.TextField(default=None, blank=True)
    processed = models.BooleanField(default=False)

    def output_folder(self):
        return os.path.dirname(self.recording_file.path) + "/recording{}/".format(self.id)

#    def __str__(self):
#        return self.filename
## EA 8/1/16 replaced with alternate below
    def __str__(self):
        return "{}_collection{}_recording{}".format(self.collection.observer.name, self.collection.id, self.id)


#    def save(self, *args, **kwargs):
        #Update duration on save if it hasn't already been initialized
#        if self.duration is None:
#            self.duration = mutagen.mp3.MP3(self.filename).info.length
#        super(Recording, self).save(*args, **kwargs)

    class Meta:
        app_label = "pika_app"


class Call(models.Model):
    recording = models.ForeignKey(Recording,
            related_name='calls',
            on_delete=models.CASCADE)
    verified = models.NullBooleanField()
    offset = models.FloatField(null=True, blank=True)
    duration = models.FloatField(null=True, blank=True)
    filename = models.FilePathField(null=True, blank=True) #May need to adjust
    @property
    def local_filename(self):
        return self.filename[len(settings.MEDIA_ROOT):]

    @property
    def spectrogram(self):
        parser = pika2.Parser(self.filename, None, self.offset, step_size_divisor=64)
        return parser.get_spectrogram(self, to_http=True)
    
    @property
    def offset_display(self):
        minutes = self.offset//60
        seconds = int(self.offset)%60
        partial = int((self.offset%1)*10)
        return "{:.0f}:{:02d}.{:1d}".format(minutes, seconds, partial)
    
    def __str__(self):
        return self.filename

    class Meta:
        app_label = "pika_app"
