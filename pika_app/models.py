from __future__ import unicode_literals

from django.db import models
from django.conf import settings

# Create your models here.
class Observer(models.Model):
    name = models.CharField(max_length=200)
    institution = models.CharField(max_length=200, blank=True)
    
    def __str__(self):
        return "{}; {}".format(self.name, self.institution)

class Collection(models.Model):
    observer = models.ForeignKey(Observer)
    folder = models.FilePathField(path=settings.MEDIA_ROOT, 
            allow_folders=True, allow_files=False, recursive=True) #May need to adjust
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

class Recording(models.Model):
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE)
    #filename = models.FilePathField(self.observation.collection.folder) #May need to adjust
    start_time = models.DateTimeField()
    filename = models.FilePathField(path=settings.MEDIA_ROOT,
            match="\.mp3$", recursive=True, default=None, blank=True) 
    duration = models.FloatField(default=None, null=True, blank=True)
    sample_frequency = models.FloatField(default=None, null=True, blank=True)
    device = models.CharField(max_length=100, default=None, 
            null=True, blank=True)
    notes = models.TextField(default=None, blank=True)
    processed = models.BooleanField(default=False)

    def __str__(self):
        return self.filename
    
class Call(models.Model):
    recording = models.ForeignKey(Recording, on_delete=models.CASCADE)
    verified = models.NullBooleanField()
    offset = models.FloatField(null=True, blank=True)
    duration = models.FloatField(null=True, blank=True)
    filename = models.FilePathField("./collections") #May need to adjust
    
    def __str__(self):
        return self.filename

