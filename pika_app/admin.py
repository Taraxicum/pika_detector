from django.contrib import admin

from .models import Observer, Collection, Recording


class RecordingInline(admin.StackedInline):
    model = Recording
    extra = 1
    fields = ["collection", "start_time", "filename", "device", "notes"]

class CollectionAdmin(admin.ModelAdmin):
    inlines = [RecordingInline]

# Register your models here.
admin.site.register(Observer)
admin.site.register(Collection, CollectionAdmin)
