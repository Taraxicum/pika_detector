from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse
from django.views import generic

from .models import Recording

# Create your views here.
def index(request):
    return HttpResponse("YAY")

class RecordingsView(generic.ListView):
    template_name = "pika_app/recordings.html"
    context_object_name = "recording_list"
    
    def get_queryset(self):
        return Recording.objects.order_by("id")[:5]

