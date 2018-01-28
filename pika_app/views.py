from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponse
from django.views import generic
from django.core.exceptions import ObjectDoesNotExist

from call_handler import CallCounter
from pika2 import Parser
from .models import Call, Recording

import soundfile

# Create your views here.
def index(request):
    return HttpResponse("YAY")

class RecordingsView(generic.ListView):
    template_name = "pika_app/recordings.html"
    context_object_name = "recording_list"
    
    def get_queryset(self):
        return Recording.objects.order_by("id")[:5]

def calls(request, recording_id=None):
    template_name = "pika_app/calls.html"
    call_summary = None
    if recording_id:
        try:
            recording = Recording.objects.get(pk=recording_id)
            call_list = recording.calls.order_by("id")
            confirmed = wrong = unclassified = 0
            for call in call_list:
                if call.verified is None:
                    unclassified += 1
                elif call.verified:
                    confirmed += 1
                else:
                    wrong += 1
            call_summary = {
                    'total':len(call_list),
                    'confirmed':confirmed,
                    'wrong':wrong,
                    'unclassified':unclassified,
                    }
        except ObjectDoesNotExist:
            call_list = []

    else:
        call_list = Call.objects.order_by("id")
    return render(request, template_name, {'call_list': call_list[:150], 'summary':call_summary})

def verify_call(request, call_id, verified=None):
    call = Call.objects.get(pk=call_id)
    try:
        next_call = Call.objects.filter(verified__isnull=True).filter(id__gt=call_id).order_by("id")[0]
    except IndexError:
        next_call = None
    calls_link = "/pika_app/recording/{}/calls".format(call.recording.id)
    template_name = "pika_app/call.html"
    return render(request, template_name, {'call': call, 
        'next_call':next_call,
        'verified':verified,
        'calls_link':calls_link,
        })

def verification_response(request, call_id, response):
    call = Call.objects.get(pk=call_id)
    if response == 'y':
        call.verified = True
        call.save()
    elif response == 'n':
        call.verified = False
        call.save()
    else:
        return redirect('verify_call', call_id=call_id)
    try:
        next_call = Call.objects.filter(verified__isnull=True).filter(id__gt=call_id).order_by("id")[0]
    except IndexError:
        return redirect('calls')
    return redirect('verify_call', next_call.id, call.id)

def call(request, call_id):
    call = Call.objects.get(pk=call_id)
    return call.spectrogram

def analyze_segment(request, recording_id, start_second, end_second):
    recording = Recording.objects.get(pk=recording_id)
    parser = Parser(recording.recording_file, CallCounter())
    title = "Frequency report: Parser {}, recording {}".format(parser.frequency, recording.sample_frequency)
    return parser.analyze_interval([int(start_second), int(end_second)], nice_plotting=True, title=title, to_http=True)

    #template_name = "pika_app/segment.html"
    #return render(request, template_name, {'test':"WEE", 'rec':recording.recording_file})
