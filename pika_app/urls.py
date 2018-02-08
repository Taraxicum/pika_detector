from django.conf import settings
from django.conf.urls import url
from django.conf.urls.static import static
from . import views

urlpatterns = [
    url(r'^segment/([0-9]+)/([\.0-9]+)/([\.0-9]+)', views.analyze_segment, name='analyze'),
    url(r'^segment_spectrogram/([0-9]+)/([\.0-9]+)/([\.0-9]+)', views.segment_spectrogram, name='analyze'),
    url(r'^recording/([0-9]+)/calls', views.calls, name='recordings'),
    url(r'^recordings/', views.RecordingsView.as_view(), name='recordings'),
    url(r'^calls/', views.calls, name='calls'),
    url(r'^call/([0-9]+)', views.call),
    url(r'^verify_call/([0-9]+)()/([tT])', views.verify_call, name='verify_call'),
    url(r'^verify_call/([0-9]+)/([0-9]+)', views.verify_call, name='verify_call'),
    url(r'^verify_call/([0-9]+)', views.verify_call, name='verify_call'),
    url(r'^verification_response/([0-9]+)/(\w+)', views.verification_response),
    url(r'^$', views.index, name='index')
    ]
