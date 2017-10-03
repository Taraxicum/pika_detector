from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^recordings/', views.RecordingsView.as_view(), name='recordings'),
    url(r'^$', views.index, name='index')
    ]
