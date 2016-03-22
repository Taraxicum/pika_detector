from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^recordings/', views.RecordingsView.as_view(), name='recordings'),
    url(r'^recording/(?P<pk>[0-9]+)/process/$', views.ProcessView.as_view(),
        name='process')
    url(r'^$', views.index, name='index')
    ]
