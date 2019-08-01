# -*- coding: utf-8 -*-
# Generated by Django 1.11.4 on 2017-10-03 03:40
from __future__ import unicode_literals

from django.db import migrations, models
import pika_app.models


class Migration(migrations.Migration):

    dependencies = [
        ('pika_app', '0002_auto_20171002_2012'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='collection',
            name='folder',
        ),
        migrations.RemoveField(
            model_name='recording',
            name='filename',
        ),
        migrations.AddField(
            model_name='recording',
            name='recording_file',
            field=models.FileField(default='', upload_to=pika_app.models.recording_path),
            preserve_default=False,
        ),
    ]