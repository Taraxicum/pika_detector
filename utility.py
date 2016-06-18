"""Utility functions for pika.py

Mostly UI helper functions

"""
import glob
import subprocess
import datetime
import os
import re
import mutagen.mp3

def confirm(prompt):
    while True:
        value = raw_input(prompt + " (y/n)").lower()
        if value == "n":
            return False
        elif value == "y":
            return True
        else:
            print "invalid input, should be 'y' or 'n' but got {}".format(value)

def get_verification(call, with_audio=True):
    """Return false if quit otherwise return True when valid response given"""
    volume_mult = 20
    if with_audio:
        play_audio(call.filename, volume_mult)
    while True:
        print "Verify as pika call?"
        r = raw_input("(Y)es/(N)o/(S)kip/(R)eplay/(L)ouder/(Q)uit (then press enter)")
        r = r.lower()
        if r == "q":
            return "q"
        if r == "y":
            return True
        elif r == "n":
            return False
        elif r == "s":
            return "s"
        elif r == "l":
            volume_mult += 20
            play_audio(call.filename, volume_mult)
        elif r == "r":
            play_audio(call.filename, volume_mult)


def play_audio(audio, vol_mult=20, start=0, duration=60):
    subprocess.call(["ffplay", "-nodisp", "-autoexit",
        "-ss", str(start), "-t", str(duration),
        "-loglevel", "0", "-af", "volume={}".format(vol_mult), audio])


