import pika2
import call_handler as ch
import sys
import os
import numpy as np

if __name__== '__main__':
    #Got this setup from:
    #https://www.stavros.io/posts/standalone-django-scripts-definitive-guide/
    proj_path = "D:/Workspace/pika_project/"
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pika_project.settings")
    sys.path.append(proj_path)
    os.chdir(proj_path)

    from django.core.wsgi import get_wsgi_application
    application = get_wsgi_application()

from pika_app.models import Recording, Call

def main(argv=None):
    r_id = -1
    analyze = False

    if len(sys.argv) == 2:
        try:
            r_id = int(sys.argv[1])
        except ValueError:
            raise(Exception("Call verification argument should be an integer, got: {}".format(sys.argv[1])))
    elif len(sys.argv) == 3:
        if sys.argv[1] == "a":
            try:
                r_id = int(sys.argv[2])
                analyze = True
            except ValueError:
                raise(Exception("Call analysis argument should be an integer, got: {}".format(sys.argv[2])))
    
    if analyze:
        base_query = Call.objects.filter(recording_id=r_id)
        total_count = base_query.count()
        true_positive = base_query.filter(verified=True).count()
        false_positive = base_query.filter(verified=False).count()
        print("Total count: {}\nTrue postives: {}\nFalse positives {}".format(
                total_count, true_positive, false_positive))
        print("times of verified calls:")
        print(", ".join(["{:.0f}:{:2.1f}".format(np.floor(x.offset/60), x.offset%60) for x in base_query.filter(verified=True)]))
    else:
        if r_id > 0:
            calls = Call.objects.filter(verified__isnull=True).filter(
                    recording_id=r_id)
        else:
            calls = Call.objects.filter(verified__isnull=True)

        print("Will be processing {} calls".format(len(calls)))

        for call in calls:
            if not pika2.verify_call(call): #Means user response was to quit
                return

if __name__ == "__main__": main()    
