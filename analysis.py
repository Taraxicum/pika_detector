import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import os, sys
#Got this setup from:
#https://www.stavros.io/posts/standalone-django-scripts-definitive-guide/
proj_path = "D:/Workspace/pika/"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pika_project.settings")
sys.path.append(proj_path)
os.chdir(proj_path)

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
from pika_app.models import Recording, Call
from pika2 import Parser
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def get_all_judged_calls():
    return Call.objects.filter(verified__isnull=False)

def get_call_features(call_file):
    parser = Parser(call_file, None, step_size_divisor=64)
    parser.filtered_fft(filter_on=True)
    frame_count = len(parser.fft)
    avg_fft = np.sum(parser.fft, axis=0)/frame_count
    max_fft = np.max(parser.fft, axis=0)
    ratios = []
    for i, f in enumerate(max_fft):
        ratios.append(f/avg_fft[i])
    median_ratio = np.median(ratios)
    mean_ratio = np.mean(ratios)
    overall_avg = np.mean(avg_fft)
    return [frame_count, overall_avg, median_ratio, mean_ratio]

def featurize_calls(calls):
    output = []
    for call in calls:
        output.append(get_call_features(call.filename))
    return output

def analyze_calls(features, target, max_depth=None, n_estimators=200):
    X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=.33, random_state=42)
    clf = RandomForestClassifier(
            max_depth=max_depth,
            class_weight={0:1, 1:2},
            n_estimators=n_estimators,
            random_state=42,
            )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    return clf, X_test, y_test

def chart_features(features, targets, colors=None):
    #colors = np.random(len(features))#['r' if target else 'b' for target in targets]
    plt.scatter([f[0] for f in features], [f[1] for f in features], c=colors)
    plt.show()

def test():
    print("hoo")
