import json
import glob
from pathlib import Path

REPORT_DIR = "./8kHz-13mfcc+delta+delta2-narrow/results"

for file in glob.glob(REPORT_DIR + "/*.json"):
    with open(file, "r") as fp:
        report = json.load(fp)
        report["returned_dict"]["loss"] = report["returned_dict"]["loss"]
    with open(file, "w+") as fp:
        json.dump(report, fp)

