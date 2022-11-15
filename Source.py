import csv
from sklearn import ensemble
###### Author:  Rares-Augustin Rascanu (421A)
###### Serving: ISIA Project
###### Date:    15/11/2022

#### Begin ####

### Working variables
data = []
tags = []
classifier = ensemble.BaggingClassifier()

### Back-end functions
## Function: getType
# Returnes steel fault based on the database value.
def getType(item):
    types = {
        1 : "Pastry",
        2 : "Z_Scratch",
        3 : "K_Scatch",
        4 : "Stains",
        5 : "Dirtiness",
        6 : "Bumps",
        7 : "Other_Faults"
    }
    for i in range(27, 34):
        if item[i] == "1":
            return types[int(i - 26)]

## Function: getAccuracy
# Calculates the accuracy of a prediction set using
# the percentage formula.
def getAccuracy(original, predictions):
    if len(original) != len(predictions):
        raise Exception('Data samples must have the exact same length.')
    correct = 0
    for i in range(0, len(original)):
        if getType(original[i]) == predictions[i]:
            correct += 1
    return (correct * 100) / len(original)

## Function: prettyFormat
# Formats a float value to a string of a two-decimal
# float.
def prettyFormat(value):
    return "{:.2f}".format(value)

### Front-end functions
## Function: setup
# Reads data from file into a list[] and determines
# a tag for each item.
def setup(file):
    with open(file) as dataFile:
        reader = csv.reader(dataFile, delimiter = "\t")
        for row in reader:
            data.append(row)
        for item in data:
            tags.append(getType(item))

## Function: test
# Predicts a type for each item in the data sample using
# the previously determined tags. Can have sample &
# feature size adjusted for result variance.
def test(samples, features):
    classifier.max_samples = samples
    classifier.max_features = features
    classifier.fit(data, tags)
    print("Accuracy for {samples}% 'in-bag' & {features}% dimensions:".format(samples = int(samples * 100), features = int(features * 100)), end=" ")
    print(prettyFormat(getAccuracy(data, classifier.predict(data))), end = "%\n")

### Hero
##Setup
setup("Faults.NNA")

## Tests for 25% 'in-bag' percentage
test(0.25, 0.1)
test(0.25, 0.5)
test(0.25, 0.8)

## Tests for 50% 'in-bag' percentage
test(0.5, 0.1)
test(0.5, 0.5)
test(0.5, 0.8)

## Tests for 85% 'in-bag' percentage
test(0.85, 0.1)
test(0.85, 0.5)
test(0.85, 0.8)

#### End ####