
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def get_quality(quality):
    """
    Return the quality of the wine.
    0-3 is bad, 4-7 is good, and 8+ is excellent
    """
    if quality <= 3:
        return 0
    
    elif quality <= 6:
        return 1
    
    else:
        return 2

# Read in the wine dataset
df = pd.read_csv("winequality-white.csv", delimiter=';')

# Split quality into bad, good, and excellent
df["quality"] = df["quality"].apply(get_quality)

data = df.values.transpose()
X = data[:len(data) - 1].transpose()
y = data[len(data) - 1]

# Train our model using 10-fold cv
clf = RandomForestClassifier()
res = cross_val_score(clf, X, y, cv=10)

print("Scores: " + str(res))
