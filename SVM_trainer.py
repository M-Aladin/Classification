'''
This examples shows how a classifier is optimized by cross-validation,
which is done using the GridSearchCV on a development set.

The parameters of the estimator used to apply these methods are optimized by
cross-validated grid-search over a parameter grid.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# load data and split it to test and train
images_frame = pd.read_csv("train.csv")
x = images_frame.iloc[0:1000,1:]
y = images_frame.iloc[0:1000,:1]

# Split the dataset in two parts
train_images, test_images,train_labels, test_labels = train_test_split(x, y, test_size=0.2 ,train_size=0.8, random_state=1)

# Set the parameters need to be tuned
param_grid = [
    {'C': [1, 10, 100], 'kernel': ['linear'],'class_weight':[ None]},
    {'C': [1, 10, 100], 'gamma': [.0001,.00015,.001,.0015,.01], 'kernel': ['rbf'],'class_weight':[ None]},
    {'C': [1, 10, 100], 'degree' :[3,4,5],'kernel': ['poly'],'class_weight':[ None]},
 ]

# construct model selection and classifier
classifier = GridSearchCV(svm.SVC(verbose=True,probability=True), param_grid)
classifier.fit(train_images,train_labels.values.ravel())
accuracy = classifier.score(test_images,test_labels)

test_data=pd.read_csv('test.csv')
results=classifier.predict(test_data)

df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)