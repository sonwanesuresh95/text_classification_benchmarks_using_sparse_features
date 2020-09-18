import time

# For calculating total execution time
init_time = time.time()

# import necessary libraries
import json
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from benchmark import benchmark, print_data

# set random state = 0
np.random.seed(0)

print('This Script trains 12 different machine learning models on 20 newsgroups dataset and saves benchmarks in '
      'benchmark.json')
# fetch dataset
print('Fetching Dataset..')
data = fetch_20newsgroups(data_home=r'./data')
print('Dataset fetched successfully.\n')

# Converting to features
print('Converting to text features into Tfidf features..')
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(data.data)
print('Transformation done.\n')
target = data.target

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(features, target)
split_dataset = (X_train, X_test, y_train, y_test)

# For storing benchmarks
models = []
training_times = []
accuracies = []

# P.S. The best hyperparameters for these classifiers were found using GridSearchCV
# For more info contact sonwanesuresh95@gmail.com

# Training a ridge classifier
model = RidgeClassifier(solver='auto', tol=1e-2)
end_time, accuracy = benchmark(model, split_dataset)
models.append(str(model).split('(')[0])
training_times.append(end_time)
accuracies.append(accuracy)
print_data(model, accuracy, end_time)

# Training a SGDCLassifier
model = SGDClassifier(penalty='l2')
end_time, accuracy = benchmark(model, split_dataset)
models.append(str(model).split('(')[0])
training_times.append(end_time)
accuracies.append(accuracy)
print_data(model, accuracy, end_time)

# Training a Perceptron
model = Perceptron()
end_time, accuracy = benchmark(model, split_dataset)
models.append(str(model).split('(')[0])
training_times.append(end_time)
accuracies.append(accuracy)
print_data(model, accuracy, end_time)

# Training a PassiveAggressiveClassifier
model = PassiveAggressiveClassifier()
end_time, accuracy = benchmark(model, split_dataset)
models.append(str(model).split('(')[0])
training_times.append(end_time)
accuracies.append(accuracy)
print_data(model, accuracy, end_time)

# Training a BernoulliNB
model = BernoulliNB(alpha=1e-10)
end_time, accuracy = benchmark(model, split_dataset)
models.append(str(model).split('(')[0])
training_times.append(end_time)
accuracies.append(accuracy)
print_data(model, accuracy, end_time)

# Training a ComplementNB
model = ComplementNB(alpha=0.01)
end_time, accuracy = benchmark(model, split_dataset)
models.append(str(model).split('(')[0])
training_times.append(end_time)
accuracies.append(accuracy)
print_data(model, accuracy, end_time)

# Training a MultinomialNB
model = MultinomialNB(alpha=0.01)
end_time, accuracy = benchmark(model, split_dataset)
models.append(str(model).split('(')[0])
training_times.append(end_time)
accuracies.append(accuracy)
print_data(model, accuracy, end_time)

# Training a KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
end_time, accuracy = benchmark(model, split_dataset)
models.append(str(model).split('(')[0])
training_times.append(end_time)
accuracies.append(accuracy)
print_data(model, accuracy, end_time)

# Training a NearestCentroid
model = NearestCentroid()
end_time, accuracy = benchmark(model, split_dataset)
models.append(str(model).split('(')[0])
training_times.append(end_time)
accuracies.append(accuracy)
print_data(model, accuracy, end_time)

# Training a DecisionTreeClassifier
model = DecisionTreeClassifier()
end_time, accuracy = benchmark(model, split_dataset)
models.append(str(model).split('(')[0])
training_times.append(end_time)
accuracies.append(accuracy)
print_data(model, accuracy, end_time)

# Training a RandomForestClassifier
model = RandomForestClassifier()
end_time, accuracy = benchmark(model, split_dataset)
models.append(str(model).split('(')[0])
training_times.append(end_time)
accuracies.append(accuracy)
print_data(model, accuracy, end_time)

# Training a AdaBoostClassifier
model = AdaBoostClassifier()
end_time, accuracy = benchmark(model, split_dataset)
models.append(str(model).split('(')[0])
training_times.append(end_time)
accuracies.append(accuracy)
print_data(model, accuracy, end_time)

# print total time taken by the script
print('Total time taken by the script = {} seconds'.format(time.time() - init_time))

# produce benchmarks.json
bench = {'models': models,
         'accuracies': accuracies,
         'training_times': training_times}

# save benchmarks in
with open('benchmarks.json', 'w') as f:
    json.dump(bench, f)
