import time

from sklearn.metrics import accuracy_score


# function to perform benchmarking operation
def benchmark(model, split_dataset):
    X_train, X_test, y_train, y_test = split_dataset
    print('Training {}..'.format(str(model).split('(')[0]))
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time() - start_time
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return end_time, accuracy


# print data
def print_data(model, accuracy, end_time):
    print('Training finished in {} seconds'.format(round(end_time, 4)))
    print('Benchmarks for {}'.format(str(model).split('(')[0]))
    print('Accuracy = {}'.format(round(100 * accuracy, 4)))
    print('Training Time = {} seconds\n'.format(round(end_time, 4)))
