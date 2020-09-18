# text_classification_benchmarks_using_sparse_features
Benchmarking performances of 12 different ml algorithms on real-world text data using tfidf features
## Info
This is a project for observing and benchmarking performance metrics of different machine learning algorithms on real-world text dataset.<br><br>
Dataset used for this project is <b>20_newsgroups dataset</b>.<br>
<b>Reference</b> : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html<br>

## Usage
For installing requirements, do<br>
<code>
$pip install requirements.txt
</code><br><br>
For downloading 20_newsgroup dataset, do<br>
<code>
$python download_dataset.py<br>
</code><br><br>
For producing text classification benchmarks on 20_newsgroup dataset, do<br>
<code>
$python train.py<br>
</code><br><br>

## Review
The train.py script trains following models and generates benchmarks<br>

| models                      |   accuracies |   training_times (seconds)|
|:----------------------------|-------------:|-----------------:|
| RidgeClassifier             |     0.924708 |        3.57152   |
| SGDClassifier               |     0.924708 |        1.49311   |
| Perceptron                  |     0.895723 |        0.866869  |
| PassiveAggressiveClassifier |<b>0.925415</b> |        1.72583   |
| BernoulliNB                 |     0.870979 |        0.198357  |
| ComplementNB                |     0.90562  |        0.168547  |
| MultinomialNB               |     0.906681 |        0.158578  |
| KNeighborsClassifier        |     0.815129 |        0.0100126 |
| NearestCentroid             |     0.767055 |        0.0907581 |
| DecisionTreeClassifier      |     0.62743  |       22.9773    |
| RandomForestClassifier      |     0.828208 |       67.5984    |
| AdaBoostClassifier          |     0.541181 |       92.879     |
