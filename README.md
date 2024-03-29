# Irony detection with LSTM y Tensorflow - Pyhton Pizza Holguín 2022
https://www.youtube.com/watch?v=cSLjgpdWytU&list=PLZwPvfQ1ZURcrfYuNuJbnCNLK7ufqK9Z2&index=13&t=205s

"Detectando ironías con LSTM y Tensorflow" 

In this repository you will find how to create and evaluate an irony classifier with an LSTM neural network that classifies twitter opinions into 0 (not ironic) or 1 (ironic). This is for the experimentation stage, wich is the more complex and longer stage in the MLOps process. The neural network is overfitting. It requires hiperparameter tuning, increasing the training dataset size, features engineering and/or a less complex algorithm.

The dataset used is from the International Workshop on Semantic Evaluation 2018 Task 3 Subtask A (https://github.com/Cyvhee/SemEval2018-Task3/tree/master/datasets). This was transformed to a csv file.

For words representation, I used the word-embedding model GloVe with 100 dimensions and 6 billion of words:
- https://nlp.stanford.edu/projects/glove/
- https://github.com/stanfordnlp/GloVe

Download and unzip "glove.6B.100d.txt" from https://nlp.stanford.edu/data/glove.6B.zip

Packages used (Python 3.7):
- Tensorflow 2.0
- Scikit-learn 0.20.3
- Pandas 0.24.2
- Numpy 1.18.5
