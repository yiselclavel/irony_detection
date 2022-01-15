# Irony detection with LSTM y Tensorflow - Pyhton Pizza Holguín 2022

"Detectando ironías con LSTM y Tensorflow"

In this repository you will find how to create and evaluate an irony classifier with LSTM neural network that classifies twitter opinions into 0 (not ironic) or 1 (ironic). This is for the experimentation stage, wich is the more complex and longer stage in the MLOps process. It requires hiperparameter tuning.

The dataset used is from the International Workshop on Semantic Evaluation 2018 Task 3 Subtask A (https://github.com/Cyvhee/SemEval2018-Task3/tree/master/datasets). This was transformed to a csv file.

For words representation, I used the word-embedding model GloVe with 100 dimensions and 6 billion of words:
- https://nlp.stanford.edu/projects/glove/
- https://github.com/stanfordnlp/GloVe

Download and unzip "glove.6B.100d.txt" from https://nlp.stanford.edu/data/glove.6B.zip

Packages used:
- Tensorflow 2.0
- Scikit-learn 0.20.3
- Pandas 0.24.2
- Numpy 1.18.5
