# irony_detection

"Detectando ironías con LSTM y Tensorflow" - Pyhton Pizza Holguín 2022

In this repository you will find an irony classifier that classifies twitter opinions into 0 (not ironic) or 1 (ironic).

The dataset used is from the International Workshop on Semantic Evaluation 2018 Task 3 Subtask A (https://github.com/Cyvhee/SemEval2018-Task3/tree/master/datasets). 

For words representation, I use the word-embedding model GloVe with 100 dimensions and 6 billion of words:
- https://nlp.stanford.edu/projects/glove/
- https://github.com/stanfordnlp/GloVe

Download and unzip "glove.6B.100d.txt" from https://nlp.stanford.edu/data/glove.6B.zip
