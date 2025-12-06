Slot Tagging with Bidirectional LSTM
=====================================

Author: NLP 243 Student
Assignment: Homework 3

DESCRIPTION
-----------
This project implements a Bidirectional LSTM model for slot tagging in 
movie-related user queries. The model tags each word in a sentence with 
its corresponding slot label (e.g., movie names, directors, genres).

MODEL ARCHITECTURE
------------------
- Embedding Layer (128 dimensions)
- 2-layer Bidirectional LSTM (256 hidden units)
- Dropout (0.5)
- Linear output layer

REQUIREMENTS
------------
Install dependencies:
    pip install -r requirements.txt

Required packages:
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- seqeval >= 1.2.0

USAGE
-----
To train the model and generate predictions:

    python main.py train.csv test.csv test_pred.csv

Arguments:
    train.csv     - Training data file with columns: utterance, IOB Slot tags
    test.csv      - Test data file with column: utterance  
    test_pred.csv - Output file for predictions

DATA FORMAT
-----------
Training data (train.csv):
    id,utterance,IOB Slot tags
    0,show me dramas from china,O O B_genre O B_country

Test data (test.csv):
    id,utterance
    0,find movies directed by spielberg

Output (test_pred.csv):
    id,tags
    0,O O O O B_director

HYPERPARAMETERS
---------------
- Embedding dimension: 128
- Hidden dimension: 256
- Number of LSTM layers: 2
- Dropout: 0.5
- Learning rate: 0.001
- Batch size: 32
- Epochs: 30

FILES
-----
- main.py       : Entry point and configuration
- model.py      : BiLSTM model definition
- dataset.py    : Data loading and preprocessing
- train.py      : Training and prediction functions
- requirements.txt : Dependencies
- readme.txt    : This file
