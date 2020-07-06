# cil-text-classification
The code contains common practises, widely used in nlp tasks, that lead to a well performing sentiment analysis system, for ETH CIL Text classification challenge.

# preprocessing
The preprocessing step aims to reduce noise in the training and test data. It also groups training data in sentence-label pairs when exporting the final dataset.

# model 
The model implements a dual-path architecture that uses both LSTMs and self implemented attention techniques in order to extract rich information from the input data.

# extra info
* A bucketing technique is used to feed the data to the model. That way, different sentence lengths are handled efficiently without padding which slows down training time. 
* Glove 300d vectors are used to initialize word embeddings. Good results can be obtained by randomly intiailizing and training the ebedding matrix too.

# run code
In order to run the code you need to set the required data paths on top of process_data.py. Then, sentiment.py can be executed, after process_data.py has created the required files (embeddings.pickle, final_train_dataset.txt, final_test_dataset.txt, word_index.txt)
