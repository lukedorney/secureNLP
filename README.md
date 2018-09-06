# secureNLP
An attempt at SemEval 2018's shared task 8 (SecureNLP).
This project addresses parts one and two of this shared task:
1. Classification of sentences as being relevant or not to the task of extracting information about malware capabilities.
2. Structure prediction of sentences (in BIO format) for Entities, Actions, and Modifiers containing information about malware capabilities.

For more information see the [official page](https://competitions.codalab.org/competitions/17262). Access to the data set can also be found through this website (although you do have to contact the administrators)

Results can be seen [here](https://competitions.codalab.org/competitions/17262#results). 
As of 9/5/2018, (compared to scores from the evaluation period) highest score for Subtask 2 relaxed score, 3rd place in Subtask 1, and 4th place for Subtask 2 strict score
***
To run this project:
   1. Obtain the data set
   2. Make sure all file locations for the data are accurate in config.py
   3. Run data_process.py
   4. Run the first task with sent_classification_sgd.py
   5. Run the second task with entity_recognition_crf.py