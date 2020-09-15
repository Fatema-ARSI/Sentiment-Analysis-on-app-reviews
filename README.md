# Sentiment-Analysis-on-app-reviews
## Project Overview:
- Created a model to analyse the sentiments of the reviews which are provided by the customers on google olay store.
- Pre processed data applying tokenization and stopword from nltk to make the reviews in the formate nltl expects.
- Optimised NaiveBayesClassifier from Natural Lnguage ToolKit to reach the best model.

## Code and Resources used:
**Python:** 3.7.6.

**Packages Used:** pandas,numpy,nltk,pickle,random,re,gdown

## Dataset:

- Dataset can be download using this google drive link: **id 1sPDkfIWZPmCd_IsHjO9SXIFJreOI0S21**


## Model Building:

- Created three different arrays containing the reviews of negative,positive and neutral sentiments.
-Define function to remove stopwords such as of,to, etc. from the sentences and for each word, we create a dictionary with all the words and True. Why a dictionary? So that words are not repeated. If a word already exists, it won’t be added to the dictionary.
- Apply tokenization to the reviews which breaks the text paragraph into words.
- Combine the dataset and split it into train and test dataset using 25% ratio.

## Model Evaluation:

-Accuracy with the most ten informative features:

Accuracy is: 59.6555514840601

Most Informative Features

                 charged = True           Negati : Positi =     27.6 : 1.0
                   waste = True           Negati : Positi =     26.9 : 1.0
                    Easy = True           Positi : Neutra =     20.1 : 1.0
            disappointed = True           Negati : Positi =     19.7 : 1.0
                  turned = True           Negati : Positi =     19.5 : 1.0
                    bien = True           Positi : Neutra =     19.5 : 1.0
                 Awesome = True           Positi : Negati =     17.8 : 1.0
                   Doesn = True           Negati : Positi =     17.3 : 1.0
                suddenly = True           Negati : Positi =     15.3 : 1.0
                 useless = True           Negati : Positi =     14.4 : 1.0


## Result:



