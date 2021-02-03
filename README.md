# Sentiment-Analysis-on-app-reviews
## Project Overview:
- Created a model to analyse the sentiments of the reviews which are provided by the customers on google play store.
- Pre processed data applying tokenization and stopword from nltk to make the reviews in the formate nltl expects.
- Optimised NaiveBayesClassifier from Natural Language ToolKit to reach the best model.

## Code and Resources used:
**Python:** 3.7.6.

**Packages Used:** pandas,numpy,nltk,pickle,random,re,gdown

## Dataset:

- Dataset can be download using this google drive id:[ id 1sPDkfIWZPmCd_IsHjO9SXIFJreOI0S21]('https://drive.google.com/file/d/1sPDkfIWZPmCd_IsHjO9SXIFJreOI0S21/view?usp=sharing)


## Model Building:

- Created three different arrays containing the reviews of negative,positive and neutral sentiments.
-Define function to remove stopwords such as of,to, etc. from the sentences and for each word, we create a dictionary with all the words and True. Why a dictionary? So that words are not repeated. If a word already exists, it won’t be added to the dictionary.
- Apply tokenization to the reviews which breaks the text paragraph into words.
- Combine the dataset and split it into train and test dataset using 25% ratio.

## Model Evaluation:

-Accuracy with the most ten informative features:

Accuracy is: 59.21092564491654

Most Informative Features

                  Nul = True           Negati : Positi =     48.5 : 1.0
             désinstallé = True           Negati : Positi =     31.7 : 1.0
             déconseille = True           Negati : Positi =     31.7 : 1.0
                     Pub = True           Negati : Positi =     30.3 : 1.0
              Impossible = True           Negati : Positi =     26.7 : 1.0
               injouable = True           Negati : Positi =     21.6 : 1.0
                 intérêt = True           Negati : Positi =     21.0 : 1.0
                   abusé = True           Negati : Positi =     18.2 : 1.0
                   gâche = True           Negati : Positi =     15.2 : 1.0
                 affiche = True           Negati : Positi =     14.6 : 1.0


