# Real or Fake News? A Machine Learning Model to Detect Fake News 🧠

![image](https://user-images.githubusercontent.com/100214297/186528994-8007a67a-b285-466c-91f5-8c872666d5df.png)

(The Week, accessed 09.08.22)

Can anyone tell what is true and what is fake news anymore? From the weaponisation of Covid-19, to devastating meteors en route to destroy earth, and that notion that Pope Francis endorsed Trump in the US presidential race…

Most fake news, however, is difficult to decipher which has led to many people airing on the side of caution and / or increasing their deep mistrust of the media.

Approximately 45% of adults in the UK believe they see or read fake news everyday (JournoLink, 2022).
 
Definition of fake news: 

    ‘false stories that appear to be news, spread on the internet or using other media, usually created to influence political views or as a joke' - Cambridge Dictionary

The reported effects of fake news includes: distrust in the media, the democratic process being undermined, increases in conspiracy theories and hate speech, and the general spread of false information to the detriment of public safety. 

Therefore, we aimed to build a machine learning model to predict whether news is real or fake. We hosted our findings and presentation on aan interactive site:

Front-end repo here: https://github.com/dianaagustinaf/is-it-fake.git 

## Dataset:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download 


### Back-End Tools:

Importing Data: Spark 
Data Preprocessing: Python, Pandas
Machine Learning: Supervised machine learning with Natural Language Programming (Natural Language Understanding) and BERT (Bidirectional Encoder Representations from Transformers). 

### Front-End & Data Visualisation:

Web Design Structure: HTML, CSS (inc. Bootstrap) 
Data Visualisations: D3, Plotly Express, Matplotlib 
Storytelling: Scrollama (JS library)

## Machine Learning Process

AI  ⇨  Machine Learning ⇨  Deep Learning 


Steps:

Extract Data

⬇️

Explore Data

⬇️

Clean Data

⬇️

Visualise Data

⬇️

Pre-process Data

⬇️

Compile (Deep Learning Models only)

⬇️

Train Model

⬇️

Evaluate Models

⬇️

Test Models

## Initial Data exploration and Visualisation

[ADD IN SHOLA'S IMAGES HERE]

## Pre-processing the  Data
Target Column: Title. Title words converted into lowercase, removed all stop words and special characters. 

## Tokenization 
We trialed 2 ways of tokenizing:
1. Text data was tokenized with RegexpTokenizer() to split strings into substrings using regular expression and then a Stemming algorithm PorterStemmer() was used to stem the tokenized words (break down sentences into words).
2. The text data was tokenized using a Tensforflow Tokenizer library and then sequenced.

## Train-Test Split
Data was split into Train and Test data to evaluate the performance of our Machine Learning Algorithm.

## Models: 6 Binary Classification models
We chose 6 machine learning algorithms best suited for Binary Classification problems to find the best model for predicting Fake New. This required an additonal preprocessing step: vectorisation - we used CountVectoriser.

Result: SVC() has the best score of all the models we tested.

![image](https://user-images.githubusercontent.com/100214297/186531513-631f22e4-cbb4-482f-935e-04dfa4271c05.png)

## Neural Network Models: NLP (sequential Model) & BERT 
To go even deeper into 

## NLP Sequntial Model

For the Supervised Learning Model we used a Keras Sequential deep learning model. We created a neural network model and the resulting accuracy was higher than the previous models. This model works best as a binary classifier and additional layers were added to account for the text classification: the Embedding layer converts sequences into arrays of word vectors and the Dense layer classifyies arrays of word vectors.

![image](https://user-images.githubusercontent.com/100214297/186530881-c24c251a-c62d-427c-b5c0-037c3d781ed9.png)

The below plot highlights the accuracy of the model:

![image](https://user-images.githubusercontent.com/100214297/186531167-ac424b6b-dc3f-46c6-ba42-cb902bb9c6ff.png)

Although, we suspected that the model may have overfitted on the training data due to the low loss rate and high accuracy rate in addition to the slight increase in the validation loss.

## BERT

Lastly, we trained a BERT model. BERT is a new method of pre-training language representations from Google which outperforms previous methods on a variety of Natural Language Processing (NLP) tasks. 

BERT stands for “Bidirectional Encoder Representations from Transformers”. It uses a transformer model, applying the bidirectional training of Transformer using an encoder-decoder architecture. BERT only uses the encoder part of this architecture because its goal is to generate a language model.

Apart from splitting the Test and Training data we skipped all the Pre-processing steps in our other Models because we used Tensorflow-hub’s BERT preprocesser and encoder.

## Comparison of Models
We used confusion matricies to evaluate the models, and we this is what we found:

[ADD IN CM (SHOLA)]

# Analysis
We tested our NLP model on completely new unseen data. We chose fake news text and true news text and ran it through our model. 

## True News Test 
'China Signals Missile Launch Over Taiwan' from USNEWS (3 Aug, 2022)
This string resulted in a prediction probability of 0.9997772. This therefore tested correctly using our model 

## Fake News Test
'All of this without even discussing the millions of fraudulent votes that were cast or altered!' Tweet by Trump [Twitter] (Jan 01, 2021).
This string resuted in a prediction probability of 0.036136717 which tested correctly using our model.

Variations of this text was repeated by Trump 76 times. The implications for democracy is explicit. With this qoute in addition to Trump's insistence that the election was rigged, and the dozens of lawsuits, aims to undermine the democratic process and intensify the emotions of Trump's followers.

Indeed, this claim directly led to a mob of Trump supporters storming the Capitol on Jan. 6 while formal certification of Biden’s victory was underway (Washington Post, 2022).

Similar actions by past leaders has lead to dire consequences on citizens of nations, world conflicts, and deep mistrust in the democratic process.

# Implications for Research:
If fake news could be distinguished from real news then the spread of misinformation would reduce. It would also have positive ramifications for democratic processes, the spread of hate speech, and public safety. 
