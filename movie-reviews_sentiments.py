import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import movie_reviews
import nltk.classify.util
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from nltk.classify import NaiveBayesClassifier

# Receive all the words from the files as dictationary 
def word_feature(words):
    return dict([(word, True) for word in words])

# Get all the negative and positive ID's from the files within moview-review corpus of NLTK
positive_ids = movie_reviews.fileids('pos')
negative_ids = movie_reviews.fileids('neg')

# Get the negative and positive word features 
positive_features = [(word_feature(movie_reviews.words(fileids=[i])), 'pos') for i in positive_ids]
negative_features = [(word_feature(movie_reviews.words(fileids=[i])), 'neg') for i in negative_ids]

# Figure out the number of negative and positive values which will be used for training/testing set
negative_values = len(negative_features)*1/2
positive_values = len(positive_features)*1/2


# Create training and testing set with mix and match with positive and negative value
training_set = negative_features[:int(negative_values)] + positive_features[:int(positive_values)]
testing_set = negative_features[int(negative_values):] + positive_features[int(positive_values):]

print ("Training the algorithm on %d ", len(training_set) , " Records")
print ("Testing the algorithm on %d ", len(testing_set) , " Records") 

classifier = NaiveBayesClassifier.train(training_set)
print ('The testing set accuracy is:', nltk.classify.util.accuracy(classifier, testing_set))
classifier.show_most_informative_features()
