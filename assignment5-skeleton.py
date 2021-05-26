import pandas as pd
import string
import os
import sys

# sklearn is installed via pip: pip install -U scikit-learn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import matplotlib.pyplot as plt

# TODO: Your custom imports here; or copy the functions to here manually.
from evaluation import computeAccuracy, computePrecisionRecall
from assignment3_olzama import predictSimplistic

# TODO: You may need to modify assignment 4 if you just had a main() there.
# my_naive_bayes() should take a string (column name) as input and return as output 10 floats (numbers)
# representing the metrics.
from olzama_assignment4 import my_naive_bayes


# From Assignment 2, copied manually here just to remind you
# that you can copy stuff manually if importing isn't working out.
# You can just use this or you can replace it with your function.
def countTokens(text):
    token_counts = {}
    tokens = text.split(' ')
    for word in tokens:
        if not word in token_counts:
            token_counts[word] = 0
        token_counts[word] += 1
    return token_counts


def largest_counts(data):  # TODO: Finish implementing this function

    # TODO: Cut up the rows in the dataset according to how you stored things.
    # The below assumes test data is stored first and negative is stored before positive.
    # If you did the same, no change is required.
    neg_test_data = data[:12500]
    neg_train_data = data[25000:37500]
    pos_test_data = data[12500:25000]
    pos_train_data = data[37500:50000]

    # TODO: SORT the count dicts which countTokens() returns
    # by value (count) in reverse (descending) order.
    # It is your task to Google and learn how to do this, but we will help of course,
    # if you come to use with questions. This can be daunting at first, but give it time.
    # Spend some (reasonable) time across a few days if necessary, and you will do it!

    # As is, the counts returned by the counter AREN'T sorted!
    # So you won't be able to easily retrieve the most frequent words.

    # NB: str.cat() turns whole column into one text
    train_counts_pos_original = countTokens(pos_train_data["review"].str.cat())
    train_counts_pos_cleaned = countTokens(
        pos_train_data["cleaned_review"].str.cat())
    train_counts_pos_lowercased = countTokens(
        pos_train_data["lowercased"].str.cat())
    train_counts_pos_no_stop = countTokens(
        pos_train_data["no stopwords"].str.cat())
    train_counts_pos_lemmatized = countTokens(
        pos_train_data["lemmatized"].str.cat())

    # Once the dicts are sorted, output the first 20 rows for each.
    # This is already done below, no change needed.
    with open('counts.txt', 'w') as f:
        f.write('Original POS reviews:\n')
        for k, v in train_counts_pos_original[:19]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Cleaned POS reviews:\n')
        for k, v in train_counts_pos_cleaned[:19]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Lowercased POS reviews:\n')
        for k, v in train_counts_pos_lowercased[:19]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('No stopwords POS reviews:\n')
        for k, v in train_counts_pos_no_stop[:19]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Lemmatized POS reviews:\n')
        for k, v in train_counts_pos_lemmatized[:19]:
            f.write('{}\t{}\n'.format(k, v))
        # TODO: Do the same for all the remaining training dicts, per Assignment spec.
            
            
    # TODO: Copy the output of the above print statements
    #  into your document/report, or otherwise create a table/visualization for these counts.
    # Manually is fine, or you may explore bar charts in pandas! Be creative :).


def main(argv):
    data = pd.read_csv('my_imdb_expanded.csv', index_col=[0])
    # print(data.head())  # <- Verify the format. Comment this back out once done.

    # TODO: Comment the call to largest_counts() out once done with Part 1.
    # Sorting is kind of slow so you don't want to do it again and again.
    largest_counts(data)

    # Part II:
    # Run all models and store the results in variables (dicts).
    # TODO: Make sure you imported your own naive bayes function and it works properly with a named column input!
    # TODO: See also the next todo which gives an example of a convenient output for my_naive_bayes()
    # which you can then easily use to collect different scores.
    # For example (and as illustrated below), the models (nb_original, nb_cleaned, etc.) can be not just lists of scores
    # but dicts where each score will be stored by key, like [TEST][POS][RECALL], etc.
    # But you can also just use lists, except then you must not make a mistake, which score you are accessing,
    # when you plot graphs.
    nb_original = my_naive_bayes(data['review'])
    nb_cleaned = my_naive_bayes(data['cleaned_review'])
    nb_lowercase = my_naive_bayes(data['lowercased'])
    nb_no_stop = my_naive_bayes(data['no stopwords'])
    nb_lemmatized = my_naive_bayes(data['lemmatized'])

    # Collect accuracies and other scores across models.
    # TODO: Harmonize this with your own naive_bayes() function!
    # The below assumes that naive_bayes() returns a fairly complex dict of scores.
    # (NB: The dicts there contain other dicts!)
    # The return statement for that function looks like this:
    # return({'TRAIN': {'accuracy': accuracy_train, 'POS': {'precision': precision_pos_train, 'recall': recall_pos_train}, 'NEG': {'precision': precision_neg_train, 'recall': recall_neg_train}}, 'TEST': {'accuracy': accuracy_test, 'POS': {'precision': precision_pos_test, 'recall': recall_pos_test}, 'NEG': {'precision': precision_neg_test, 'recall': recall_neg_test}}})
    # This of course assumes that variables like "accuracy_train", etc., were assigned the right values already.
    # You don't have to do it this way; we are giving it to you just as an example.
    train_accuracies = []
    test_accuracies = []
    # TODO: Initialize other score lists similarly. The precision and recalls, for negative and positive, train and test.
    for model in [nb_original, nb_cleaned, nb_lowercase, nb_no_stop, nb_lemmatized]:
        # TODO: See comment above about where this "model" dict comes from.
        # If you are doing something different, e.g. just a list of scores,
        # that's fine, change the below as appropriate,
        # just make sure you don't confuse where which score is.
        train_accuracies.append(model['TRAIN']['accuracy'])
        test_accuracies.append(model['TEST']['accuracy'])
        # TODO: Collect other scores similarly. The precision and recalls, for negative and positive, train and test.

    # TODO: Create the plot(s) that you want for the report using matplotlib (plt).
    # Use the below to save pictures as files:
    # plt.savefig('filename.png')


if __name__ == "__main__":
    main(sys.argv)
