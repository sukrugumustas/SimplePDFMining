#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#   Marmara University - Computer Engineering Department
#   Simple PDF Mining Project
#   Hilal Balcı, 150114047
#   Atakan Ülgen, 150115066
#   Şükrü Gümüştaş, 150114032

import math
from collections import Counter


#   In this function we are calculating tf values for each document


def tf_word_list_creator(document):
    #   This line deletes words occured more than once, aka getting unique words
    return_list = list(set(document))
    #   This line counts the maximum term in document
    maximum = Counter(document).most_common(1)[0][1]
    i = 0
    #   From Wikipedia, we used tf formula as (count of word in that document/maximum occuring term number)
    while i < len(return_list):
        return_list[i] = [return_list[i], document.count(return_list[i]) / maximum]
        i += 1
    #   We are sorting our words
    return_list.sort(key=lambda x: x[1], reverse=True)
    #   We are returning the 50 greatest word with tf values
    return return_list[:50]


#   This function calculates idf values.
#   Again from Wikipedia, idf is equal to log(1 + (number of documents/number of documents containing that term))


def idfvalue(word, documents):
    total_count = 0
    for i in documents:
        if word in i:
            total_count += 1
    return math.log(1 + (len(documents) / total_count))


#   In this function we are multiplying tf and idf values for each word in each document


def tf_idf_word_list_creator(document, documents):
    return_list = list(set(document))
    maximum = Counter(document).most_common(1)[0][1]
    i = 0
    while i < len(return_list):
        return_list[i] = [return_list[i],
                          (document.count(return_list[i]) / maximum) * idfvalue(return_list[i], documents[:-1])]
        i += 1
    return_list.sort(key=lambda x: x[1], reverse=True)
    return return_list[:50]
