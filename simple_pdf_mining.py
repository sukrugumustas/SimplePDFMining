#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hilal Balcı, 150114047
# Atakan Ülgen, 150115066
# Şükrü Gümüştaş, 150114032
from __future__ import division

import io
import re
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from wordcloud import WordCloud
from stopwords_lib import wide_range_stopwords
from tf_idf_lib import tf_word_list_creator, tf_idf_word_list_creator


def convert(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = io.StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)
    infile = open(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close()
    return text


def splitwords(documents):
    cleandoc = []
    for each_document in documents:
        each_document = cleaneachdocument(each_document)
        cleandoc.append(each_document)
    return cleandoc


def cleaneachdocument(document_to_clean):
    document_to_clean = document_to_clean.lower()
    document_to_clean = re.sub(r"[^\w\s]", ' ', document_to_clean)
    document_to_clean = re.sub(r"(^|\W)\d+($|\W)", ' ', document_to_clean)
    document_to_clean = re.sub(r"[0-9]+", ' ', document_to_clean)
    document_to_clean = re.sub(r"[\s+]", ' ', document_to_clean).strip()
    document_to_clean = nltk.word_tokenize(document_to_clean)
    document_to_clean = [word for word in document_to_clean if word not in wide_range_stopwords]
    return document_to_clean


def csvfile(all_documents, string):
    length = len(all_documents)
    i = 0
    while i < length:
        if i == length - 1:
            filename = "all_documents" + string + "_list.csv"
        else:
            filename = "document_" + str(i + 1) + string + "_list.csv"
        file = open(filename, 'w', encoding='utf-8')
        for peer in all_documents[i]:
            file.write("%s,%.16f\n" % (peer[0], peer[1]))
        file.close()
        i += 1
    return


def wordcloudfile(all_documents, string):
    length = len(all_documents)
    i = 0
    while i < length:
        if i == length - 1:
            filename = "all_documents" + string + "_WordCloud.pdf"
        else:
            filename = "document_" + str(i + 1) + string + "_WordCloud.pdf"
        d = {}
        for a, x in all_documents[i]:
            d[a] = float(x)
        wordcloud = WordCloud(width=2000, height=1000, background_color="white", relative_scaling=1.0,
                              stopwords={'to', 'of'}).generate_from_frequencies(d)
        plt.figure(figsize=(20, 10), facecolor='k')
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig(filename, facecolor='k', bbox_inches='tight')
        i += 1
    return


def main():
    doc1 = convert("inputs/doc1.pdf")
    doc2 = convert("inputs/doc2.pdf")
    doc3 = convert("inputs/doc3.pdf")
    doc4 = convert("inputs/doc4.pdf")
    doc5 = convert("inputs/doc5.pdf")
    doc6 = convert("inputs/doc6.pdf")
    doc7 = convert("inputs/doc7.pdf")
    doc8 = convert("inputs/doc8.pdf")

    alldoc = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8]

    cleanwords = splitwords(alldoc)

    tf_words_all_documents = []

    tf_idf_words_all_documents = []
    # Here we are normalizing our words to calculate values for all of them at once.
    allinone = []
    for each in cleanwords:
        allinone += each

    cleanwords.append(allinone)

    #   We are calculating all tf and tf-idf values for each unique word in each file.

    for document in cleanwords:
        tf_words_all_documents.append(tf_word_list_creator(document))
        tf_idf_words_all_documents.append(tf_idf_word_list_creator(document, cleanwords))

    #   After that we are creating output files seperately.

    csvfile(tf_words_all_documents, "_tf")
    wordcloudfile(tf_words_all_documents, "_tf")

    csvfile(tf_idf_words_all_documents, "_tfidf")
    wordcloudfile(tf_idf_words_all_documents, "_tfidf")
    return


if __name__ == '__main__':
    main()
