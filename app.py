from __future__ import unicode_literals
from urllib.request import urlopen
from bs4 import BeautifulSoup
from flask import Flask, render_template, url_for, request
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk_summarization import nltk_summarizer
from sumy_summarization import sumy_summary
from string import punctuation
from heapq import nlargest
import time
from transformers import pipeline
import os
import collections
collections.Callable = collections.abc.Callable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
nlp = spacy.load('en_core_web_sm')
app = Flask(__name__)


def readingTime(mytext):
    total_words = len([token.text for token in nlp(mytext)])
    estimatedTime = total_words/200.0
    return estimatedTime


def text_summarizer(raw_docx):
    raw_text = raw_docx
    docx = nlp(raw_text.lower())

    stopwords = list(STOP_WORDS)

    word_frequencies = {}
    for word in docx:
        if word.text not in stopwords:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    # Sentence Tokens
    sentence_list = [sentence for sentence in docx.sents]

    # Sentence Scores
    sentence_scores = {}
    for sent in sentence_list:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    select_length = int(len(sentence_list)*0.2)
    summarized_sentences = nlargest(
        select_length, sentence_scores, key=sentence_scores.get)
    final_sentences = [w.text for w in summarized_sentences]
    summary = ' '.join(final_sentences)
    return summary

# Fetch Text From Url


def get_text(url):
    page = urlopen(url)
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return fetched_text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        lenraw = len(rawtext)
        final_reading_time = readingTime(rawtext)
        final_summary = text_summarizer(rawtext)

        lenfinal = len(final_summary)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end-start
    return render_template('index.html', ctext=rawtext, lenraw=lenraw, lenfinal=lenfinal, final_summary=final_summary.capitalize(), final_time=final_time, final_reading_time=final_reading_time, summary_reading_time=summary_reading_time)


@app.route('/analyze_url', methods=['GET', 'POST'])
def analyze_url():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        rawtext = get_text(rawtext)
        final_reading_time = readingTime(rawtext)
        final_summary = text_summarizer(rawtext)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end-start
    return render_template('index.html', ctext=rawtext, final_summary=final_summary.capitalize(), final_time=final_time, final_reading_time=final_reading_time, summary_reading_time=summary_reading_time)


@app.route('/abstract', methods=['POST'])
def abstract():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        summarizer = pipeline('summarization', model="lidiya/bart-base-samsum")
        lenraw = len(rawtext)
        final_reading_time = readingTime(rawtext)
        final_summary = text_summarizer(rawtext)
        lenfinal = len(final_summary)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end-start
    final_summary = summarizer(rawtext)[0]['summary_text']
    return render_template('index.html', final_summary=final_summary.capitalize(), ctext=rawtext, lenraw=lenraw, lenfinal=lenfinal, final_time=final_time, final_reading_time=final_reading_time, summary_reading_time=summary_reading_time)


@app.route('/compare_summary')
def compare_summary():
    return render_template('compare_summary.html')


@app.route('/comparer', methods=['GET', 'POST'])
def comparer():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        final_reading_time = readingTime(rawtext)
        final_summary_spacy = text_summarizer(rawtext)
        summary_reading_time = readingTime(final_summary_spacy)

        # NLTK
        final_summary_nltk = nltk_summarizer(rawtext)
        summary_reading_time_nltk = readingTime(final_summary_nltk)
        # Sumy
        final_summary_sumy = sumy_summary(rawtext)
        summary_reading_time_sumy = readingTime(final_summary_sumy)

        end = time.time()
        final_time = end-start
    return render_template('compare_summary.html', ctext=rawtext, final_summary_spacy=final_summary_spacy.capitalize(), final_summary_nltk=final_summary_nltk.capitalize(), final_summary_sumy=final_summary_sumy.capitalize(), final_time=final_time, final_reading_time=final_reading_time, summary_reading_time=summary_reading_time, summary_reading_time_nltk=summary_reading_time_nltk, summary_reading_time_sumy=summary_reading_time_sumy)


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run()
