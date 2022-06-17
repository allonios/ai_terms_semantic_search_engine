import spacy
from spacy import displacy
from collections import Counter
import en_core_web_lg

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

nlp = en_core_web_lg.load()

question1 = """
what is the long term short memory input?
"""

question2 = """
what is the long term short memory input?
"""

# spacy
doc = nlp(question1)

# nltk
tokenized_question1 = nltk.word_tokenize(question1)
named_question1 = nltk.pos_tag(tokenized_question1)

tokenized_question2 = nltk.word_tokenize(question2)
named_question2 = nltk.pos_tag(tokenized_question2)
