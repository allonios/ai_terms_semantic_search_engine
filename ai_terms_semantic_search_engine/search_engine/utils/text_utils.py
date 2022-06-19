import re

import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from numba import jit


def names_preprocessing(name):
    return re.sub(r"[^a-zA-Z ] ", "", name)


@jit(nopython=True)
def edit_distance(s1, s2):
    m = len(s1) + 1
    n = len(s2) + 1

    tbl = {}
    for i in range(m):
        tbl[i, 0] = i
    for j in range(n):
        tbl[0, j] = j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            tbl[i, j] = min(
                tbl[i, j - 1] + 1, tbl[i - 1, j] + 1, tbl[i - 1, j - 1] + cost
            )

    length = len(s1) + len(s2) / 2

    return tbl[i, j] / length


def is_abbrev(abbrev, text):
    abbrev = abbrev.lower()
    text = text.lower()
    words = text.split()
    if not abbrev:
        return True
    if abbrev and not text:
        return False
    if abbrev[0] != text[0]:
        return False
    else:
        return is_abbrev(abbrev[1:], " ".join(words[1:])) or any(
            is_abbrev(abbrev[1:], text[i + 1 :]) for i in range(len(words[0]))
        )


def get_words_sim(w1, w2):
    # print("compairing ", w1, w2)
    if w1 == w2:
        # print("Equals")
        return 1.0
    # syn1 = wordnet.synsets(w1)
    # syn2 = wordnet.synsets(w2)
    return 1 - edit_distance(w1, w2)
    # if len(syn1) == 0 or len(syn2) == 0:
    #     pass
    # print("Edit_distance", w1, w2)
    # else:
    #     # print("WIP_Similarity", w1, w2)
    #     return max(s1.wup_similarity(s2) for s1 in syn1 for s2 in syn2)


def get_abbreviation_indices(w, tokens, index):
    for i in range(len(tokens)):
        if is_abbrev(w, " ".join(tokens[: i + 1])):
            return index, index + i


def fast_calc_similarity(name1_tokens, name2_tokens, syns1, syns2):
    sim_matrix = np.array(np.zeros(len(name1_tokens)))

    for i1, w1 in enumerate(name1_tokens):
        for _i2, _w2 in enumerate(name2_tokens):
            if is_abbrev(w1, " ".join(name2_tokens[i1:])):
                # print(i1)
                sim_matrix[i1] = 1
                break

    for _i2, w2 in enumerate(name2_tokens):
        for i1, _w1 in enumerate(name1_tokens):
            if is_abbrev(w2, " ".join(name1_tokens[i1:])):
                start, end = get_abbreviation_indices(
                    w2, name1_tokens[i1:], i1
                )
                sim_matrix[start:end] = 1
                break

    for i1, w1 in enumerate(name1_tokens):
        if sim_matrix[i1] == 0:
            for w2 in name2_tokens:
                sim_matrix[i1] = max(get_words_sim(w1, w2), sim_matrix[i1])

    return sim_matrix

def camel_case_split(str):
    words = [[str[0]]]
  
    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)
  
    return " ".join([''.join(word) for word in words])


def calc_similarity(name1, name2):
    name1 = camel_case_split(name1)
    name2 = camel_case_split(name2)
    
    name1 = names_preprocessing(name1)
    name2 = names_preprocessing(name2)

    name1_tokens = word_tokenize(name1)
    name2_tokens = word_tokenize(name2)

    name1_tokens = [
        word for word in name1_tokens if word not in stopwords.words("english")
    ]
    name2_tokens = [
        word for word in name2_tokens if word not in stopwords.words("english")
    ]

    # print(name1_tokens)

    syns1 = [wordnet.synsets(w1) for w1 in name1_tokens]
    syns2 = [wordnet.synsets(w2) for w2 in name2_tokens]

    simliarity_matrix = fast_calc_similarity(
        name1_tokens, name2_tokens, syns1, syns2
    )

    # print(simliarity_matrix)

    if not simliarity_matrix.size:
        return 0

    return simliarity_matrix.mean()
