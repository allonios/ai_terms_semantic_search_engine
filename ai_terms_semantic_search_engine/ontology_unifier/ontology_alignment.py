from owlready2 import *
from owlready2.prop import DataPropertyClass, ObjectPropertyClass, DataProperty
from owlready2.annotation import AnnotationPropertyClass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from owlready2.entity import ThingClass
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from functools import reduce
from numba import jit
import numpy as np
import math
import re


epsilon_1 = 0.9
epsilon_2 = 0.4
epsilon_3 = 0.25


def names_preprocessing(name):
    regex = re.compile(r"[^a-zA-Z ] ")
    name = regex.sub('', name)
    return name    
    
def Meta(e):
    name = names_preprocessing(e.name)
    return word_tokenize(name) 

def Hier(e):
    return list(e.subclasses())

def Inst(e):
    return list(e.instances())

def Rest(c):
    return list(c.get_class_properties())

def Doma(p):
    return list(p.domain)

def Rang(p):
    return list(filter(lambda c: isinstance(c, ThingClass),p.range))

def Sibl(e, onto):
    sibls = []
    if isinstance(e, ThingClass):
        entities = list(onto.classes())
    if isinstance(e, ObjectPropertyClass) or isinstance(e, DataPropertyClass):
        entities = list(onto.properties())
    
    parents = onto.get_parents_of(e)
    for e1 in entities:
        if e1 == e:
            continue
        e1_parents = onto.get_parents_of(e1)
        if(len(list(set(e1_parents) & set(parents)))):
            sibls.append(e1)
    
    return sibls

def get_length_from_root(onto, c, length=0):
    parents = onto.get_parents_of(c)
    if(len(parents) == 1):
        return get_length_from_root(onto, parents[0], length + 1)
    return length


def get_classes_names(onto):
    return list(map(lambda c: c.name, onto.classes()))


def get_properties_names(onto):
    return list(map(lambda c: c.name, onto.properties()))


def FLS(onto1, onto2):
    classes_names_1 = get_classes_names(onto1)
    classes_names_2 = get_classes_names(onto2)
    properties_names_1 = get_properties_names(onto1)
    properties_names_2 = get_properties_names(onto2)

    iden_conc_label = len(set(classes_names_1) & set(classes_names_2))
    iden_prop_label = len(set(properties_names_1) & set(properties_names_2))

    return (
        (iden_conc_label + iden_prop_label)
        / max(
            len(classes_names_1) + len(properties_names_1),
            len(classes_names_2) + len(properties_names_2)
        )
    )

def get_classes_with_sub_classes(onto):
    return list(filter(lambda c: len(list(c.subclasses())), onto.classes()))


def get_properties_with_sub_properties(onto):
    return list(filter(lambda c: len(list(c.subclasses())), onto.properties()))


def get_number_of_common_nonl_classes(onto1, onto2):
    nonl_classs_1 = get_classes_with_sub_classes(onto1)
    nonl_classs_2 = get_classes_with_sub_classes(onto2)
    _classes_1 = list(map(
        lambda c: f'{len(list(c.subclasses()))}-{get_length_from_root(onto1, c)}', nonl_classs_1))

    _classes_2 = list(map(
        lambda c: f'{len(list(c.subclasses()))}-{get_length_from_root(onto2, c)}', nonl_classs_2))

    return len(list(filter(lambda x: x in _classes_1, _classes_2))), len(nonl_classs_1), len(nonl_classs_2)


def get_number_of_common_nonl_properties(onto1, onto2):
    nonl_property_1 = get_properties_with_sub_properties(onto1)
    nonl_property_2 = get_properties_with_sub_properties(onto2)
    _properties_1 = list(map(
        lambda p: f'{len(list(p.subclasses()))}-{get_length_from_root(onto1, p)}', nonl_property_1))

    _properties_2 = list(map(
        lambda p: f'{len(list(p.subclasses()))}-{get_length_from_root(onto2, p)}', nonl_property_2))

    return len(list(filter(lambda x: x in _properties_1, _properties_2))), len(nonl_property_1), len(nonl_property_2)


def FSS(onto1, onto2):
    common_nonl_classes, nonl_classs_1_len, nonl_classes_2_len = get_number_of_common_nonl_classes(
        onto1, onto2)

    common_nonl_properties, nonl_properties_1_len, nonl_properties_2_len = get_number_of_common_nonl_properties(
        onto1, onto2)

    maximum = max(
        (nonl_classs_1_len + nonl_properties_1_len),
        (nonl_classes_2_len + nonl_properties_2_len)
    )

    if maximum == 0:
        return 0

    return (common_nonl_classes + common_nonl_properties) / maximum

@jit(nopython=True)
def edit_distance(s1, s2):
    m=len(s1)+1
    n=len(s2)+1

    tbl = {}
    for i in np.arange(m): tbl[i,0]=i
    for j in np.arange(n): tbl[0,j]=j
    for i in np.arange(1, m):
        for j in np.arange(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    length = (len(s1) + len(s2) / 2)
    
    return tbl[i,j]/length

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
        return (is_abbrev(abbrev[1:], ' '.join(words[1:])) or
                any(is_abbrev(abbrev[1:], text[i + 1:])
                    for i in range(len(words[0]))))

        
def get_words_sim(w1, w2):
    if (w1 == w2):
        return 1.
    syn1 = wordnet.synsets(w1)
    syn2 = wordnet.synsets(w2)
    if len(syn1) == 0 or len(syn2) == 0:
        return 1 - edit_distance(w1, w2)
    else:
        return max(s1.wup_similarity(s2) for s1 in syn1 for s2 in syn2)



def get_abbreviation_indices(w, tokens, index):
    for i in range(len(tokens)):
        if is_abbrev(w, " ".join(tokens[:i+1])):
            return index, index + i
     
     
def fast_calc_similarity(name1_tokens, name2_tokens):
    sim_matrix = np.array(np.zeros(len(name1_tokens)))

    for i1, w1 in enumerate(name1_tokens):
        for _, w2 in enumerate(name2_tokens):
            if is_abbrev(w1, " ".join(name2_tokens[i1:])):
                sim_matrix[i1] = 1
                break

    for _, w2 in enumerate(name2_tokens):
        for i1, w1 in enumerate(name1_tokens):
            if is_abbrev(w2, " ".join(name1_tokens[i1:])):
                start, end = get_abbreviation_indices(
                    w2, name1_tokens[i1:], i1
                )
                sim_matrix[start:end] = 1
                break

    for i1, w1 in enumerate(name1_tokens):
        if (sim_matrix[i1] == 0):
            for w2 in name2_tokens:
                sim_matrix[i1] = max(get_words_sim(w1, w2), sim_matrix[i1])

    return sim_matrix


def calc_similarity(name1, name2):
    name1 = names_preprocessing(name1)
    name2 = names_preprocessing(name2)

    name1_tokens = word_tokenize(name1)
    name2_tokens = word_tokenize(name2)

    name1_tokens = [word for word in name1_tokens if
                    word not in stopwords.words('english')]
    name2_tokens = [word for word in name2_tokens if
                    word not in stopwords.words('english')]

    simliarity_matrix = fast_calc_similarity(
        name1_tokens, name2_tokens
    )
    return simliarity_matrix.mean()

def sim_name(e1, e2):
    return calc_similarity(e1.name, e2.name)


def calculate_concept_matrix(concept, use_hier_features=True):
    results = []
    results += Meta(concept)
    if use_hier_features:
        results += list(reduce(lambda acc, c: acc + Meta(c),Hier(concept), []))  
    results += list(reduce(lambda acc, c: acc + Meta(c),Inst(concept), []))
    results += list(reduce(lambda acc, c: acc + Meta(c),Rest(concept), []))  
    return results 

def calculate_property_matrix(property, use_hier_features=True):
    results = []
    results += Meta(property)
    if use_hier_features:
        results += list(reduce(lambda acc, p: acc + Meta(p),Hier(property), []))  
    # inst_mata = list(reduce(lambda acc, p: acc + Meta(p),Inst(property), []))
    results += list(reduce(lambda acc, p: acc + Meta(p),Doma(property), []))  
    results += list(reduce(lambda acc, p: acc + Meta(p),Rang(property), []))  
    return results  



def sim_vec(onto1, onto2, e1, e2, use_hier_features = True, add_more_features = False):
    if isinstance(e1, ThingClass):
        e1_matrix = calculate_concept_matrix(e1, use_hier_features)
    elif isinstance(e1, ObjectPropertyClass) or isinstance(e1, DataPropertyClass) or isinstance(e1, AnnotationPropertyClass):  
        e1_matrix = calculate_property_matrix(e1, use_hier_features)
    if isinstance(e2, ThingClass):
        e2_matrix = calculate_concept_matrix(e2, use_hier_features)
    elif isinstance(e2, ObjectPropertyClass) or isinstance(e2, DataPropertyClass) or isinstance(e2, AnnotationPropertyClass): 
        e2_matrix = calculate_property_matrix(e2, use_hier_features)

    vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
    vectorizer.fit(set(e1_matrix + e2_matrix)) 
    vec1 = vectorizer.transform([" ".join(e1_matrix)]).toarray()
    vec2 = vectorizer.transform([" ".join(e2_matrix)]).toarray()
    
    if add_more_features:
        vec1 = np.append(vec1, np.array([[get_length_from_root(onto1, e1), len(Rest(e1)), len(Hier(e1))]]),axis=1)
        vec2 = np.append(vec2, np.array([[get_length_from_root(onto2, e2), len(Rest(e2)), len(Hier(e2))]]),axis=1)
    
    return cosine_similarity(vec1, vec2)[0][0]

labels = {
    "HasSubConcept": 1,
    "HasProperty": 2,
    "HasConceptSibling": 3,
    "HasSubProperty": 4,
    "HasRange": 5,
    "HasPropertySibling": 6
}

def NO_O(onto):
    return list(onto.classes()) + list(onto.properties())
    
def DLG_O_for_property(p: DataPropertyClass, onto):
    triples = []
    triples = triples + list(map(lambda sp: (p, labels["HasSubProperty"], sp),Hier(p)))
    triples = triples + list(map(lambda sp: (p, labels["HasRange"], sp),Rang(p)))
    triples = triples + list(map(lambda sc: (p, labels["HasPropertySibling"], sc),Sibl(p, onto)))
    return triples

def DLG_O_for_concept(c: ThingClass, onto):
    triples = []
    triples = triples + list(map(lambda sc: (c, labels["HasSubConcept"], sc),Hier(c)))
    triples = triples + list(map(lambda p: (c, labels["HasProperty"], p),Rest(c)))
    triples = triples + list(map(lambda sc: (c, labels["HasConceptSibling"], sc),Sibl(c, onto)))
    return triples
    
def DLG_O(onto):
    triples = []
    for e in NO_O(onto):
        if isinstance(e, ObjectPropertyClass) or isinstance(e, DataPropertyClass):
            triples = triples +  DLG_O_for_property(e, onto)
        if isinstance(e, ThingClass):
            triples = triples +  DLG_O_for_concept(e, onto)
    
    return triples
        
        
def PCG_O(onto_1, onto_2):
    triples_1 = DLG_O(onto_1)
    triples_2 = DLG_O(onto_2)
    
    pcg = []
    
    for triple_1 in triples_1:
        for triple_2 in triples_2:
            if(triple_1[1] == triple_2[1]):
                pcg.append(((triple_1[0], triple_2[0]), triple_1[1], (triple_1[2], triple_2[2])))     
                
    return pcg


def calc_sigma():
    ...

def SPG_O(onto_1, onto_2):
    pcg = PCG_O(onto_1, onto_2)
    
    grouped_by_left = {}
    
    for triples in pcg:
        left = grouped_by_left.get(triples[0], False)
        if not left:
            names_sim = calc_similarity(triples[0][0].name, triples[0][1].name)
            grouped_by_left[triples[0]] = {"sigma": names_sim, "sigma_0": names_sim, "connections": []}
        grouped_by_left[triples[0]]["connections"].append((triples[1], triples[2]))
    
    return grouped_by_left


def calc_phi(rel, spg, propagation_only_on_CP= False):
    result = 0
    for rel_type, out_rel in rel["connections"]:
        if propagation_only_on_CP and rel_type != labels["HasRange"] and rel_type != labels["HasProperty"]:
           continue 
        result += (1/len(rel["connections"])) * spg.get(out_rel, {"sigma": 0})["sigma"]
    
    return result

def update_SPG_O(spg, propagation_only_on_CP = False):
    max_sigma = 0
    for _, value in spg.items():
        value["sigma"] = value["sigma_0"] + value["sigma"] + calc_phi(value, spg, propagation_only_on_CP)
        if value["sigma"] > max_sigma:
            max_sigma = value["sigma"]
    
    for _, value in spg.items():
        value["sigma"] = value["sigma"] / max_sigma
    
    return spg
    
def get_structural_matched_entities(onto1, onto2, threshold=0.8, iterations=1000):
    result = {}
    spg = SPG_O(onto1,onto2)
    for _ in range(iterations):
        spg = update_SPG_O(spg, False)
    
    for k, v in spg.items():
        if v["sigma"] > 0.8:
            result[k] = v["sigma"]

    return result    



def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sim_combination(onto1, onto2, e1, e2):
    fls = FLS(onto1, onto2)
    fss = FSS(onto1, onto2)
    use_hier_features = True
    add_more_features_to_VD = False
    if fss > epsilon_1:
        use_hier_features = True
        if fls < epsilon_2:
            add_more_features_to_VD = True
    
    svec = sim_vec(onto1, onto2, e1, e2, use_hier_features, add_more_features_to_VD)
    sname = sim_name(e1 ,e2)
    
    wname = fls/max(fls,fss)
    wvec = fss/max(fls,fss)
    
    return  (wname * sname + wvec * svec) / (wname + wvec) 
        
    
def get_textual_matched_entities(onto1, onto2, threshold=0.8):
    entities_1 = list(onto1.classes()) + list(onto1.properties())
    entities_2 = list(onto2.classes()) + list(onto2.properties())
    results = {}
    
    matchings_1to2 = {}
    max_1to2 = defaultdict(int)
    
    matchings_2to1 = {}
    max_2to1 = defaultdict(int)
    
    for e1 in entities_1:
        for e2 in entities_2:
            sim = sim_combination(onto1, onto2, e1, e2)
            if  sim > threshold:
                if max_1to2[e1] < sim:
                    max_1to2[e1] = sim
                    matchings_1to2[e1] = e2
                if max_2to1[e2] < sim:
                    max_2to1[e2] = sim
                    matchings_2to1[e2] = e1
                    
    for e1, e2 in matchings_1to2.items():
        if matchings_2to1[e2] != e1 and max_2to1[e2] > max_1to2[e1]:
            results[(matchings_2to1[e2],e2)] = max_2to1[e2]
        else: results[(e1,e2)] = max_1to2[e1]
                    
    return results


def get_hybred_matched_entities(onto1, onto2):
    fls = FLS(onto1, onto2)
    fss = FSS(onto1, onto2)
    fls = fls / (max(fls,fss) + 0.0000001)
    fss = fss / (max(fls,fss) + 0.0000001)
    
    textual_matchings = defaultdict(int)
    structural_matchings = defaultdict(int)
    
    matchings_1to2 = {}
    matchings_2to1 = {}
    
    if fls != 0:
        textual_matchings = defaultdict(int,get_textual_matched_entities(onto1, onto2))
    
    if fss != 0:
        structural_matchings = defaultdict(int,get_structural_matched_entities(onto1, onto2))
    
    all_keys = set(list(textual_matchings.keys()) + list(structural_matchings.keys()))

    results = {}
    
    matchings_1to2 = {}
    max_1to2 = defaultdict(lambda:0)
    
    matchings_2to1 = {}
    max_2to1 = defaultdict(lambda:0)
    
    for e1, e2 in all_keys:
        sim = textual_matchings[(e1,e2)] * fls + structural_matchings[(e1,e2)] * fss
        if max_1to2[e1] < sim:
            max_1to2[e1] = sim
            matchings_1to2[e1] = e2
        if max_2to1[e2] < sim:
            max_2to1[e2] = sim
            matchings_2to1[e2] = e1
                
    for e1, e2 in matchings_1to2.items():
        if matchings_2to1[e2] != e1 and max_2to1[e2] > max_1to2[e1]:
            results[(matchings_2to1[e2],e2)] = max_2to1[e2]
        else: results[(e1,e2)] = max_1to2[e1]
        

    return results
    
    
    
def align_ontologies(onto1_path, onto2_path, file_path):
    world = World()
    onto1 = world.get_ontology(onto1_path).load()
    onto2 = world.get_ontology(onto2_path).load()
    matchings = get_hybred_matched_entities(onto1, onto2)
    for e1, e2 in matchings.keys():
        e1.equivalent_to.append(e2)
    world.save(file_path)
    new_ontology = get_ontology(file_path).load()
    new_ontology.save(file_path)
    world.close()
    
    
    