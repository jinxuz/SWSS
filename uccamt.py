import os
import re
import spacy
import pickle
import string
import pandas as pd
from collections import Counter
from tupa.parse import Parser
from ucca.convert import from_text
from ucca.textutil import extract_terminals

from nltk import word_tokenize, pos_tag
from nltk.translate.bleu_score import sentence_bleu
from nltk.stem import PorterStemmer

import align
from meteor5 import Meteor, nlp
from process import calibrate_ucca_single

PARSER = None
PARSER_PATH = None
PARSED_SENT = {}
PARSED_SENT_PATH = "parsed_sentences.pkl"  # a preprocessed pickle file storing UCCA of sentences

MeteorMetric = None
NLP = {'en': nlp, 'de': None}
os.environ['SPACY_WARNING_IGNORE'] = 'W008'


class NoSentence:
    def __init__(self):
        pass


# get preprocessed UCCA parsings
def get_parsed_sent():
    global PARSED_SENT
    global PARSED_SENT_PATH
    try:
        if PARSED_SENT_PATH is not "" and len(PARSED_SENT) == 0:
            with open(PARSED_SENT_PATH, 'rb') as f:
                PARSED_SENT = pickle.load(f)
    finally:
        return PARSED_SENT


# save preprocessed UCCA parsings
def save_parsed_sent():
    global PARSED_SENT
    global PARSED_SENT_PATH
    try:
        with open(PARSED_SENT_PATH, 'wb') as f:
            pickle.dump(PARSED_SENT, f)
    except FileNotFoundError:
        print("File Not Found in %s" % PARSED_SENT_PATH)


# update preprocessed UCCA parsings
def update_parsed_sent():
    global PARSED_SENT
    global PARSED_SENT_PATH
    with open(PARSED_SENT_PATH, 'wb') as f:
        pickle.dump(PARSED_SENT, f)


def get_parser(model_path):
    global PARSER
    global PARSER_PATH
    if PARSER_PATH is not model_path or PARSER is None:
        PARSER_PATH = model_path
        PARSER = Parser(model_path)
    return PARSER


# get the Spacy model
def get_nlp(lang):
    global NLP
    if NLP[lang] is None:
        if lang == 'de':
            NLP['de'] = spacy.load("de_core_news_md")
    return NLP[lang]

'''
def normalize_sentence(s):
    s = re.sub(r"\W+", r" ", s)
    s = re.sub(r"(\s[a-zA-Z])\s([a-zA-Z]\s)", r"\1\2", s)
    # s = s.lower()
    # s = Meteor._normalize(s)
    s = s.strip()
    return s
'''
def normalize_sentence(s, filter_comma=False, lang='en'):
    """
    The function to normalize the punctuations of sentences.
    When used in UCCA, filter_comma is set to False,
    as commas will be reserved before UCCA parsing to improve its performance and tackled with later.
    When used in Meteor or sth else, no punctuations will be reserved.
    """
    sentence = re.sub(r"[+\!:\/_;$%^*(+\")-]+|[+——()?【】“”！，。？；、~@#￥%……&*（）]+", "", s.strip())
    if filter_comma:
        sentence = re.sub(r",", "", sentence)
    _nlp = get_nlp(lang)
    token_list = [i.text for i in _nlp(sentence)]
    if '.' in token_list:
        token_list.remove('.')
    return ' '.join(token_list)


# UCCA parsing sentences
def ucca_parse_sentences(sentences, model_path='models/ucca-bilstm', model=None, lang='en', to_save=True):
    get_parsed_sent()
    sentences = [normalize_sentence(sentence, lang=lang) for sentence in sentences]
    to_parse = []
    for i in range(len(sentences)):  # check the preprocess pickle file to see if any update is needed
        if sentences[i] in PARSED_SENT:
            sentences[i] = PARSED_SENT[sentences[i]]
        elif len(sentences[i].strip()) == 0:
            sentences[i] = NoSentence()
        else:
            to_parse.append((i, sentences[i]))
    if len(to_parse) > 0:
        print("Parsing", len(to_parse), "sentences.", len(sentences) - len(to_parse), "sentences already parsed.")
        if model is None:
            parser = get_parser(model_path)
        else:
            parser = model
        ids, text = zip(*to_parse)
        text = list(from_text(text, split=True, one_per_line=True, lang=lang))
        for i, (passage, *_) in enumerate(parser.parse(text)):
            PARSED_SENT[sentences[ids[i]]] = passage
            sentences[ids[i]] = passage
        if to_save:
            save_parsed_sent()
    return sentences


def save_normalized(files, out):
    d = {}
    for file in files:
        with open (file, 'r', encoding='utf8') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines[:560]]
        for line in lines:
            d[line] = normalize_sentence(line)
    print(len(d))
    with open(out, 'wb') as f:
        pickle.dump(d, f)


def _usim(source_sentence, target_sentence, source_passage, target_passage):
    # print(source_sentence.strip())
    # print(target_sentence.strip())
    if align.regularize_word(source_sentence) == "":
        if align.regularize_word(target_sentence) == "":
            return 1
        else:
            return 0
    elif align.regularize_word(target_sentence) == "":
        return 0
    return align.fully_aligned_distance(source_passage, target_passage)


def usim(source_sentences, target_sentences, model='models/ucca-bilstm'):
    if type(source_sentences) is not list:
        source_passages = ucca_parse_sentences([source_sentences], model)[0]
        target_passages = ucca_parse_sentences([target_sentences], model)[0]
        return _usim(source_sentences, target_sentences, source_passages, target_passages)
    source_passages = ucca_parse_sentences(source_sentences, model)
    target_passages = ucca_parse_sentences(target_sentences, model)
    score_list = []
    for i in range(len(source_sentences)):
        score_list.append(_usim(source_sentences[i], target_sentences[i], source_passages[i], target_passages[i]))
    return score_list


def bleu(source_sentences, target_sentences):
    score_list = []
    for i in range(len(source_sentences)):
        source_sentence = normalize_sentence(source_sentences[i], filter_comma=True).lower().strip().split(' ')
        target_sentence = normalize_sentence(target_sentences[i], filter_comma=True).lower().strip().split(' ')
        score_list.append(sentence_bleu([source_sentence], target_sentence))
    return score_list, score_list


# UCCA-MTE
def ucca_mod(reference, candidate, reference_passage=None, candidate_passage=None, pos=False, **kwargs):
    """

    :param reference: reference sentence: string
    :param candidate: candidate sentence: string
    :param reference_passage: UCCA representation of reference sentence
    :param candidate_passage: UCCA representation of candidate sentence
    :param pos: Use POS instead of UCCA to determine core words. default: False
    :param kwargs: kwargs used in calibration(call calibrate_ucca_single),
                   including length_weight, scene_weight, edge_weight and node_weight
    :return: the weighted UCCA-MTE score
    """

    # return weight of a word based on its path tags
    def find_score(core_set: dict, tagchain: list):
        if tagchain[0] not in core_set:
            return 0
        return core_set[tagchain[0]]

    # extract word nodes from UCCA representations
    if reference_passage is None or candidate_passage is None:
        reference_passage, candidate_passage = tuple(ucca_parse_sentences([reference, candidate], 'models/ucca-bilstm'))

    if type(reference_passage) is NoSentence or type(candidate_passage) is NoSentence:
        return 0

    reference_terminals = [node for node in extract_terminals(reference_passage)]
    candidate_terminals = [node for node in extract_terminals(candidate_passage)]

    core_set = {'P': 1, 'S': 1, 'A': 1, 'C': 1}  # semantic role tag set of semantic core words

    # define core POSs
    def good_pos(s: str):
        pos = ['V', 'N', 'PRP', 'WP']
        return any([s.startswith(p) for p in pos])

    # POS tagging
    if pos:
        reference_pos = pos_tag([node.text for node in filter(lambda x: x, reference_terminals)])
        candidate_pos = pos_tag([node.text for node in filter(lambda x: x, candidate_terminals)])
        for i in range(len(reference_terminals)):
            if reference_terminals[i] is None:
                reference_pos.insert(i, ("", ""))
        for i in range(len(candidate_terminals)):
            if candidate_terminals[i] is None:
                candidate_pos.insert(i, ("", ""))

    # find core words
    reference_cores = {}
    for i in range(len(reference_terminals)):
        if reference_terminals[i]:
            tags, parents = align.find_ancester(reference_terminals[i])  # get path tags
            if not pos:
                # determine core by UCCA tags
                if len(set(tags[0][0:1]) - core_set.keys()) == 0:
                    reference_cores[i] = (reference_terminals[i], find_score(core_set, tags[0]), tags, parents)
            else:
                # determine core by POS tags
                if good_pos(reference_pos[i][1]):
                    reference_cores[i] = (reference_terminals[i], 1, tags, parents)
    candidate_cores = {}
    for i in range(len(candidate_terminals)):
        if candidate_terminals[i]:
            tags, parents = align.find_ancester(candidate_terminals[i])
            if not pos:
                if len(set(tags[0][0:1]) - core_set.keys()) == 0:
                    candidate_cores[i] = (candidate_terminals[i], find_score(core_set, tags[0]), tags, parents)
            else:
                if good_pos(candidate_pos[i][1]):
                    candidate_cores[i] = (candidate_terminals[i], 1, tags, parents)

    # get stems of core words
    stemmer = PorterStemmer()
    reference_stems = Counter([stemmer.stem(core[0].text.lower()) for core in reference_cores.values()])
    candidate_stems = Counter([stemmer.stem(core[0].text.lower()) for core in candidate_cores.values()])

    # compute matching proportion
    reference_count = 0
    for k, v in reference_stems.items():
        reference_count += min(v, candidate_stems.get(k, 0))
    reference_core_score = reference_count / max(len(reference_cores), 1)
    candidate_count = 0
    for k, v in candidate_stems.items():
        candidate_count += min(v, reference_stems.get(k, 0))
    candidate_core_score = candidate_count / max(len(candidate_cores), 1)

    # compute F1
    if reference_core_score + candidate_core_score == 0:
        core_score = 0.5
    else:
        core_score = 2 * reference_core_score * candidate_core_score / (reference_core_score + candidate_core_score)

    # calibration
    core_score = calibrate_ucca_single(core_score, reference, candidate, reference_passage,
                                       candidate_passage, **kwargs)
    return core_score


# Meteor with UCCA-MTE
def meteor(references, candidates, ner=""):
    """
    :param references: list of string
    :param candidates: list of string
    :param ner: file path of preprocessed NER words, used in Meteor++. Default: "" as pure Meteor.
    :return: two lists of float
    """
    global MeteorMetric
    if MeteorMetric is None:
        # ner = "data/ner_wmt15-17.txt"
        MeteorMetric = Meteor(weights="1.0,0.6,0.8,0.6,0", hyper="0.85,0.2,0.6,0.75", lang="EN", paraphrase_invariant="",
                              paraphrase_file='data/filtered_paraphrase_table', paraphrase_file_format="pair",
                              function_words='data/english.function.words', ner_copy=ner)
    metric = MeteorMetric
    ucca_scores = []
    meteor_scores = []
    for line1, line2 in zip(references, candidates):
        meteor_score = metric.sentence_meteor(reference=line1.strip(), candidate=line2.strip(), norm=True)
        meteor_scores.append(meteor_score)
        ucca_score = ucca_mod(line1, line2)
        ucca_scores.append(ucca_score + meteor_score)
    # print((sum(meteor_scores) - sum(original_scores)) / 560)
    return ucca_scores, meteor_scores


# BLEU with UCCA-MTE
def bleu_mod(references, candidates):
    ucca_scores = []
    bleu_scores = []
    for line1, line2 in zip(references, candidates):
        ucca_score = ucca_mod(line1, line2)
        bleu_score = sentence_bleu([normalize_sentence(line1, filter_comma=True).lower().strip().split(' ')],
                                   normalize_sentence(line2, filter_comma=True).lower().strip().split(' '))
        ucca_scores.append(ucca_score + bleu_score)
        bleu_scores.append(bleu_score)
    # print((sum(meteor_scores) - sum(original_scores)) / 560)
    return ucca_scores, bleu_scores


# the test function on WMT dataset
def test_wmt(ref, system, human):
    with open(ref, "r", encoding='utf8') as f, open(system, "r", encoding='utf8') as f1, \
            open(human, "r", encoding='utf8') as f2:
        references = f.readlines()[:560]
        candidates = f1.readlines()[:560]
        ucca_parse_sentences(references)
        ucca_parse_sentences(candidates)
        scores = [float(score.strip()) for score in f2.readlines()][:560]
        # score_list = bleu_mod(references, candidates)
        score_list = meteor(references, candidates, ner="")
        df = pd.DataFrame({'meteor': score_list[1], 'ucca': score_list[0], 'human': scores})
        wmt.append(pd.DataFrame({'meteor': score_list[1], 'ucca': score_list[0], 'human': scores}))
        print(df.corr())
        return df.corr()


# the test function on WMT dataset
def test_wmt_multi(dataset, langs):
    print(dataset, langs)
    for lang in langs:
        reference_file = dataset + lang + '-en/reference'
        system_file = dataset + lang + '-en/system'
        human_file = dataset + lang + '-en/human'
        test_wmt(reference_file, system_file, human_file)


if __name__ == "__main__":
    wmt = []
    dataset = 'wmt15/'
    langs = ['de', 'fi', 'cs', 'ru']
    test_wmt_multi(dataset, langs)
    dataset = 'wmt16/'
    langs = ['de', 'fi', 'cs', 'ro', 'tr', 'ru']
    test_wmt_multi(dataset, langs)
    dataset = 'wmt17/'
    langs = ['de', 'fi', 'cs', 'lv', 'zh', 'tr', 'ru']
    # test_wmt_multi(dataset, langs)
