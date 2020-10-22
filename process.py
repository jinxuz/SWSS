import math
import pickle
import numpy as np
import pandas as pd


def get_parsed():
    global parsed
    with open('normal17.pkl', 'rb') as f1, open('parsed_sentences.pkl', 'rb') as f2:
        dnormal = pickle.load(f1)
        dparsed = pickle.load(f2)
    for k in dnormal.keys():
        parsed[k] = dparsed[dnormal[k]]


def pearson(a, b):
    return pd.DataFrame({'a': a, 'b': b}).corr().iloc[1, 0]


# get scene num
def get_scene(passage):
    scene = set()
    for node in passage.nodes.values():
        inedges = {e.tag for e in node.incoming}
        outedges = {e.tag for e in node.outgoing}
        if 'H' in inedges and 'Terminal' not in outedges or \
                (len({'A'}.intersection(inedges)) or node.ID == '1.1') and len({'P', 'S', 'D'}.intersection(outedges)):
            scene.add(node)
    return len(scene)


# get edge num of certain tags
def get_edge(passage, tag_set={'A', 'P', 'S'}):
    edges = set()
    for node in passage.nodes.values():
        for e in node.outgoing:
            if e.tag in tag_set:
                edges.add(e)
    return len(edges)


def reset_ucca(_series):
    # series[series < 0] = sum(series[series >= 0]) / len(series[series >= 0])
    _series[_series < 0] = 0.5
    return _series


# Using statistical features to modify the UCCA-MTE scores
def calibrate_ucca_single(score, reference_sentence, candidate_sentence, reference_passage, candidate_passage,
                          ucca_weight=0.2, length_weight=0.01, scene_weight=0.2, node_weight=1, edge_weight=0.5):

    if score < 0:
        score = 0.5

    rs, cs = get_scene(reference_passage), get_scene(candidate_passage)
    sence_bias = 1 - max(min(rs, cs), 1) / max(rs, cs, 1)
    score *= math.e ** (-sence_bias * scene_weight)

    rn = len(reference_passage.nodes)
    cn = len(candidate_passage.nodes)
    node_bias = 1 - min(rn, cn) / max(rn, cn, 1)
    score *= math.e ** (-node_bias * node_weight)

    re, ce = get_edge(reference_passage), get_edge(candidate_passage)
    edge_bias = 1 - min(re, ce) / max(re, ce, 1)
    score *= math.e ** (-edge_bias * edge_weight)

    score *= math.e ** (-(len(reference_sentence.split()) + len(candidate_sentence.split())) / 2 * length_weight)
    score *= ucca_weight

    return score


def calibrate_ucca(series, references=None, candidates=None):

    # reset_ucca(df.ucca)
    global parsed
    for i in range(len(series)):
        reference_passage = parsed[references[i]]
        candidate_passage = parsed[candidates[i]]

        series[i] = calibrate_ucca_single(series[i], references[i], candidates[i], reference_passage,
                                          candidate_passage)
    return series


if __name__ == "__main__":

    with open('wmt17.pkl', 'rb') as f:
        data = pickle.load(f)
        langs = pickle.load(f)
        langs = langs.split(' ')
    parsed = {}
    get_parsed()

    results = []
    for i in range(len(langs)):
        lang = langs[i]
        with open('wmt17/' + lang + '-en/reference', 'r', encoding='utf8') as f, \
                open('wmt17/' + lang + '-en/system', 'r', encoding='utf8') as f1:
            references = [s.strip() for s in f.readlines()[:560]]
            candidates = [s.strip() for s in f1.readlines()[:560]]
        df = data[i]
        df.ucca = calibrate_ucca(df.ucca, references=references, candidates=candidates)
        # print(pearson(df.human, df.ucca))
        df.ucca += df.meteor
        print(pearson(df.human, df.ucca))
        results.append(pearson(df.human, df.ucca))
        continue

