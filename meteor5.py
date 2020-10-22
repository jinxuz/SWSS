# coding=utf-8
# METEOR 算法 + 转述不变词 + 字符重叠

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
import os
from collections import namedtuple
import argparse
import numpy as np
import codecs
import re
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()
# stemmer = WordNetLemmatizer()

from itertools import tee, zip_longest
from nltk.corpus import stopwords

import nltk
import spacy
import math
from sklearn.feature_extraction import stop_words


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/", help="Data folder.")
parser.add_argument("--lang", type=str, default="CN",
                    help="Target language. Chinese by default.")
parser.add_argument("--cand", type=str, default="candidate.txt",
                    help="Candidate translation/paraphrase file.")
parser.add_argument("--ref", type=str, default="reference.txt",
                    help="Reference translation/paraphrase file.")

# Default settings for a universal language. Say, Chinese.
parser.add_argument("--weights", type=str, default="1.0,0,0,0.6,0.9",
                    help="Weights for exact/stem/synset/paraphrase/overlap matching, float numbers separated by comma. "
                         "0 indicates not applicable.")
parser.add_argument("--hyper", type=str, default="0.85,0.2,0.6,0.75,0.35",
                    help="Hyper parameters alpha, beta, gamma, and delta.")
parser.add_argument("--paraphrase_invariant", type=str, default="paraphrase_invariant.txt",
                    help="Invariant words during paraphrasing, i.e.: copy words.")
parser.add_argument("--paraphrase_file", type=str, default="checked.copypp.clean",
                    help="Paraphrase file.")
parser.add_argument("--paraphrase_file_format", type=str, default="pair",
                    help="Format of paraphrase file. Could be either 'pair' or 'clique'.")
parser.add_argument("--function_words", type=str, default="english.function.words",
                    help="Function words list")
args = parser.parse_args()

# "pair": each line contains a pair of paraphrases. It's okay to also include other information,
#           e.g.: weight/probability. These additional information will be omitted.
# "clique": each line contains a group of paraphrases.

# 五个匹配阶段
NUM_MODULES = 5
EXACT, STEM, SYNSET, PARA, SYNTAX = 0, 1, 2, 3, 4
FAKE_COUNT = 8
# FAKE_COUNT_ = 4
AVERAGE_MATCHED = 0
THRESHOLD = 1


# Matcher：某一个局部匹配
#       module: 指匹配的模式，取值为 EXACT/STEM/SYNSET/PARAPHRASE/OVERLAP
#       prob: 字面意思应该匹配的概率，不过 METEOR 的 Java 代码中好像都设置为 1
#       start & matchStart: 如果第一个句子的第 i 个词匹配了第二个句子的第 j 个词，则 start = j, matchStart = i
#       length & matchLength: 如果有短语匹配，第一个句子中的 m 个词匹配了第二个句子中的 n 个词，则 length = n, matchLength = m
Matcher = namedtuple("Matcher", ["module", "prob", "start", "length", "matchStart", "matchLength"])


nlp = spacy.load("en_core_web_md")


class Stage(object):
    # Stage: 暂存区，用于保存各个阶段的、所有可能的匹配结果
    def __init__(self, length1, length2):
        # length1, length2 分别是 candidate 和 reference 的长度
        # self.matches 中的每个元素都是由 Matcher 组成的列表
        self.matches = [list() for _ in range(length2)]
        self.line1Coverage = np.zeros(length1)
        self.line2Coverage = np.zeros(length2)

    def show(self):
        print("===== Show Stage: =====")
        print("# of possible matches:", sum([len(matcher_list) for matcher_list in self.matches]))

        for i, matcher_list in enumerate(self.matches):
            for matcher in matcher_list:
                print(matcher)
        print(self.line1Coverage)
        print(self.line2Coverage)
        print("-" * 30 + "\n")


class PartialAlignment(object):
    # PartialAlignment: 搜索最佳匹配的时候的中间结果
    def __init__(self, length1=-1, length2=-1, pa_obj=None):
        # length1 和 length2 分别是 candidate 和 reference 的长度
        # pa_obj 是另一个 PartialAlignment 对象
        # 两种初始化方式：提供 pa_obj 时 copy 一份；否则初始化一个空的 PartialAlignment 对象
        if pa_obj is None:
            # 记录句子 2 中每个位置对应的匹配（一开始全部初始化为空）
            self.matches = [None for _ in range(length2)]
            # 当前部分匹配的个数
            self.matchCount = 0
            # 当前部分匹配在第一个句子和第二个句子中的考虑匹配类别信息之后的总权重
            self.matches1, self.matches2 = 0.0, 0.0
            # 当前部分匹配覆盖到的句子 1 和句子 2 的词数
            self.allMatches1, self.allMatches2 = 0, 0
            self.chunks = 0
            # 句子 2 中当前要考虑的匹配起始位置的下标
            self.idx = 0
            # 所有匹配位置下标绝对值之差的和
            self.distance = 0
            # 最后一个匹配在句子 1 中的位置
            self.lastMatchEnd = -1
            self.line1UsedWords = np.zeros(length1, dtype=np.bool)
            self.line2UsedWords = np.zeros(length2, dtype=np.bool)
        else:
            self.matches = copy.copy(pa_obj.matches)
            self.matchCount = pa_obj.matchCount
            self.matches1, self.matches2 = pa_obj.matches1, pa_obj.matches2
            self.allMatches1, self.allMatches2 = pa_obj.allMatches1, pa_obj.allMatches2
            self.chunks = pa_obj.chunks
            self.idx = pa_obj.idx
            self.distance = pa_obj.distance
            self.lastMatchEnd = pa_obj.lastMatchEnd
            self.line1UsedWords = pa_obj.line1UsedWords.copy()
            self.line2UsedWords = pa_obj.line2UsedWords.copy()

    def isUsed(self, matcher):
        if np.sum(self.line2UsedWords[matcher.start:matcher.start+matcher.length]) > 0:
            # line2 used
            return True
        if np.sum(self.line1UsedWords[matcher.matchStart:matcher.matchStart + matcher.matchLength]) > 0:
            return True
        return False

    def setUsed(self, matcher, bool_flag):
        self.line2UsedWords[matcher.start:matcher.start + matcher.length] = bool_flag
        self.line1UsedWords[matcher.matchStart:matcher.matchStart + matcher.matchLength] = bool_flag

    def show(self):
        print("===== Show PartialAlignment: =====")
        print("# of matches:", len([matcher for matcher in self.matches if matcher is not None]))
        print("-")
        for i, matcher in enumerate(self.matches):
            print(matcher)
        print("-")

        print("Match weights:")
        print(self.matches1, self.matches2)

        print("# of covered words:")
        print(self.allMatches1, self.allMatches2)

        print("Used words:")
        print(self.line1UsedWords)
        print(self.line2UsedWords)
        print("-" * 30 + "\n")


class Alignment(object):
    # Alignment: 最终匹配结果哦，包括由所有最终选中的 Matcher 组成的集合，以及 function/content word 等信息
    def __init__(self, token_list1, token_list2):
        # token_list1, token_list2 分别是 candidate 和 reference 句子
        self.words1 = token_list1
        self.words2 = token_list2

        # 其实没必要存下面这四个列表，只要存四个数字就行了，管它到底哪个词是实的哪个词是虚的呢
        self.line1ContentWords = []
        self.line2ContentWords = []
        self.line1FunctionWords = []
        self.line2FunctionWords = []
        self.line1NERWords = []
        self.line2NERWords = []
        self.matches = []

        # matches[i] contains a match starting at index i in line2
        self.line1Matches, self.line2Matches = 0, 0
        # Per-module match totals (Content)
        self.moduleContentMatches1, self.moduleContentMatches2 = None, None
        # Per-module match totals (Function)
        self.moduleFunctionMatches1, self.moduleFunctionMatches2 = None, None
        # Per-module match totals (NER)
        self.moduleNERMatches1, self.moduleNERMatches2 = None, None


        self.numChunks = 0
        self.avgChunkLength = 0

    def set_count_and_chunks(self):
        self.line1Matches, self.line2Matches = 0, 0
        self.numChunks = 0
        idx, lastMatchEnd = 0, -1

        while idx < len(self.matches):
            matcher = self.matches[idx]
            if matcher is None:
                # A break in line 2 indicates end of a chunk.
                if lastMatchEnd != -1:
                    self.numChunks += 1
                    lastMatchEnd = -1
                idx += 1
            else:
                self.line1Matches += matcher.matchLength
                self.line2Matches += matcher.length
                if lastMatchEnd != -1 and matcher.matchStart != lastMatchEnd:
                    self.numChunks += 1
                idx = matcher.start + matcher.length
                lastMatchEnd = matcher.matchStart + matcher.matchLength

        if lastMatchEnd != -1:
            # print("before chunks:", self.numChunks)
            self.numChunks += 1
            pass
        # if self.numChunks > 0:
        #     self.avgChunkLength = (self.line1Matches + self.line2Matches) / (2.0 * self.numChunks)

    def show(self):

        print("===== Alignment: =====")
        print("Sentence 1: " + " ".join([token + "(" + str(i) + ")" for i, token in enumerate(self.words1)]))
        print("Sentence 2: " + " ".join([token + "(" + str(i) + ")" for i, token in enumerate(self.words2)]))
        print("# of matches:", len([matcher for matcher in self.matches if matcher is not None]))
        print("-")
        for i, matcher in enumerate(self.matches):
            
                print(matcher)
        print("-")

        print("-" * 30 + "\n")


class Meteor(object):
    def __init__(self, weights, hyper, lang, synset_file="", function_words="", ner_copy="",
                 paraphrase_invariant="", paraphrase_file="", paraphrase_file_format="pair"):
        self.lang = lang
        self.eps = 1e-6
        self.weights = [float(p) for p in weights.split(",")]
        self.hyper = [float(p) for p in hyper.split(",")]
        self.alpha, self.beta, self.gamma, self.delta = self.hyper
        self.beamSize = 40
        self.er, self.ec = 0, 0
        self.ner = set()

        self.stored_align = None

        self.function_words = {"（", "。", "；", "：", "，", "）", "、", "‘", "’", "“",
                               "”", "？", "！", "—", "《", "》", "…", ".", "•"}

        print(self.weights)
        print(self.hyper)
        # print("FAKE_COUNT:",FAKE_COUNT)
        # print("THRESHOLD:", THRESHOLD)
        
        # print("AVERAGE_MATCHED:",AVERAGE_MATCHED)
        self.total_matched = 0
        self.sentence_cnt = 0

        self.stemmer = PorterStemmer()

        # 加载同义词词表
        # 将一个单词 映射为 它出现的所有同义词集的 ID 的集合
        # self.possible_synsets = dict()
        # if os.path.isfile(synset_file):
        #     print("Loading synset from file " + synset_file)
        #     with codecs.open(synset_file, "r", encoding="utf-8") as f:
        #         # synset_file 的格式为每行一个同义词集
        #         for synset_clique_id, line in enumerate(f):
        #             tokens = line.split()
        #             for token in tokens:
        #                 if token in self.ossible_synsets:
        #                     self.possible_synsets[token].add(synset_clique_id)
        #                 else:
        #                     self.possible_synsets[token] = {synset_clique_id}
        # else:
        #     print("No synset knowledge.")

        self.paraphrase_invariant_words = set()

        #加载实词部分
        if os.path.isfile(function_words):
            print("Loading function_words from file " + function_words)

            with open(function_words, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.function_words.add(line)
            # print(self.function_words)

        # 加转述不变词
        if os.path.isfile(paraphrase_invariant):
            print("Loading paraphrase invariants from file " + paraphrase_invariant)
            with codecs.open(paraphrase_invariant, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.split()[0]
                    self.paraphrase_invariant_words.add(word)
        else:

            print("", paraphrase_invariant)

        # 加入实体识别知识
        # if os.path.isfile(ner_copy):
        #     print("Loading NER COPY from file " + ner_copy)
        #     with codecs.open(ner_copy, "r", encoding="utf-8") as f:
        #         for line in f:
        #             word = line.split()[0]
        #             self.paraphrase_invariant_words.add(word)
        # else:
        #     print("nonono:",ner_copy)

        if os.path.isfile(ner_copy):
            print("Loading NER COPY from file " + ner_copy)
            with codecs.open(ner_copy, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.split()[0]
                    self.paraphrase_invariant_words.add(word.lower())
        else:
            print("Not find ", ner_copy)

        self.stop_words = stop_words.ENGLISH_STOP_WORDS

        self.paraphrase_invariant_words -= self.stop_words
        self.paraphrase_invariant_words -= self.function_words
        self.paraphrase_invariant_words -= self.ner

        print(len(self.paraphrase_invariant_words))

        # 加载转述词表 对应的是Meteor中的paraphrase知识
        # 将一个单词 映射为 它的所有转述词的集合
        self.possible_paraphrases = dict()
        if os.path.isfile(paraphrase_file):
            print("Loading paraphrases from file " + paraphrase_file)
            with open(paraphrase_file, "r", encoding='utf8') as f:
                if paraphrase_file_format == "pair":
                    # paraphrase_file 的格式为每行两个词，以及一些可能的权重信息
                    for i, line in enumerate(f):
                        tokens = line.split("||||")
                        x, y = tokens[0].strip(), tokens[1].strip()
                        # print(x)
                        # print(y)
                        # print("+"*60)
                        if x in self.possible_paraphrases:
                            self.possible_paraphrases[x].add(y)
                        else:
                            self.possible_paraphrases[x] = {y}

                        if y in self.possible_paraphrases:
                            self.possible_paraphrases[y].add(x)
                        else:
                            self.possible_paraphrases[y] = {x}
                else:
                    # paraphrase_file 的格式为每行一个同义词集
                    for i, line in enumerate(f):
                        # possible_paraphrases[token] 里也包含了这个词自己，不过不影响程序正确性
                        tokens = line.split()
                        for token in tokens:
                            if token in self.possible_paraphrases:
                                self.possible_paraphrases[token] = self.possible_paraphrases[token] | set(tokens)
                            else:
                                self.possible_paraphrases[token] = set(tokens)
        
        else:
            print("No paraphrase knowledge.")

        # 加入句法相关的知识
        self.possible_syntax_possibles = dict()

    # 同义词词林的加入

    def possible_synsets(self, word):
        possible_sets = []
        for synset in wn.synsets(word):
            possible_sets += synset.lemma_names()
        return set(possible_sets)

    def NER_parsing(self, text):
        
        ner_sets = set()
        words = nltk.word_tokenize(text)
        tags = nltk.pos_tag(words)
        ners = nltk.ne_chunk(tags, binary=True)
        for t in ners.subtrees(lambda t: t.label() == 'NE'):
            ner_sets |= set(leaf[0].lower() for leaf in t.leaves())
        
        return ner_sets

    @staticmethod
    def _normalize(sentence):
        # print(sentence)
        
        # for i in ["[",".","!",":"."/","_",",".";","$","%","^",'"'*(+\")]+|[+——()?【】“”！，。？；、~@#￥%……&*（）]+"]
        sentence = re.sub(r"[+\!:\/_,;$%^*(+\")-]+|[+——()?【】“”！，。？；、~@#￥%……&*（）]+", "", sentence)

        # token_list = [i.lower()for i in nltk.word_tokenize(sentence)]
        token_list = [i.text.lower() for i in nlp(sentence)]
        if '.' in token_list:
            token_list.remove('.')
        # sentence = sentence.replace("s'", "s")
        # sentence = re.sub("\'", " \'", sentence).lower().split()
        # candidate = re.sub("[+\.\!:\/_,;$%^*(+\")]+|[+——()?【】“”！，。？；、~@#￥%……&*（）]+","",candidate).lower().split()

        return token_list

    def align(self, candidate_token_list, reference_token_list, mode="RUN"):

        # def align(self, candidate_token_list, reference_token_list, reference_NER_set, candidate_NER_set, mode="RUN"):
        # mode: "RUN" 中间不输出调试信息，"DEBUG" 输出中间结果以供调试
        # 两个列表相等时进行特判，只做 EXACT MATCH，防止 beam search 出问题
        is_identity = " ".join(candidate_token_list) == " ".join(reference_token_list)

        # 依次考虑 exact/stem/synset/paraphrase 四个阶段
        stage = Stage(length1=len(candidate_token_list), length2=len(reference_token_list))

        if self.weights[0] > self.eps:
            self._exact_matching(stage, candidate_token_list, reference_token_list)
        if not is_identity and self.weights[1] > self.eps and self.stemmer is not None:
            # TODO: 其实如果权重大于 eps 但是没有 stemmer 应该报个错，以下三个分支同
            self._stem_matching(stage, candidate_token_list, reference_token_list)
        if not is_identity and self.weights[2] > self.eps  > 0:
            self._synset_matching(stage, candidate_token_list, reference_token_list)
        if not is_identity and self.weights[3] > self.eps and len(self.possible_paraphrases) > 0:
            self._paraphrase_matching(stage, candidate_token_list, reference_token_list)
        if not is_identity and self.weights[4] > self.eps:
            self._syntax_matching(stage, candidate_token_list, reference_token_list)

        if mode == "DEBUG":
            stage.show()

        # 通过启发式搜索找到最佳匹配方案
        # 先预处理一些能够确认的 matcher (one-to-one, non-overlapping matches)，缩小搜索范围
        initialPath = PartialAlignment(length1=len(candidate_token_list), length2=len(reference_token_list))
        for matcher_list in stage.matches:
            if len(matcher_list) == 1:
                matcher = matcher_list[0]
                # 注意：i, j 分别是句子 2 和句子 1 的起始位置（而非反过来！）
                i, j = matcher.start, matcher.matchStart
                if np.sum(stage.line2Coverage[i:i+matcher.length]) > 1:
                    continue
                if np.sum(stage.line1Coverage[j:j+matcher.matchLength]) > 1:
                    continue
                # 此处未更新 initialPath 的匹配计数和权重等信息，在 resolve() 函数里会统计
                # 这样设计的主要考虑是，后续搜索的时候才能知道这些匹配的 chunk 信息（它们跟别的匹配有没有连起来）
                initialPath.matches[i] = matcher
                initialPath.line2UsedWords[i:i+matcher.length] = True
                initialPath.line1UsedWords[j:j+matcher.matchLength] = True

        # 然后进行 beam search
        best = self.resolve(stage, initialPath, mode)
        if mode == "DEBUG":
            print("Best path:")
            best.show()

        # 最后做一些统计工作，并返回最终的对齐结果
        a = Alignment(candidate_token_list, reference_token_list)

        a.moduleContentMatches1 = np.zeros(NUM_MODULES, dtype=np.int)
        a.moduleContentMatches2 = np.zeros(NUM_MODULES, dtype=np.int)
        a.moduleFunctionMatches1 = np.zeros(NUM_MODULES, dtype=np.int)
        a.moduleFunctionMatches2 = np.zeros(NUM_MODULES, dtype=np.int)

        for i, token in enumerate(candidate_token_list):
            if token in self.function_words:
                a.line1FunctionWords.append(i)

            else:
                a.line1ContentWords.append(i)

        for j, token in enumerate(reference_token_list):
            # print(token)
            if token in self.function_words:
                a.line2FunctionWords.append(j)

            else:
                a.line2ContentWords.append(j)

        for matcher in best.matches:
            if matcher is not None:
                # 如果第二个句子的某个词 j 的匹配不为空，再更新整个句子的匹配信息
                for k in range(matcher.matchLength):
                    if candidate_token_list[matcher.matchStart + k] in self.function_words:
                        a.moduleFunctionMatches1[matcher.module] += 1
                    
                    # elif candidate_token_list[matcher.matchStart + k] in self.paraphrase_invariant_words or testNERMatch(matcher, "c"):
                    #     a.moduleNERMatches1[matcher.module] += 1
                    #     print("candidate matched:",candidate_token_list[matcher.matchStart + k])
                    # #     
                    else:
                        a.moduleContentMatches1[matcher.module] += 1
                for k in range(matcher.length):
                    if reference_token_list[matcher.start + k] in self.function_words:
                        a.moduleFunctionMatches2[matcher.module] += 1
                    # elif reference_token_list[matcher.start + k] in total_NER or testNERMatch(matcher, "r"):
                    #     print("reference matched:",reference_token_list[matcher.start + k])
                    #     a.moduleNERMatches2[matcher.module] += 1
                    # elif reference_token_list[matcher.start + k] in self.paraphrase_invariant_words or testNERMatch(matcher, "r"):
                    #     a.moduleNERMatches2[matcher.module] += 1
                    #     print("reference matched:",reference_token_list[matcher.start + k])
                
                    else:
                        a.moduleContentMatches2[matcher.module] += 1
        a.matches = best.matches
        a.set_count_and_chunks()
        return a

    def _exact_matching(self, stage, token_list1, token_list2):
        
        for j in range(len(token_list2)):
            for i in range(len(token_list1)):
                if token_list1[i] == token_list2[j]:
                    stage.matches[j].append(Matcher(module=EXACT, prob=1.0, start=j, length=1,
                                                    matchStart=i, matchLength=1))
                    stage.line1Coverage[i] += 1
                    stage.line2Coverage[j] += 1

    def _stem_matching(self, stage, token_list1, token_list2, verbose=False):
        # TODO: currently stemmer is None!
        stem_list1 = [self.stemmer.stem(token) for token in token_list1]
        stem_list2 = [self.stemmer.stem(token) for token in token_list2]
        for j in range(len(token_list2)):
            for i in range(len(token_list1)):
                if stem_list1[i] == stem_list2[j] and token_list1[i] != token_list2[j]:

                    stage.matches[j].append(Matcher(module=STEM, prob=1.0, start=j, length=1,
                                                    matchStart=i, matchLength=1))
                    stage.line1Coverage[i] += 1
                    stage.line2Coverage[j] += 1

    def _synset_matching(self, stage, token_list1, token_list2, verbose=False):

        for j in range(len(token_list2)):
            for i in range(len(token_list1)):
                t1, t2 = token_list1[i], token_list2[j]
                t1_synsets = self.possible_synsets(t1)
                t2_synsets = self.possible_synsets(t2)
                if verbose:
                    print("\nsynset module result\n")
                    print("reference:", t1, t1_synsets)
                    print("candidate:", t2, t2_synsets)
                    print("intersection:", t1_synsets & t2_synsets)
                if t1 != t2 and len(t1_synsets & t2_synsets) > 0:
                    stage.matches[j].append(Matcher(module=SYNSET, prob=1.0, start=j, length=1,
                                                    matchStart=i, matchLength=1))
                    stage.line1Coverage[i] += 1
                    stage.line2Coverage[j] += 1

    def _syntax_matching(self, stage, token_list1, token_list2, verbose=False):
        pass

    # 为paraphrase部分选出n-gram
    def uniwise(self,s):
        # 列表元素是元组合，分别代表的内容是(index,length,(n-gram))
        pair = []
        for i in range(len(s)):
            pair.append([i, 1, (s[i],)])
        return pair

    def pairwise(self, s):
        # 列表元素是元组合，分别代表的内容是(index,length,(n-gram))
        pair = []
        for i in range(len(s)-1):
            pair.append([i, 2, (s[i], s[i+1])])
        return pair

    def triwise(self, s):
        # 列表元素是元组合，分别代表的内容是(index,length,(n-gram))
        pair = []
        for i in range(len(s)-2):
            
            pair.append([i, 3, (s[i], s[i+1], s[i+2])])
        return pair

    def fourwise(self, s):
        # 列表元素是元组合，分别代表的内容是(index,length,(n-gram))
        pair = []
        for i in range(len(s)-3):
            pair.append([i, 4, (s[i], s[i+1], s[i+2], s[i+3])])
        return pair

    def n_gram(self, s):
        # 获得句子的所有n-gram组合(n = 1,2,3,4)
        return self.uniwise(s) + self.pairwise(s) + self.triwise(s) + self.fourwise(s)

    def _paraphrase_matching(self, stage, token_list1, token_list2):
        # 我挖掘的是词汇转述网络，因此我的转述极大团实际上只相当于 METEOR 里的 synset
        # METEOR 里的 paraphrase 是可以多对多的（当然，一对一也包含了进来）
        # TODO: 以后要改成支持多对多的话，可以参考如下代码：
        # https://github.com/cmu-mtlab/meteor/blob/master/src/edu/cmu/meteor/aligner/ParaphraseMatcher.java
        # 暂时先写成和 SYNSET 匹配几乎一样的好了

        # 先找出所有的n-gram集合
        t1 = self.n_gram(token_list1)
        t2 = self.n_gram(token_list2)

        for j in range(len(t2)):
            for i in range(len(t1)):
                
                t1_, t2_ = ' '.join(t1[i][2]), ' '.join(t2[j][2])
                if (t1_ != t2_) and (t1_ in self.possible_paraphrases and t2_ in self.possible_paraphrases[t1_]):
                    try:
                        stage.matches[t2[j][0]].append(Matcher(module=PARA, prob=1.0, start=t2[j][0], length=t2[j][1],
                                                               matchStart=t1[i][0], matchLength=t1[i][1]))
                        # print("debug Matcher!:",Matcher(module=PARA, prob=1.0, start=t2[j][0], length=t2[j][1],
                        #                                  matchStart=t1[i][0], matchLength=t1[i][1]))

                    except Exception as e:
                        exit()

                        # print(t1,t2)
                        # exit()

                    # 更新覆盖范围
                    for index, word in enumerate(t1[i][2]):
                        stage.line1Coverage[t1[i][0] + index ] += 1
                    for index, word in enumerate(t2[j][2]):
                        stage.line2Coverage[t2[j][0] + index ] += 1

    # def _overlap_matching(self, stage, token_list1, token_list2):
    #     for j in range(len(token_list2)):
    #         for i in range(len(token_list1)):
    #             t1, t2 = token_list1[i], token_list2[j]
    #             if (t1 != t2) and len(set(t1) & set(t2)) > 0:
    #                 stage.matches[j].append(Matcher(module=OVERLAP, prob=1.0, start=j, length=1,
    #                                                      matchStart=i, matchLength=1))
    #                 stage.line1Coverage[i] += 1
    #                 stage.line2Coverage[j] += 1

    def resolve(self, stage, start, mode):
        # mode: "RUN" 中间不输出调试信息，"DEBUG" 输出中间结果以供调试
        # 使用 beam search 从所有可能的匹配里搜索一个最好的

        def pa_to_key(pa):
            # 把 PartialAlignment 对象转换成 key
            # 相当于 Meteor 里的 PARTIAL_COMPARE_TOTAL
            # pa1 < pa2    <==>    pa_to_key(pa1) < pa_to_key(pa2)
            return pa.matches1 + pa.matches2, -pa.chunks, -pa.distance

        # 当前搜索队列和下一步待搜索队列
        paths, nextPaths = [], []
        nextPaths.append(start)
        if mode == "DEBUG":
            print("Begining search: ", len(nextPaths))

        # 注意 stage.matches 是一个长为 length2 的列表，其中每个元素是对应的句子 2 中的词的所有可能匹配的列表
        length2 = len(stage.matches)
        # Proceed from left to right
        for current in range(length2 + 1):
            # Advance
            paths = nextPaths
            nextPaths = []
            paths.sort(key=lambda pa: pa_to_key(pa), reverse=True)

            if mode == "DEBUG":
                print("In beam search step " + str(current) + ", PartialAlignment list is:")
                for path in paths:
                    path.show()

            # Try as many paths as beam allows
            num_trials = min(self.beamSize, len(paths))

            for rank in range(num_trials):
                path = paths[rank]

                if mode == "DEBUG":
                    print("Beam search base on following path:")
                    path.show()

                # Case 1: Path is complete
                if current == length2:
                    # Close last chunk
                    if path.lastMatchEnd != -1:
                        path.chunks += 1
                    nextPaths.append(path)
                    if mode == "DEBUG":
                        print("Append Case 1!")
                    continue

                # Case 2: Current index word is in use
                if path.line2UsedWords[current]:
                    # If this is still part of a match
                    #   如果之前一个匹配覆盖了多个词，就会出现这种情况
                    if current < path.idx:
                        nextPaths.append(path)
                        if mode == "DEBUG":
                            print("Append Case 2.1!")
                    # If fixed match
                    #   如果在预处理阶段把这个 matcher 包含了进来，就会出现这种情况
                    elif path.matches[path.idx] is not None:
                        matcher = path.matches[path.idx]
                        path.matchCount += 1
                        path.matches1 += matcher.matchLength * self.weights[matcher.module]
                        path.matches2 += matcher.length * self.weights[matcher.module]
                        path.allMatches1 += matcher.matchLength
                        path.allMatches2 += matcher.length
                        # Not conitnuous in line1
                        if path.lastMatchEnd != -1 and matcher.matchStart != path.lastMatchEnd:
                            path.chunks += 1
                        # Advance to end of match + 1
                        path.idx = matcher.start + matcher.length
                        path.lastMatchEnd = matcher.matchStart + matcher.matchLength
                        path.distance += abs(matcher.start - matcher.matchStart)
                        nextPaths.append(path)
                        if mode == "DEBUG":
                            print("Append Case 2.2!")
                    continue

                # Case 3: Multiple possible matches
                # 前两种情况直接修改 path 然后 continue 即可；这种情况需要将 path 复制多份
                matches = stage.matches[current]
                for matcher in matches:
                    if not path.isUsed(matcher):
                        newPath = PartialAlignment(pa_obj=path)
                        newPath.setUsed(matcher, True)
                        newPath.matches[current] = matcher

                        newPath.matchCount += 1
                        newPath.matches1 += matcher.matchLength * self.weights[matcher.module]
                        newPath.matches2 += matcher.length * self.weights[matcher.module]

                        if newPath.lastMatchEnd != -1 and matcher.matchStart != newPath.lastMatchEnd:
                            newPath.chunks += 1
                        newPath.idx = matcher.start + matcher.length
                        newPath.lastMatchEnd = matcher.matchStart + matcher.matchLength
                        path.distance += abs(matcher.start - matcher.matchStart)
                        nextPaths.append(newPath)
                        if mode == "DEBUG":
                            print("Append Case 3!")

                # Case 4: skip this index
                if path.lastMatchEnd != -1:
                    path.chunks += 1
                    path.lastMatchEnd = -1
                path.idx += 1
                nextPaths.append(path)
                if mode == "DEBUG":
                    print("Append Case 4!")

            if len(nextPaths) == 0:
                print("Warning: unexpected conditions - skipping matches until possible to continue")

        nextPaths.sort(key=lambda pa: pa_to_key(pa), reverse=True)
        return nextPaths[0]

    def testNERMatch(self, candidate_token_list, reference_token_list, matcher, flag):
            if flag == "r":
                for k in range(matcher.matchLength):
                    if candidate_token_list[matcher.matchStart + k] in self.paraphrase_invariant_words:
                        self.ec += 1
                        return True
                
                return False
            else:
                for k in range(matcher.length):
                    if reference_token_list[matcher.start + k] in self.paraphrase_invariant_words:
                        self.er += 1
                        return True
                return False

    def sentence_meteor_ner(self, candidate, reference, norm, verbose=False):
        self.er = 0
        self.ec = 0
        # reference 和 candidate 可以是 Unicode string，也可以是 Unicode string 的列表，
        #       例如 "我爱你。" 或 ["我", "爱", "你", "。"]
        # 只支持一个 reference 的情形
        # norm 表示是否对 Unicode string 进行需要 tokenize
        # print("="*60)
        assert type(reference) == type(candidate)
        # print("reference:",reference)
        # print("candidate:",candidate)
        # reference_NER_set = set()
        # candidate_NER_set = set()
        reference_NER_set = self.NER_parsing(reference)-self.function_words
        candidate_NER_set = self.NER_parsing(candidate)-self.function_words

        # print("before norm:",candidate)
        if candidate and norm:
            candidate = self._normalize(candidate)
            reference = self._normalize(reference)
        # print("candidate after norm:",candidate)

        # if verbose:
        #     print("Candidate: ", candidate)
        #     print("Reference: ", reference)

        # 此时 reference 和 candidate 都应该是 Unicode string 的列表，即形如 ["我", "爱", "你", "。"]
        # a = self.align(candidate_token_list=candidate, reference_token_list=reference,\
        #                reference_NER_set = reference_NER_set, candidate_NER_set = candidate_NER_set)
        a = self.align(candidate_token_list=candidate, reference_token_list=reference)
        self.stored_align = a

        if verbose:
            a.show()

        # P, R, ch, m = 1.0, 1.0, 6, 6
        # P, R, ch, m = 1.0, 1.0, 1, 6
        # P, R, ch, m = 0.8571, 1.0, 2, 6
        P, R = 0, 0
        for i in range(NUM_MODULES):
            # P += self.weights[i] * (self.delta1 * a.moduleNERMatches1[i]
            #                         + self.delta2 * a.moduleContentMatches1[i]
            #                         + (1-self.delta1 - self.delta2)*a.moduleFunctionMatches1[i])
            # R += self.weights[i] * (self.delta1 * a.moduleNERMatches2[i]
            #                         + self.delta2 * a.moduleContentMatches2[i]
            #                         + (1 - self.delta1-self.delta2) * a.moduleFunctionMatches2[i])

            P += self.weights[i] * (self.delta * a.moduleContentMatches1[i]
                                    + (1 - self.delta) * a.moduleFunctionMatches1[i])
            R += self.weights[i] * (self.delta * a.moduleContentMatches2[i]
                                    + (1 - self.delta) * a.moduleFunctionMatches2[i])

        P /= (self.delta * len(a.line1ContentWords) + (1 - self.delta) * len(a.line1FunctionWords))

        R /= (self.delta * len(a.line2ContentWords) + (1 - self.delta) * len(a.line2FunctionWords))

        line1_matched_stable_words, line2_matched_stable_words = 0, 0
        # can_ner = set()
        # ref_ner = set()
        # can_ner = self.NER_parsing(ori_can) #- self.stop_words
        # ref_ner = self.NER_parsing(ori_ref) #- self.stop_words

        for matcher in a.matches:
            if matcher is not None:
                # 如果第二个句子的某个词 j 的匹配不为空，再更新整个句子的匹配信息
                # print("+"*60)
                for k in range(matcher.matchLength):
                    if candidate[
                                matcher.matchStart + k] in self.paraphrase_invariant_words:  # or self.testNERMatch(candidate_token_list = candidate, reference_token_list = reference ,matcher = matcher, flag = "c"):
                        line1_matched_stable_words += 1
                        # print("matched_candidate:",candidate[matcher.matchStart + k])
                        # print("candidate stable:",candidate[matcher.matchStart + k])
                # print("*"*60)
                for k in range(matcher.length):
                    if reference[
                                matcher.start + k] in self.paraphrase_invariant_words:  # or self.testNERMatch(candidate_token_list = candidate, reference_token_list = reference ,matcher = matcher, flag = "r"):
                        line2_matched_stable_words += 1
                        # print("matchedreference:",reference[matcher.start + k])
                        # print("reference stable:",reference[matcher.start + k])

        # # 如果转述不变词在匹配中被漏掉了，加以相应的惩罚
        # print("total_reference:",[word for word in reference if word in self.paraphrase_invariant_words])
        # print("total_candidate:",[word for word in candidate if word in self.paraphrase_invariant_words])

        line1_total_stable_words = len(
            [word for word in candidate if word in self.paraphrase_invariant_words])  # + self.er
        line2_total_stable_words = len(
            [word for word in reference if word in self.paraphrase_invariant_words])  # + self.ec

        # print(FAKE_COUNT)
        # print(AVERAGE_MATCHED)

        self.total_matched += line1_total_stable_words
        self.total_matched += line2_total_stable_words
        self.sentence_cnt += 2

        # # if len(ref_ner) > 0:
        # #     print(ref_ner)
        # #     print(ori_ref)
        # #     print("reference:",[word for word in reference if word in ref_ner])
        # #     print("reference_matched_words:",line2_matched_stable_words)
        # # if len(can_ner) > 0:
        # #     print(can_ner)
        # #     print(ori_can)
        # #     print("candidate:",[word for word in candidate if word in can_ner])
        # #     print("candidate_matched_words:",line1_matched_stable_words)
        # print("ref_total_stable:", [word for word in reference if word in self.paraphrase_invariant_words])
        # print("can_total_stable:", [word for word in candidate if word in self.paraphrase_invariant_words])

        # Pen_P = (line1_matched_stable_words + FAKE_COUNT) / (line1_total_stable_words + FAKE_COUNT)
        # Pen_R = (line2_matched_stable_words + FAKE_COUNT) / (line2_total_stable_words + FAKE_COUNT)

        Pen_P = (line1_total_stable_words - line1_matched_stable_words) / (line1_total_stable_words + FAKE_COUNT)
        Pen_R = (line2_total_stable_words - line2_matched_stable_words) / (line2_total_stable_words + FAKE_COUNT)

        # if line1_total_stable_words == 0:
        #     Pen_P = 0
        # else:
        #     Pen_P = (line1_total_stable_words - line1_matched_stable_words ) / (line1_total_stable_words )
        # if line2_total_stable_words == 0:
        #     Pen_R = 0

        # else:
        #     Pen_R = (line2_total_stable_words - line2_matched_stable_words ) / (line2_total_stable_words )

        # Pen_P = math.exp((line1_matched_stable_words + FAKE_COUNT) / (line1_total_stable_words + FAKE_COUNT))
        # Pen_R = math.exp((line2_matched_stable_words + FAKE_COUNT) / (line2_total_stable_words + FAKE_COUNT) - 1)

        # Pen_P = A * Pen_P  * THRESHOLD + THRESHOLD
        # Pen_R = Pen_R  * THRESHOLD + THRESHOLD

        # print("orignal_P:", P, " orignal_R:", R)
        # print("before:", " Pen_P:",Pen_P,"    Pen_R:",Pen_R)
        # Pen_P =  w * (Pen_P ** THRESHOLD)
        # Pen_R =  w * (Pen_R ** THRESHOLD)

        Pen_P = (1 - Pen_P)
        Pen_R = (1 - Pen_R)

        # #if line1_total_stable_words == 0:
        #     Pen_P = 1
        #     Pen_P_ = 1
        # else:
        #     Pen_P = (line1_matched_stable_words + FAKE_COUNT) / (line1_total_stable_words + FAKE_COUNT)
        #     Pen_P_ = (line1_matched_stable_words + FAKE_COUNT_) / (line1_total_stable_words + FAKE_COUNT_)

        # if line2_total_stable_words == 0:
        #     Pen_R = 1
        #     Pen_R_ = 1
        # else:
        #     Pen_R = (line2_matched_stable_words + FAKE_COUNT) / (line2_total_stable_words + FAKE_COUNT)
        #     Pen_R_ = Pen_R_ = (line2_matched_stable_words + FAKE_COUNT_) / (line2_total_stable_words + FAKE_COUNT_)
        # # if Pen_P != 1 or Pen_R != 1:
        #      print("PENALTY:",Pen_P, Pen_R,P,R)
        # print("orignal_P:",P)
        # print("orignal_R:",R)
        # print("after:", " Pen_P:",Pen_P,"    Pen_R:",Pen_R)
        # print(FAKE_COUNT_,"  Pen_P_:",Pen_P_,"    Pen_R_:",Pen_R_)
        P, R = P * (Pen_P), R * (Pen_R)

        num_chunks = a.numChunks  # `ch` in Meteor formula
        # print("num_chunks:", num_chunks)
        num_matched_words = (a.line1Matches + a.line2Matches) / 2.0  # `m` in Meteor formula
        try:
            F_mean = (P * R) / (self.alpha * P + (1 - self.alpha) * R)
            # print("changdu candidate:", len(candidate))
            # Pen = self.gamma * ((num_chunks / num_matched_words) * (1/len(candidate))) ** self.beta
            Pen = self.gamma * (num_chunks / num_matched_words) ** self.beta
            # print("Pen:",Pen)
        except Exception as e:
            # print("X"*60)
            # print(reference)
            # print(candidate)
            # print(P)
            # print(R)
            return 0
            # print(F_mean)

        score = (1 - Pen) * F_mean
        # score = F_mean
        # print("final_score:", score)
        if verbose:
            print("Statistics:")
            print("P = ", P, ", R = ", R, ", ch = ", num_chunks, ", m = ", num_matched_words,
                  ", Pen = ", Pen, " , F_mean = ", F_mean)
        return score

    def sentence_meteor(self, candidate, reference, norm, verbose=False):
        self.er = 0
        self.ec = 0
        # reference 和 candidate 可以是 Unicode string，也可以是 Unicode string 的列表，
        #       例如 "我爱你。" 或 ["我", "爱", "你", "。"]
        # 只支持一个 reference 的情形
        # norm 表示是否对 Unicode string 进行需要 tokenize
        # print("="*60)
        assert type(reference) == type(candidate)
        # print("reference:",reference)
        # print("candidate:",candidate)
        # reference_NER_set = set()
        # candidate_NER_set = set()
        # reference_NER_set = self.NER_parsing(reference)-self.function_words
        # candidate_NER_set = self.NER_parsing(candidate)-self.function_words
        
        # print("before norm:",candidate)
        if candidate and norm:
            candidate = self._normalize(candidate)
            reference = self._normalize(reference)
        # print("candidate after norm:",candidate)

        # if verbose:
        #     print("Candidate: ", candidate)
        #     print("Reference: ", reference)

        # 此时 reference 和 candidate 都应该是 Unicode string 的列表，即形如 ["我", "爱", "你", "。"]
        # a = self.align(candidate_token_list=candidate, reference_token_list=reference,\
        #                reference_NER_set = reference_NER_set, candidate_NER_set = candidate_NER_set)
        a = self.align(candidate_token_list=candidate, reference_token_list=reference)
        self.stored_align = a

        if verbose:
            a.show()

        # P, R, ch, m = 1.0, 1.0, 6, 6
        # P, R, ch, m = 1.0, 1.0, 1, 6
        # P, R, ch, m = 0.8571, 1.0, 2, 6
        P, R = 0, 0
        for i in range(NUM_MODULES):    
            # P += self.weights[i] * (self.delta1 * a.moduleNERMatches1[i]
            #                         + self.delta2 * a.moduleContentMatches1[i]
            #                         + (1-self.delta1 - self.delta2)*a.moduleFunctionMatches1[i])
            # R += self.weights[i] * (self.delta1 * a.moduleNERMatches2[i]
            #                         + self.delta2 * a.moduleContentMatches2[i]
            #                         + (1 - self.delta1-self.delta2) * a.moduleFunctionMatches2[i])

            P += self.weights[i] * (self.delta * a.moduleContentMatches1[i]
                                    + (1-self.delta)*a.moduleFunctionMatches1[i])
            R += self.weights[i] * (self.delta * a.moduleContentMatches2[i]
                                    + (1 - self.delta) * a.moduleFunctionMatches2[i])

        P /= (self.delta * len(a.line1ContentWords) + (1 - self.delta) * len(a.line1FunctionWords))

        R /= (self.delta * len(a.line2ContentWords) + (1 - self.delta) * len(a.line2FunctionWords))

        line1_matched_stable_words, line2_matched_stable_words = 0, 0
        # can_ner = set()
        # ref_ner = set()
        # can_ner = self.NER_parsing(ori_can) #- self.stop_words
        # ref_ner = self.NER_parsing(ori_ref) #- self.stop_words
        
        for matcher in a.matches:
            if matcher is not None:
                # 如果第二个句子的某个词 j 的匹配不为空，再更新整个句子的匹配信息
                # print("+"*60)
                for k in range(matcher.matchLength):
                    if candidate[matcher.matchStart + k] in self.paraphrase_invariant_words:# or self.testNERMatch(candidate_token_list = candidate, reference_token_list = reference ,matcher = matcher, flag = "c"):
                        line1_matched_stable_words += 1
                        # print("matched_candidate:",candidate[matcher.matchStart + k])
                        # print("candidate stable:",candidate[matcher.matchStart + k])
                # print("*"*60)
                for k in range(matcher.length):
                    if reference[matcher.start + k] in self.paraphrase_invariant_words :#or self.testNERMatch(candidate_token_list = candidate, reference_token_list = reference ,matcher = matcher, flag = "r"):
                        line2_matched_stable_words += 1
                        # print("matchedreference:",reference[matcher.start + k])
                        # print("reference stable:",reference[matcher.start + k])

        # # 如果转述不变词在匹配中被漏掉了，加以相应的惩罚
        # print("total_reference:",[word for word in reference if word in self.paraphrase_invariant_words])
        # print("total_candidate:",[word for word in candidate if word in self.paraphrase_invariant_words])

        line1_total_stable_words = len([word for word in candidate if word in self.paraphrase_invariant_words]) #+ self.er
        line2_total_stable_words = len([word for word in reference if word in self.paraphrase_invariant_words]) #+ self.ec
        
        # print(FAKE_COUNT)
        # print(AVERAGE_MATCHED)

        self.total_matched += line1_total_stable_words
        self.total_matched += line2_total_stable_words
        self.sentence_cnt += 2
        
        # # if len(ref_ner) > 0:
        # #     print(ref_ner)
        # #     print(ori_ref)
        # #     print("reference:",[word for word in reference if word in ref_ner])
        # #     print("reference_matched_words:",line2_matched_stable_words)
        # # if len(can_ner) > 0:
        # #     print(can_ner)
        # #     print(ori_can)
        # #     print("candidate:",[word for word in candidate if word in can_ner])
        # #     print("candidate_matched_words:",line1_matched_stable_words)
        # print("ref_total_stable:", [word for word in reference if word in self.paraphrase_invariant_words])
        # print("can_total_stable:", [word for word in candidate if word in self.paraphrase_invariant_words])
        
        # Pen_P = (line1_matched_stable_words + FAKE_COUNT) / (line1_total_stable_words + FAKE_COUNT)
        # Pen_R = (line2_matched_stable_words + FAKE_COUNT) / (line2_total_stable_words + FAKE_COUNT)
        
        Pen_P = (line1_total_stable_words - line1_matched_stable_words) / (line1_total_stable_words + FAKE_COUNT)
        Pen_R = (line2_total_stable_words - line2_matched_stable_words) / (line2_total_stable_words + FAKE_COUNT)

        # if line1_total_stable_words == 0:
        #     Pen_P = 0
        # else:
        #     Pen_P = (line1_total_stable_words - line1_matched_stable_words ) / (line1_total_stable_words )
        # if line2_total_stable_words == 0:
        #     Pen_R = 0

        # else:
        #     Pen_R = (line2_total_stable_words - line2_matched_stable_words ) / (line2_total_stable_words )

        # Pen_P = math.exp((line1_matched_stable_words + FAKE_COUNT) / (line1_total_stable_words + FAKE_COUNT)) 
        # Pen_R = math.exp((line2_matched_stable_words + FAKE_COUNT) / (line2_total_stable_words + FAKE_COUNT) - 1) 

        # Pen_P = A * Pen_P  * THRESHOLD + THRESHOLD
        # Pen_R = Pen_R  * THRESHOLD + THRESHOLD

        # print("orignal_P:", P, " orignal_R:", R)
        # print("before:", " Pen_P:",Pen_P,"    Pen_R:",Pen_R)
        # Pen_P =  w * (Pen_P ** THRESHOLD)
        # Pen_R =  w * (Pen_R ** THRESHOLD)

        Pen_P = (1 - Pen_P)
        Pen_R = (1 - Pen_R)

        # #if line1_total_stable_words == 0:
        #     Pen_P = 1
        #     Pen_P_ = 1
        # else:
        #     Pen_P = (line1_matched_stable_words + FAKE_COUNT) / (line1_total_stable_words + FAKE_COUNT)
        #     Pen_P_ = (line1_matched_stable_words + FAKE_COUNT_) / (line1_total_stable_words + FAKE_COUNT_)
        
        # if line2_total_stable_words == 0:
        #     Pen_R = 1
        #     Pen_R_ = 1
        # else:
        #     Pen_R = (line2_matched_stable_words + FAKE_COUNT) / (line2_total_stable_words + FAKE_COUNT)
        #     Pen_R_ = Pen_R_ = (line2_matched_stable_words + FAKE_COUNT_) / (line2_total_stable_words + FAKE_COUNT_)
        # # if Pen_P != 1 or Pen_R != 1:
        #      print("PENALTY:",Pen_P, Pen_R,P,R)
        # print("orignal_P:",P)
        # print("orignal_R:",R)
        # print("after:", " Pen_P:",Pen_P,"    Pen_R:",Pen_R)
        # print(FAKE_COUNT_,"  Pen_P_:",Pen_P_,"    Pen_R_:",Pen_R_)
        P, R = P * (Pen_P), R * (Pen_R)

        num_chunks = a.numChunks  # `ch` in Meteor formula
        # print("num_chunks:", num_chunks)
        num_matched_words = (a.line1Matches + a.line2Matches) / 2.0  # `m` in Meteor formula
        try:
            F_mean = (P * R) / (self.alpha * P + (1 - self.alpha) * R)
            # print("changdu candidate:", len(candidate))
            # Pen = self.gamma * ((num_chunks / num_matched_words) * (1/len(candidate))) ** self.beta
            Pen = self.gamma * (num_chunks / num_matched_words) ** self.beta
            # print("Pen:",Pen)
        except Exception as e:
            # print("X"*60)
            # print(reference)
            # print(candidate)
            # print(P)
            # print(R)
            return 0
            # print(F_mean)
            
        score = (1 - Pen) * F_mean
        # score = F_mean
        # print("final_score:", score)
        if verbose:
            print("Statistics:")
            print("P = ", P, ", R = ", R, ", ch = ", num_chunks, ", m = ", num_matched_words,
                  ", Pen = ", Pen,  " , F_mean = ", F_mean)
        return score

    def file_meteor(self, candidate_file, reference_file, output_file, norm=True):
        with codecs.open(candidate_file, "r", encoding="utf-8") as f1:
            with codecs.open(reference_file, "r", encoding="utf-8") as f2:
                with codecs.open(output_file, "w", encoding="utf-8") as f3:
                    for line1, line2 in zip(f1, f2):
                        meteor = self.sentence_meteor(candidate=line1.strip(), reference=line2.strip(), norm=norm)
                        f3.write(str(meteor) + "\n")


def unit_test():
    metric = Meteor(weights="1,1,1,1", hyper="0.9,3.0,0.5,1.0", lang="EN")
    metric.function_words = metric.function_words | {"I", "you", "."}
    # A little hack.

    # The following examples come from wiki: https://en.wikipedia.org/wiki/METEOR

    # For this example, there are 2 possible alignments (a) and (b). See the figure in wiki for details.
    # Wiki said alignment (a) is prefered, and the Meteor score is:
    # Score: 0.5000 = Fmean: 1.0000 * (1 - Penalty: 0.5000)
    # Fmean: 1.0000 = 10 * Precision: 1.0000 * Recall: 1.0000 / （Recall: 1.0000 + 9 * Precision: 1.0000）
    # Penalty: 0.5000 = 0.5 * (Fragmentation: 1.0000 ^ 3)
    # Fragmentation: 1.0000 = Chunks: 6.0000 / Matches: 6.0000

    # However, my program produces alignment (b), which is correct in fact.
    # The detailed output is:
    # P = 1.0, R = 1.0, ch = 3, m = 6.0, Pen = 0.0625, F_mean = 1.0
    # Meteor score:  0.9375
    res = metric.sentence_meteor(candidate="on the mat sat the cat", reference="the cat sat on the mat", norm=True)
    print("Meteor score: ", res)

    # Score: 0.9977 = Fmean: 1.0000 * (1 - Penalty: 0.0023)
    # Fmean: 1.0000 = 10 * Precision: 1.0000 * Recall: 1.0000 / （Recall: 1.0000 + 9 * Precision: 1.0000）
    # Penalty: 0.0023 = 0.5 * (Fragmentation: 0.1667 ^ 3)
    # Fragmentation: 0.1667 = Chunks: 1.0000 / Matches: 6.0000
    res = metric.sentence_meteor(candidate="the cat sat on the mat", reference="the cat sat on the mat", norm=True)
    print("Meteor score: ", res)

    # Score: 0.9654 = Fmean: 0.9836 * (1 - Penalty: 0.0185)
    # Fmean: 0.9836 = 10 * Precision: 0.8571 * Recall: 1.0000 / （Recall: 1.0000 + 9 * Precision: 0.8571）
    # Penalty: 0.0185 = 0.5 * (Fragmentation: 0.3333 ^ 3)
    # Fragmentation: 0.3333 = Chunks: 2.0000 / Matches: 6.0000
    res = metric.sentence_meteor(candidate="the cat was sat on the mat", reference="the cat sat on the mat", norm=True)
    print("Meteor score: ", res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/", help="Data folder.")
    parser.add_argument("--test_set", type=str, default="pos_neg", help="Test set file name. (Input)")
    parser.add_argument("--paraphrase_file", type=str, default="filtered_paraphrase_table",
                        help="Paraphrase file.")
    parser.add_argument("--paraphrase_file_format", type=str, default="pair",
                        help="Format of paraphrase file. Could be either 'pair' or 'clique'.")
    parser.add_argument("--paraphrase_invariant", type=str, default="sorted_copy_words_freq0.6_20000000",
                        help="Invariant words during paraphrasing, i.e.: copy words_wmt15111")
    parser.add_argument("--function_words", type=str, default="english.function.words",
                        help="Function words list")
    # parser.add_argument("--ner_copy", type = str, default = "ner_copy_words_quora")
    parser.add_argument("--ner_copy", type=str, default="ner_copy_words_wmt16")
    args = parser.parse_args()

    metric = Meteor(weights="1.0,0.6,0.8,0.6,0", hyper="0.85,0.2,0.6,0.75", lang="EN",
                            paraphrase_invariant=os.path.join(args.data_dir, args.paraphrase_invariant),
                            paraphrase_file=os.path.join(args.data_dir, args.paraphrase_file),
                            paraphrase_file_format=args.paraphrase_file_format,
                            function_words=os.path.join(args.data_dir, args.function_words),
                            ner_copy=os.path.join(args.data_dir, args.ner_copy))

    metric.sentence_meteor(reference="I saw Monika in Evita three times and she was definitely an inspiration to me.",
                           candidate="I saw Monica in Evita three times and it was definitely inspirational to me.",
                           norm=True, verbose=True)
