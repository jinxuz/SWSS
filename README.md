# Semantically Weighted Sentence Similarity (SWSS)

The implementation of [Incorporate Semantic Structures into Machine Translation Evaluation via UCCA](https://arxiv.org/abs/2010.08728).

## Structure

- model/						
  from TUPA pretrained model
- vocab/						 
  from TUPA pretrained model
- data/						    
  for data storage
- align.py						
  modified scripts from UCCA
- meteor5.py				
  a self-implemented Meteor (users may want to use other Meteor instead)
- process.py                 
  the functions to modify original scores
- uccamt.py
  the main functions (modify this to get model run on specific dataset)

## Dependencies

- UCCA 1.2.3
- TUPA 1.4.1
- nltk 3.4.3
- Spacy 2.1.3 (higher version may not work) with datasets Spacy en core web md 2.1.0 & Spacy en core web lg 2.1.0

### Download

[TUPA BiLSTM Model](https://github.com/huji-nlp/tupa/releases/download/v1.3.10/ucca-bilstm-1.3.10.tar.gz)

[Spacy en core web md](https://github.com/explosion/spacy-models/releases/tag/en_core_web_md-2.1.0)

[Spacy en core web lg](https://github.com/explosion/spacy-models/releases/tag/en_core_web_lg-2.1.0)

 





