import tensorlayer
import tensorflow
import numpy as np
import jieba
import os
import io


def segment(raw, dict_path='data/vocabularies', outputfile=None):
    dicts = os.listdir(dict_path)
    for dict in dicts:
        jieba.load_userdict(f'{dict_path}/{dict}')
    cut = jieba.cut(raw)
    if outputfile is not None:
        with open(outputfile, 'w', encoding='utf8') as f:
            f.write(' '.join(cut))
    return cut


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    dictionary = {}
    reverse_dictionary = {}
    id = 0
    embedding_vec = []
    for line in fin:
        tokens = line.rstrip().split(' ')
        dictionary.update({tokens[0]: id})
        reverse_dictionary.update({id: tokens[0]})
        word_vec = list(map(float, tokens[1:]))
        embedding_vec.append(word_vec)
        id += 1
    return dictionary, reverse_dictionary, np.array(embedding_vec, dtype=np.float32)
