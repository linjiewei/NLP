import io
import numpy as np
import tensorlayer as tl
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
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


if __name__ == '__main__':
    dictionary, reverse_dictionary, embedding_vec = load_vectors('./data/4_fasttext_model.vec')

    # start_time = time.time()
    # tl.visualize.tsne_embedding(embedding_vec, reverse_dictionary, 500, second=5, saveable=True, name='embedding_tsne')
    # print(f'Plot t-sne took {time.time()-start_time} sec')

    print('Save model, dictionary, and reverse_dictionary!')
    tl.files.save_any_to_npy(save_dict={
        'dictionary': dictionary,
        'reverse_dictionary': reverse_dictionary,
        'embedding_vec': embedding_vec
    }, name='model.npy')
