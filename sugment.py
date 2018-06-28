import jieba
import os
import time


def segment(filename='data/1_Tianlongbabu_tw.txt', dict_path='data/vocabularies', outputfile=None):
    dicts = os.listdir(dict_path)
    for dict in dicts:
        jieba.load_userdict(f'{dict_path}/{dict}')
    with open(filename, 'r', encoding='utf8') as f:
        raw = f.read()
    cuted = jieba.cut(raw)
    if outputfile is not None:
        with open(outputfile, 'w', encoding='utf8') as f:
            f.write(' '.join(cuted))
    return cuted


if __name__ == '__main__':
    start_time = time.time()
    segment(filename='data/2_Tianlongbabu_cleaned.txt',
            dict_path='data/vocabularies',
            outputfile='3_Tianlongbabu_segmented'
    )
    print(f'Total took {time.time()-start_time} second')
