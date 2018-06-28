import re
import subprocess
import os


def basic_clean(string):
    string = re.sub(r'\n', ' ', string)
    string = re.sub(r'“', '"', string)
    string = re.sub(r'”', '"', string)
    string = re.sub(r'\[', '', string)
    string = re.sub(r'\]', '', string)
    string = re.sub(r'‘', '"', string)
    string = re.sufab(r'’', '"', string)
    string = re.sub(r' {2,}', ' ', string)
    return string


def chinese_trans(string, preset='s2t', opencc_path='opencc/bin/opencc.exe'):
    with open('temp.txt', 'w', encoding='utf8') as f:
        f.write(string)
    subprocess.run([opencc_path, '-c', preset, '-i', 'temp.txt', '-o', 'temp_complete.txt'])
    with open('temp_complete.txt', 'r', encoding='utf8') as f:
        result = f.read()
    os.remove('temp.txt')
    os.remove('temp_complete.txt')
    return result


if __name__ == '__main__':
    with open('data/1_Tianlongbabu_tw.txt', 'r', encoding='utf8') as f:
        raw = f.read()
        processed = basic_clean(raw)
    with open('data/2_Tianlongbabu_cleaned.txt', 'w', encoding='utf8') as f:
        f.write(processed)
