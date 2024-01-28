import nltk
from nltk.corpus import wordnet
import re
from functools import partial

nltk.download('wordnet')

SLANG_PATH = 'processing/slang.txt'


def preprocess(dicti):
    
    def replaceElongated(word):
        
        repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        repl = r'\1\2\3'
        if wordnet.synsets(word):
            return word
        repl_word = repeat_regexp.sub(repl, word)
        if repl_word != word:
            return replaceElongated(repl_word)
        else:
            return repl_word


    with open(SLANG_PATH) as file:
        slang_map = dict(map(str.strip, line.partition('\t')[::2])
                         for line in file if line.strip())

    slang_words = sorted(slang_map, key=len, reverse=True)
    regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))))
    replaceSlang = partial(regex.sub, lambda m: slang_map[m.group(1)])

    
    dicti['text'] = dicti['text'][0].replace('\n', ' ')
    dicti['text'] = replaceSlang(replaceElongated(dicti['text']))
    
    dicti['objects'] = set([el[0] for el in dicti['objects']])
    
    for label in ['Photo caption', 'Photography', 'Font', 'Text', 'Internet meme']:
        if label in dicti['labels']:
            dicti['labels'].remove(label)
    return dicti
