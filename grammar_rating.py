from nltk import CFG
import nltk

corpus = open("rand_sent.txt", 'r').read().splitlines()
for sent in corpus:
    sent = sent.split()
    sent = [nltk.word_tokenize(s) for s in sent]
    sent = [nltk.pos_tag(s) for s in sent]
    parser = nltk.ChartParser(groucho_grammar, nltk.parse.BU_STRATEGY)
    trees = parser.nbest_parse(sent, trace=2)
    print(trees)
