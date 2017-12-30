# this controls all classifiers by feeding each of them data.
# It then aggregates each classifier's vote to decide which video will recieve the most likes.
import json
from collections import namedtuple, Counter
from sentiment_classifier import sent_classifier, sent_test
from cooccurrence import cooccurrence_classifier, cooccurrence_test
from itertools import combinations
import time
import random


comment = namedtuple('comment', 'content, likes')
video = namedtuple('video', 'id, title, like_count, cat_id')

def ensemble_test(training_video_dict, testing_video_dict, video_ids):
    classifier_sent = sent_classifier(training_video_dict, video_ids)
    classifier_cooccur = cooccurrence_classifier(training_video_dict, video_ids)

    test_pre = []
    for vid in video_ids:
        test_pre += testing_video_dict[vid]
    # print(test_pre)
    test_set = [(combo[0].content, combo[1].content, 0 if combo[0].likes > combo[1].likes else 1) for combo in list(combinations(test_pre, 2)) if combo[0].likes != combo[1].likes]
    sent_classified = []
    cooccur_classified = []
    together_classified = []
    correct = []
    test_start = time.time()
    for pair in test_set:
        s = sent_test(classifier_sent, pair[0], pair[1])
        sent_classified.append(s)
        c = cooccurrence_test(training_video_dict, video_ids, classifier_cooccur, pair[0], pair[1])
        cooccur_classified.append(c)
        t = combine_classifiers(pair[0], pair[1], classifier_sent, classifier_cooccur, training_video_dict, video_ids)
        together_classified.append(t)
        correct.append(pair[2])
        # print('sent {}, cooc {}, correct {}'.format(s, c, pair[2]))
    test_end = time.time()

    f1s = []
    for guess in [sent_classified, cooccur_classified, together_classified]:
        tp = sum([1 for i in range(0, len(correct), 1) if correct[i]==1 and guess[i]==1])
        tn = sum([1 for i in range(0, len(correct), 1) if correct[i]==0 and guess[i]==0])
        fp = sum([1 for i in range(0, len(correct), 1) if correct[i]==0 and guess[i]==1])
        fn = sum([1 for i in range(0, len(correct), 1) if correct[i]==1 and guess[i]==0])
        accuracy = (tp + tn)/float(tp + tn + fp + fn)
        prec_p = tp / float(tp + fp) if tp + tp != 0 else 0
        prec_n = tn / float(tn + fn) if tn + fn != 0 else 0
        rec_p = tp / float(tp + fn) if tp + fn != 0 else 0
        rec_n = tn / float(tn + fp) if tn + fp != 0 else 0
        f1_p = (2*prec_p*rec_p) / float(prec_p + rec_p) if prec_p + rec_p != 0 else 0
        f1_n = (2*prec_n*rec_n) / float(prec_n + rec_n) if prec_n + rec_n != 0 else 0
        f1s.append(f1_p)
    print('f1 for sent = {}, f1 for cooccur = {}, f1 overall = {}'.format(f1s[0], f1s[1], f1s[2]))

    print("It took {}m to classify {} combinations.".format((test_end - test_start)/60.0, len(test_set)))

# use this when comparing to get to best generated comment
def ensemble_run(video_dict, generated, video_ids):
    for vid in video_ids:
        print('{} has {} comments to analyze.'.format(vid, len(video_dict[vid])))
    classifier_sent = sent_classifier(video_dict, video_ids)
    classifier_cooccur = cooccurrence_classifier(video_dict, video_ids)

    # generated = generated[:16]
    matchups = list(combinations(generated, 2))
    winners = [m[0] if combine_classifiers(m[0], m[1], classifier_sent, classifier_cooccur, video_dict, video_ids) == 0 else m[1] for m in matchups]
    counts = Counter(winners)
    print(counts.most_common())
    return counts.most_common(1)[0][0]

def combine_classifiers(c1, c2, sent, cooc, video_dict, video_ids):
    s = sent_test(sent, c1, c2)
    c = cooccurrence_test(video_dict, video_ids, cooc, c1, c2)
    return c

def parse_video_dict():
    with open('video_dict.json', 'r') as f:
        video_dict_raw = json.load(f)
        video_dict = {}
        for key in video_dict_raw:
            # if key in ['TPnuT2TLvLQ', '_qb4_uvYSG0', 'YYwB63YslbA', 'CiHV9oFXFzY', 'j-JOG2mUt0c']:
            video_dict[key] = []
            for com_thing in video_dict_raw[key]:
                try:
                    int(com_thing[1])
                    video_dict[key].append(comment(str(com_thing[0]), int(com_thing[1])))
                except ValueError:
                    1 + 1
    return video_dict

def split_video_dict(dic):
    seed = 0.5
    train = {}
    test = {}
    for vid in dic:
        coms = dic[vid]
        random.shuffle(coms, lambda: seed)
        train[vid] = coms[:int(.7*len(coms))]
        test[vid] = coms[int(.7*len(coms)):]
    return (train, test)

if __name__ == '__main__':
    video_dict = parse_video_dict()
    splits = split_video_dict(video_dict)
    train_set = splits[0]
    test_set = splits[1]
    ensemble_test(train_set, test_set, ['cLdxuaxaQwc'])
