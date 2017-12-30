import json
import sys
from jsonTest import process_text
from LM import preprocess, generate_sent


with open('category_dict.json') as json_data:
    category_dict = json.load(json_data)

with open('video_dict.json') as json_data:
    video_dict = json.load(json_data)

# n is the factor in which we decrease the smoothing in.
# best is 1.
def category_generator_likeSmoothing(name, n = 1):
    category_corpus = []
    for i in range(len(category_dict[name])):
        video = category_dict[name][i]
        print(video_dict[video[1]])
        try:
            while int(video_dict[video[1]])>=0:
                category_corpus.append(video_dict[video[0]])
                video_dict[video[1]] = int(video_dict[video[1]])-n
        except KeyError:
            1+1
    return category_corpus

def category_generator(name):
    category_corpus = []
    for i in range(len(category_dict[name])):
       	video = category_dict[name][i]
        try:
            category_corpus.append(video_dict[video[0]])
        except KeyError:
            1 + 1
    return category_corpus

def corpus_processing(category_corpus):
    corpus =[]
    for e in category_corpus:
        for a in e:
            corpus.append(a[0])
    return corpus

def corpus_generation(name):
    corpus = corpus_processing(category_generator(name))
    with open('category_corpus.txt', 'w') as f:
        for e in corpus:
            f.write(str(e) +'\n')


if __name__ == '__main__':
    # requires a category name to be passed in as the first command line arguement
    # ex. 'entertainment'
    category_name = sys.argv[1]
    corpus_generation(process_text(category_name))
    corpus = preprocess()
    final_text = [generate_sent(5, corpus)for i in range(100)]
    with open('comment_generated.json', 'w') as f:
        json.dump(final_text, f)
