import numpy as np
import pandas as pd
import json
from itertools import combinations
from collections import namedtuple

comment = namedtuple('comment', 'content, likes')

def read_comments():

    UScomments = pd.read_csv('youtube/UScomments.csv', encoding = 'utf8', error_bad_lines = False)
    del(UScomments['replies'])

    video_dict = {}
    for index, row in UScomments.iterrows():
        video_dict.setdefault(row['video_id'], []).append(comment(row['comment_text'], row['likes']))

    return (video_dict)


video = namedtuple('video', 'video_id, title, likes')

def read_cat():

    columns = ['category_id', 'video_id', 'title', 'likes']
    USvideos = pd.read_csv('youtube/USvideos.csv',usecols = columns, encoding = 'utf8', error_bad_lines = False)

    category_dict = {}
    for index, row in USvideos.iterrows():
        category_dict.setdefault(row['category_id'], []).append(video(row['video_id'], row['title'], row['likes']))

    return(category_dict)


if __name__ == '__main__':

    category_dict = read_cat() 
    with open('category_dict.json', 'w') as f:
        json.dump(category_dict, f)

    video_dict = read_comments() 
    with open('video_dict.json', 'w') as f:
        json.dump(video_dict, f)
