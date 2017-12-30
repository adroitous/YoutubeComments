# requires the video_id to be read in as first command line arg
from controller import ensemble_run, parse_video_dict
from random import sample
import json

def top(category_id, video_dict, top_comments):
    with open('category_dict.json', 'r') as f:
        category_dict = json.load(f)
    videos = [video_info[0] for video_info in category_dict[category_id]]
    videos = [videos[i] for i in sample(range(len(videos)), 1)]
    top_comments = [c.strip() for c in top_comments if not c.isspace() and c != ""]
    video_dict = parse_video_dict()
    best = ensemble_run(video_dict, top_comments, videos)
    return best

if __name__ == '__main__':
    '''
    video_id = sys.argv[1]
    with open('comment_generated.json', 'r') as f:
        coms = json.load(f)
        # coms = [c.split(':')[1] for c in coms if ':' in c]
        coms = [c for c in coms if not c.isspace() and c != ""]
        for c in coms:
            c.strip()
        video_dict = parse_video_dict()
        best = ensemble_run(video_dict, coms, video_id)
        print('Best comment is: {}'.format(best))
        '''
    video_dict = parse_video_dict()
    with open('comment_generated.json', 'r') as f:
        coms = json.load(f)
        # coms = [c.split(':')[1] for c in coms if ':' in c]
        coms = [c for c in coms if not c.isspace() and c != ""]
        for c in coms:
            c.strip()
    best = top("24", video_dict, coms)
    print('Best comment is: {}'.format(best))
