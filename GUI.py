from tkinter import *
import json
from collections import namedtuple
from cooccurrence import cooccurrence_classifier
from sentiment_classifier import sent_classifier
from LM import preprocess, generate_sent
from Generator import corpus_generation
from jsonTest import process_text
from controller import combine_classifiers, parse_video_dict
from markovchain import generate_markov
from top_comment import top
from random import sample

comment = namedtuple('comment', 'content, likes')

#tkinter doesn't have certain emojis, so when it does comments it might skip some of the emojis
class classifier:
    
    prev_id = ''
    cooccur = None
    sent = None

    def __init__(self, master, video_dict):

        ## First frame contains the intro and the textboxes 
        frame_id = Frame(master)
        frame_id.pack()         

        lab = Label(frame_id, text = "Which comment would get more likes?")
        lab.pack()

        lab_id = Label(frame_id, text = "Input your video ID")
        lab_id.pack(side = LEFT)

        e_id = Entry(frame_id)
        e_id.insert(END, 'cLdxuaxaQwc')
        e_id.pack(side = LEFT)

        frame = Frame(master)
        frame.pack()
        
        label = Label(frame, text = "Input your comments")
        label.pack()

        ## Entry allows the backend to receive user input
        e1 = Entry(frame)
        e1.pack(side = LEFT)

        e2 = Entry(frame)
        e2.pack(side = LEFT)

        ## Message displays which comment is better
        message = StringVar()
        Label(master, textvariable = message).pack()

        def compare():

            if e_id.get() != self.prev_id:
                self.sent = sent_classifier(video_dict, [e_id.get()])
                self.cooccur = cooccurrence_classifier(video_dict, [e_id.get()])
                self.prev_id = e_id.get()
                
            choice = combine_classifiers(e1.get(), e2.get(), self.sent, self.cooccur, video_dict, [e_id.get()]) + 1
            string = "Based on our SCIENTIFIC CALCULATION " + str(choice) + " is better"
            message.set(string)

        ## This frame contains all the buttons
        frame_buttons = Frame(master)
        frame_buttons.pack()

        quit_button = Button(
            frame_buttons, text = "QUIT", fg = "red", command = frame_buttons.quit
            )
        quit_button.pack(side = LEFT)

        compare_button = Button(frame_buttons, text = "Compare", command = compare)
        compare_button.pack(side=LEFT)

class generator:
    
    prev_id = None
    video_dict = parse_video_dict()

    def __init__(self, master):
        
        ## This frame contains title
        frame_title = Frame(master)
        frame_title.pack()

        title = Label(frame_title, text = "Title: ")
        title.pack(side = LEFT)

        e1 = Entry(frame_title)
        e1.pack(side = LEFT)

        ## This frame contains tags
        frame_tags = Frame(master)
        frame_tags.pack()

        tags = Label(frame_tags, text = "Tags: ")
        tags.pack(side = LEFT)

        e2 = Entry(frame_tags)
        e2.pack(side = LEFT)

        ## This frame contains category
        frame_category = Frame(master)
        frame_category.pack()

        category = Label(frame_category, text = "Category: ")
        category.pack(side = LEFT)

        e3 = Entry(frame_category)
        e3.pack(side = LEFT)

        ## Message allows us to return the generated message
        message = StringVar()
        Label(master, textvariable = message).pack()

        def generate():
            ## do things here
            ## Use e1.get, e2.get to get user input
            ## Use set to set output
            category = str(process_text(e3.get()))
            final_text = generate_markov()
            #top_comment = top(category, self.video_dict, final_text)
            text = final_text[sample(range(len(final_text)), 1)[0]]
            print(text)
            message.set(text)

        ## This frame contains the buttons
        frame_buttons = Frame(master)
        frame_buttons.pack()

        quit_button = Button(
            frame_buttons, text = "QUIT", fg = "red", command = frame_buttons.quit
            )
        quit_button.pack(side = LEFT)

        generate_button = Button(frame_buttons, text = "Generate", command = generate)
        generate_button.pack(side = LEFT)

if __name__ == '__main__':
    
    video_dict = parse_video_dict()          
    root = Tk()  
    input_raw = input("classify or generate: ")
    if str(input_raw) == 'generate':
        app = generator(root)
    elif str(input_raw) == 'classify':
        app = classifier(root, video_dict) 
    else:
        while str(input_raw) != 'generate' and str(input_raw) != 'classify':
            input_raw = input("classify or generate: ")
            if str(input_raw) == 'generate':
                app = generator(root)
            elif str(input_raw) == 'classify':
                app = classifier(root, video_dict)  
    root.mainloop()
    root.destroy()

    root = Tk()
    app = generator(root)
    root.mainloop()
    root.destroy()
