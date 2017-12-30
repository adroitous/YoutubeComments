# CSCI375_Top_Comment
By: Will Burford, Vincent Lin, Haoyu Sheng, Tongyu Zhou

Our goal is to develop a YouTube comment generator, which will generate the top comment on a given YouTube video. On the way to this goal, we will create a classifier which will quantify the probability of a comment getting the most likes on a given video.

## Running the Program
To setup your local environment, run:
```
$ make
```
This will create a virtual environment in, stored in `venv`, and install all dependencies listed in `requirements.txt`. It will also extract the relevant files from `youtube.zip` into `youtube/` (these files are `US_category_id.json`, `UScomments.csv`, and `USvideos.csv`). After this, you must activate the virtual environment by running:
```
$ source venv/bin/activate
```
You must run this last command every time you start a new terminal session so that you will be in the virtual environment and have the correct package versions!

If the first make command fails when trying to create a virtual environment, try installing the packages globally and extracting the data manually by running:
```
pip3 install -r requirements.txt
make extract-zip
```
To process the files so that they are ready for the program to ingest, run:
```
make read
```

Then, you will be able to generate potential comments for a the entertainment category using the command (this can take quite a while):
```
make generate
```
It is possible to change which category of videos to source the generation from by going into the `Makefile` and changing which category string is passed to `Generator.py` when it is run.

Now, the classifier can be trained and the best of the generated comments selected by running:
```
make run
```

## Classifier Testing
We used a traditional 70-30 split to validate our classifier. To see the f1 score resulting from this, run:
```
make test
```
