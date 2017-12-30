from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import random
# Change the path according to your system
def NERreplacement(text, classifier_path = '/Users/michaelqi/Downloads/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz', ner_path = '/Users/michaelqi/Downloads/stanford-ner-2017-06-09/stanford-ner.jar'):
	stanford_classifier = classifier_path
	stanford_ner_path = ner_path

	# Creating Tagger Object
	st = StanfordNERTagger(stanford_classifier, stanford_ner_path, encoding='utf-8')


	tokenized_text = word_tokenize(text)
	classified_text = st.tag(tokenized_text)	
	NER =[]
	for e in classified_text:
		if e[1] != 'O':
			NER.append(e[0])
	return NER

def replacement(titleText, commentText):
	titleNER = NERreplacement(titleText)
	commentNER = NERreplacement(commentText)
	commentText = commentText.split()
	final_string = " "
	for e in commentText:
		if e in commentNER:
			final_string+=" "+titleNER[random.randint(0, len(titleNER)-1)]
		else:
			final_string+=" "+e
	print(final_string)
	return final_string

#test
#replacement('While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.','While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.' )