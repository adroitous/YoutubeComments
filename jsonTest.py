import json

def process_text(str):
	with open('youtube/US_category_id.json') as f:
		category_match = json.load(f)
	id_title = {}
	for e in category_match['items']:
		snippet = e['snippet']
		id_title[snippet['title']] = e['id']
	return id_title[str]
