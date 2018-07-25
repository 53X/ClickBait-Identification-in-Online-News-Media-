'''

Preparing the  data from the .jsonl files 
present in the dataset.

'''

text = []
labels = []

def get_data():

	with open('/home/pranay/clickbait_data/truth.jsonl', mode ='r', encoding = 'utf-8', errors = 'ignore') as file:

		for row in file:
			label = row.split(' ')[-5].replace('"','').rstrip(',')
			labels.append(1) if label == 'clickbait' else labels.append(0)

	print("Clickbait samples collected : {}".format(labels.count(1)))
	print("Non-Clickbait samples collected : {}".format(labels.count(0)))		

	