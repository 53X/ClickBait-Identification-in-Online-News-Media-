'''

Preparing the  data from the .jsonl files 
present in the dataset.

'''
from ast import literal_eval

def get_data():

	samples = []
	labels = []

	with open('/home/pranay/clickbait_data/truth.jsonl', mode ='r', encoding = 'utf-8', errors = 'ignore') as annotation:

		for row in annotation:
			row =  literal_eval(row)
			labels.append(1) if row['truthClass'] == 'clickbait' else labels.append(0)

	print("Clickbait samples collected : {}".format(labels.count(1)))
	print("Non-Clickbait samples collected : {}".format(labels.count(0)))		

	with open('/home/pranay/clickbait_data/instances.jsonl', mode='r', encoding='utf-8', errors='ignore') as content:

		for row in content:
			text = literal_eval(row)
			samples+=text['postText']

	print("Gathered {} sample texts".format(len(samples)))	

	return samples, labels	

if __name__ == '__main__':

	get_data()
