from sklearn.feature_extraction.text	import TfidfVectorizer
from nltk.stem import SnowballStemmer
from sklearn.datasets import load_mlcomp,fetch_20newsgroups
from sklearn.cluster import KMeans
from scipy import linalg
import os
class StemmedTfidfVectorizer(TfidfVectorizer):

	def	build_analyzer(self):

		analyzer=super(TfidfVectorizer, self).build_analyzer()

		return lambda doc: (english_stemmer.stem(w)	for	w in analyzer(doc))
english_stemmer = SnowballStemmer('english')

vectorizer	= StemmedTfidfVectorizer(min_df=10,stop_words='english',decode_error='ignore')
MLCOMP_DIR = os.path.join(os.getcwd())
print(MLCOMP_DIR)
groups=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardhware','comp.sys.mac.hardware','comp.windows.x','sci.space']
#data = load_mlcomp("20news-18828",mlcomp_root=MLCOMP_DIR)
#train_data = fetch_20newsgroups(subset='train')
#test_data = fetch_20newsgroups(subset='test')
test_data = load_mlcomp("20news-18828","test",mlcomp_root=MLCOMP_DIR)
train_data = load_mlcomp("20news-18828","train",mlcomp_root=MLCOMP_DIR)

vectorized=vectorizer.fit_transform(train_data.data)
num_samples,num_features=vectorized.shape
print (num_samples,num_features)
num_clusters=50
km=KMeans(n_clusters=num_clusters,init='random',n_init=1,verbose=1)

km.fit(vectorized)
new_post=  "Disk	drive	problems.	Hi,	I	have	a	problem	with	my hard	disk. After	1	year	it	is	working	only	sporadically	now. I	tried	to	format	it,	but	now	it	doesn't	boot	any more. Any	ideas?	Thanks."
new_post_vec = vectorizer.transform([new_post])
new_post_label=km.predict(new_post_vec)[0]
print (new_post_label)
similar_indices=(km.labels_==new_post_label).nonzero()[0]
similar=[]
for i in similar_indices:
	dist=linalg.norm((new_post_vec - vectorized[i]).toarray())
	similar.append((dist,dataset.data[i]))
similar=sorted(similar)
print(len(similar))

