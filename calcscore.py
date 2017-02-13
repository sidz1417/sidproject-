import os 
from nltk.tokenize import sent_tokenize
import string
import cPickle as pickle  
import numpy as np 
from gensim import utils 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
from random import shuffle

f1=open("samplereviews.txt","r")
f2=open("processedreviews.txt","w")

with open("stoplist.pkl","rb") as f3:
	stops=pickle.load(f3)

def preprocess(sentence):
	sentence=sentence.lower()
	sentence_list=sent_tokenize(sentence)
	sentence_list=[''.join(c for c in s if c not in string.digits and c not in string.punctuation) for s in sentence_list]
	sentence_list=[x for x in sentence_list if x]
	return sentence_list

for line in f1:
	f2.write(os.linesep.join(x for x in preprocess(str(line))))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class taggedlinedocument(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

sources={'processedreviews.txt','PRO_SENT'}

sentences=taggedlinedocument(sources)

print "started training"

dimensions=20
model = Doc2Vec(alpha=0.025, min_alpha=0.025,size=dimensions,min_count=1,iter=10,workers=2)  # use fixed learning rate
model.build_vocab(sentences.to_array())
for epoch in range(10):
    model.train(sentences.sentences_perm())
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
	
print "finished training"

#save the model 	
model.save('sampledoc2vecmodel.doc2vec')

def predict_topic(query):		
    tokens=tokenizer.tokenize(temp)
    tokens=[w for w in tokens if not w in stops]
    
    rev_vec = dictionary.doc2bow(tokens)
    topics = sorted(lda[rev_vec],key=lambda x:x[1],reverse=True)
    if topics[0][1]>=60.0:
    	return topics[0][0]
    else:
    	return None

#example aspect ratings 
aspect_dict={"0":"screen","1":"build quality"}
aspectratings_dict={"screen":0,"build quality":0}
aspectlines_dict={"screen":0,"build quality":0}

docvecmodel = Doc2Vec.load('doc2vecmodel.doc2vec')
lda=LdaModel.load('gamelda.model')
dictionary = corpora.Dictionary.load('gamedict.dict')
with open ("logisticregression.pkl","rb") as f4:
    classifier=pickle.load(f4)

line_number=0

for line in f2:
	if predict_topic(line):
		sent_topic=aspect_dict[int(predict_topic(line))]
		sent_vector = 'PRO_SENT_' + str(line_number)
		sent_rating=classifier.predict(docvecmodel.docvecs[sent_vector])
		aspectratings_dict[sent_topic]+=sent_rating
		aspectlines_dict[sent_topic]+=1

#compute average rating for each aspect 
for category in aspectlines_dict:
	avg_rating=aspectratings_dict[category]/aspectlines_dict[category]
	print "%s : %f" %(category,avg_rating)











