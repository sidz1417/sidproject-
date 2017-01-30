import pandas as pd
import string 
from string import digits 
from nltk.tokenize import sent_tokenize
import os

def preprocess(sentence):
	sentence=sentence.lower()
	sentence_list=sent_tokenize(sentence)
	sentence_list=[''.join(c for c in s if c not in string.digits and c not in string.punctuation) for s in sentence_list]
	sentence_list=[x for x in sentence_list if x]
	return sentence_list

location=r'amazonreviews.csv'
df=pd.read_csv(location,names=['x','y','z','a','text','rating','summary','b','c'])

review_count=2000
unsupervised_review_count=20000

test1star=review_count
train1star=review_count
test2star=review_count
train2star=review_count
test3star=review_count
train3star=review_count
test4star=review_count
train4star=review_count
test5star=review_count
train5star=review_count
unsup=unsupervised_review_count

f1=open('testset/test1star.txt','w')
f2=open('trainset/train1star.txt','w')
f3=open('testset/test2star.txt','w')
f4=open('trainset/train2star.txt','w')
f5=open('testset/test3star.txt','w')
f6=open('trainset/train3star.txt','w')
f7=open('testset/test4star.txt','w')
f8=open('trainset/train4star.txt','w')
f9=open('testset/test5star.txt','w')
f10=open('trainset/train5star.txt','w')
f11=open('unsupervised.txt','w')

for row in df.itertuples(index=True, name='Pandas'):
				#1star 
		if int(getattr(row,'rating'))==1 and test1star>0:
			f1.write(os.linesep.join(x for x in preprocess(str(getattr(row,'text')))))
			test1star-=1
		elif int(getattr(row,'rating'))==1 and train1star>0:
			f2.write(os.linesep.join(x for x in preprocess(str(getattr(row,'text')))))
			train1star-=1
				#2star
		elif int(getattr(row,'rating'))==2 and test2star>0:
			f3.write(os.linesep.join(x for x in preprocess(str(getattr(row,'text')))))
			test2star-=1
		elif int(getattr(row,'rating'))==2 and train2star>0:
			f4.write(os.linesep.join(x for x in preprocess(str(getattr(row,'text')))))
			train2star-=1
				#3star
		elif int(getattr(row,'rating'))==3 and test3star>0:
			f5.write(os.linesep.join(x for x in preprocess(str(getattr(row,'text')))))
			test3star-=1
		elif int(getattr(row,'rating'))==3 and train3star>0:
			f6.write(os.linesep.join(x for x in preprocess(str(getattr(row,'text')))))
			train3star-=1
				#4star
		elif int(getattr(row,'rating'))==4 and test4star>0:
			f7.write(os.linesep.join(x for x in preprocess(str(getattr(row,'text')))))
			test4star-=1
		elif int(getattr(row,'rating'))==4 and train4star>0:
			f8.write(os.linesep.join(x for x in preprocess(str(getattr(row,'text')))))
			train4star-=1
				#5star
		elif int(getattr(row,'rating'))==5 and test5star>0:
			f9.write(os.linesep.join(x for x in preprocess(str(getattr(row,'text')))))
			test5star-=1
		elif int(getattr(row,'rating'))==5 and train5star>0:
			f10.write(os.linesep.join(x for x in preprocess(str(getattr(row,'text')))))
			train5star-=1
				#for unsupervised training 
		else:
			f11.write(os.linesep.join(x for x in preprocess(str(getattr(row,'text')))))
			unsup-=1
