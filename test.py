'''
sources={'testset/test1star.txt':'TEST_ONE','trainset/train1star.txt':'TRAIN_ONE','testset/test2star.txt':'TEST_TWO','trainset/train2star.txt':'TRAIN_TWO','testset/test3star.txt':'TEST_THREE','trainset/train3star.txt':'TRAIN_THREE','testset/test4star.txt':'TEST_FOUR','trainset/train4star.txt':'TRAIN_FOUR','testset/test5star.txt':'TEST_FIVE','trainset/train5star.txt':'TRAIN_FIVE','unsupervised.txt':'TRAIN_UNS'}	

def linecount(sources):
	for source,prefix in sorted(sources.items()):
		lines=0
		with open(source,"r") as f:
			for line in f:
				lines+=1
		print "%r : %d" %(prefix,lines) 

#linecount(sources)
'''

import pandas as pd

location=r'amazonreviews.csv'
df=pd.read_csv(location,names=['x','y','z','a','text','rating','summary','b','c'])
rating1=0
rating2=0
rating3=0
rating4=0
rating5=0


for row in df.itertuples(index=True, name='Pandas'):
	if int(getattr(row,'rating'))==1:
		rating1+=1
	elif int(getattr(row,'rating'))==2:
		rating2+=1
	elif int(getattr(row,'rating'))==3:
		rating3+=1
	elif int(getattr(row,'rating'))==4:
		rating4+=1
	else:
		rating5+=1

print "1star: %d" %(rating1)
print "2star: %d" %(rating2)
print "3star: %d" %(rating3)
print "4star: %d" %(rating4)
print "5star: %d" %(rating5)

print rating1+rating2+rating3+rating4+rating5

 