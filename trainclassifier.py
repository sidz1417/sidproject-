from gensim.models import Doc2Vec
#import logging
import numpy
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
import cPickle as pickle 

print "Finished importing"

model = Doc2Vec.load('doc2vecmodel.doc2vec')

'''
f=open("doc2veclabels.txt","w")
for label,vec in sorted(model.docvecs.doctags.items()):
    print >>f,label 
'''

test1=10593
test2=11771
test3=10536
test4=10899
test5=7677

train1=4375
train2=1019
train3=10058
train4=9806
train5=9961

dimensions=20

trainsize=train1+train2+train3+train4+train5
testsize=test1+test2+test3+test4+test5

train_arrays = numpy.zeros((trainsize, dimensions))
train_labels = numpy.zeros(trainsize)
test_arrays = numpy.zeros((testsize, dimensions))
test_labels = numpy.zeros(testsize)

print "forming train array...."

for i in xrange(train1):
    prefix_train_1star = 'TRAIN_ONE_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_1star]
    train_labels[i] = 1

for i in xrange(train2):
    prefix_train_2star = 'TRAIN_TWO_' + str(i)
    train_arrays[i+train1] = model.docvecs[prefix_train_2star]
    train_labels[i+train1] = 2

for i in xrange(train3):
    prefix_train_3star = 'TRAIN_THREE_' + str(i)
    train_arrays[i+train2] = model.docvecs[prefix_train_3star]
    train_labels[i+train2] = 3

for i in xrange(train4):
    prefix_train_4star = 'TRAIN_FOUR_' + str(i)
    train_arrays[i+train3] = model.docvecs[prefix_train_4star]
    train_labels[i+train3] = 4

for i in xrange(train5):
    prefix_train_5star = 'TRAIN_FIVE_' + str(i)
    train_arrays[i+train4] = model.docvecs[prefix_train_5star]
    train_labels[i+train4] = 5


print "forming test array...."

for i in xrange(test1):
    prefix_test_1star = 'TEST_ONE_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_1star]
    test_labels[i] = 1

for i in xrange(test2):
    prefix_test_2star = 'TEST_TWO_' + str(i)
    test_arrays[i+test1] = model.docvecs[prefix_test_2star]
    test_labels[i+test1] = 2

for i in xrange(test3):
    prefix_test_3star = 'TEST_THREE_' + str(i)
    test_arrays[i+test2] = model.docvecs[prefix_test_3star]
    test_labels[i+test2] = 3

for i in xrange(test4):
    prefix_test_4star = 'TEST_FOUR_' + str(i)
    test_arrays[i+test3] = model.docvecs[prefix_test_4star]
    test_labels[i+test3] = 4

for i in xrange(test5):
    prefix_test_5star = 'TEST_FIVE_' + str(i)
    test_arrays[i+test4] = model.docvecs[prefix_test_5star]
    test_labels[i+test4] = 5


print "training classifier...." 
#classifier = RandomForestClassifier(n_estimators=100,n_jobs=2)
classifier = LogisticRegression()

classifier.fit(train_arrays, train_labels)

with open("logisticregression.pkl","wb") as f:
    pickle.dump(classifier,f)

print classifier.score(test_arrays, test_labels)
