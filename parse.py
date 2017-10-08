import cPickle as cp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

def accuracy(a,b):
	acc=0
	for i in range(0,len(a)):
		if a[i]==b[i]:
			acc+=1
	return (acc/float(len(a)))*100

def emotion(x):
	print x
	if x[0]>=4:
		if x[1]>=4:
			return 0 #happy
		else:
			return 1 #angry
	elif x[0]<4:
		if x[1]>=4:
			return 2 #relaxed
		else:
			return 3 #sad



data = cp.load(open('/home/sanchit/Desktop/Semester 5/AI/Project/EEG Data/data_preprocessed_python/s01.dat', 'rb'))
print "Done"
x=data['data']
y=data['labels']
# print y
# print x.shape
# print y.shape
l=[]
for obs in x:
	ch=[]
	for channel in obs:
		avg=0
		for samp in channel:
			avg+=samp
		avg=avg/float(8064)
		ch.append(avg)
	l.append(ch)

l=np.array(l)

for i in y:
	print max(i)

clf=svm.SVC(kernel='rbf')
clf=
y_val=[]
for i in range(0,len(y)):
	y_val.append(emotion(y[i]))

x_train, x_test, y_train, y_test = train_test_split(l,y_val,test_size=0.25)
# print x_test
# print x_train
clf.fit(x_train,y_train)
labels=clf.predict(x_test)
print y_test
print labels.tolist()
# print y_train
print accuracy(labels.tolist(),y_test)