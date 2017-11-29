from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import h5py
import numpy as np
import math
from bahsic import CBAHSIC
import vector
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['dataset_1'][:]
	return X

def accuracyclassification(a,b):
	acc=0
	for j in range(len(a[0])):
		x=[]
		y=[]
		for i in range(4):
			x.append(a[i][j])
			y.append(b[i][j])

		# print x,y
		# print np.argmax(x),np.argmax(y)
		x=np.argsort(x)


		if np.argmax(x)==np.argmax(y):
			acc+=1

	return acc/float(len(a[0]))



def accuracyregression(a,b):
	count=0
	# print a.shape[0]
	for i in range(a.shape[0]):
		# print a[i],b[i]
		if abs(a[i]-b[i])<=0.004:
			count+=1
	return (count*100.0)/a.shape[0]

def normalize(y1,data_no):
	y1=y1[:,None]
	# Normalize the labels.
	y1 = 1.0*y1
	tmp_no = np.sum(y1)
	pno = (data_no + tmp_no) / 2
	nno = (data_no - tmp_no) / 2
	y1[y1>0] = y1[y1>0]/pno
	y1[y1<0] = y1[y1<0]/nno
	# y1=y1.reshape((y1.shape[0],))
	return y1

def main():
	X=load('xdata.h5')
	y1=load('y1data.h5')
	y2=load('y2data.h5')
	y3=load('y3data.h5')
	y4=load('y4data.h5')
	# print X.shape
	# print y1.shape


	data_no=X.shape[0]

	y1=normalize(y1,data_no)
	y2=normalize(y2,data_no)
	y3=normalize(y3,data_no)
	y4=normalize(y4,data_no)

	# print y

	# Normalize the data.
	m = X.mean(0)
	s = X.std(0)
	# print m,s
	X.__isub__(m).__idiv__(s)

	# foo=np.copy(X)

	xlabels=[]
	ylabels=[]

	foo=np.copy(X)
	for features_tokeep in range(500,5001,500):
		X=np.copy(foo)
		xlabels.append(features_tokeep)
		print xlabels
		bahsic = CBAHSIC()
		bhs = bahsic.BAHSICRaw(X, y1, vector.CLinearKernel(), vector.CLinearKernel(), features_tokeep, 0.1)
		hsicfeatures= np.zeros(shape=(data_no,features_tokeep))

		for i in range(0,data_no):
			for j in range(0,features_tokeep):
				hsicfeatures[i][j] = X[i][bhs[features_tokeep+j]]

		X=hsicfeatures

		print X.shape

		# y1=y1.reshape((y1.shape[0],))
		# print y1.shape


		#feature extraction
		# test train split
		# xlabels=[]
		# ylabels=[]
		# y1labels=[]
		# y2labels=[]
		# y3labels=[]
		# y4labels=[]

		# pca = PCA(n_components=pcanum)
		# X=pca.fit_transform(foo)
		# print X.shape

		labels=list(zip(y1,y2,y3,y4))
		xtrain, xtest, ytrain, ytest = train_test_split(X,labels,test_size=0.33,random_state=42)


		tmp=zip(*ytest)
		ytest1=tmp[0]
		ytest2=tmp[1]
		ytest3=tmp[2]
		ytest4=tmp[3]
		ytest1=np.array(ytest1)
		ytest2=np.array(ytest2)
		ytest3=np.array(ytest3)
		ytest4=np.array(ytest4)

		tmp=zip(*ytrain)
		ytrain1=tmp[0]
		ytrain2=tmp[1]
		ytrain3=tmp[2]
		ytrain4=tmp[3]
		ytrain1=np.array(ytrain1)
		ytrain2=np.array(ytrain2)
		ytrain3=np.array(ytrain3)
		ytrain4=np.array(ytrain4)


		# clf=SVR(kernel='rbf')
		clf=LinearRegression()
		clf.fit(xtrain,ytrain1)
		vals1=clf.predict(xtest)
		# print vals1
		# print accuracyregression(ytest1,vals1)


		# clf=SVR(kernel='linear',C=1e3)
		clf.fit(xtrain,ytrain2)
		vals2=clf.predict(xtest)
		# print accuracyregression(ytest2,vals2)

		# clf=SVR(kernel='linear',C=1e3)
		clf.fit(xtrain,ytrain3)
		vals3=clf.predict(xtest)
		# print accuracyregression(ytest3,vals3)

		# clf=SVR(kernel='linear',C=1e3)
		clf.fit(xtrain,ytrain4)
		vals4=clf.predict(xtest)
		# print accuracyregression(ytest4,vals4)

		truth=np.zeros((4,len(ytest1)))
		predicted=np.zeros((4,len(vals1)))
		for j in range(len(ytest1)):
			truth[0][j]=ytest1[j]
			predicted[0][j]=vals1[j]
			truth[1][j]=ytest2[j]
			predicted[1][j]=vals2[j]
			truth[2][j]=ytest3[j]
			predicted[2][j]=vals3[j]
			truth[3][j]=ytest4[j]
			predicted[3][j]=vals4[j]

		ylabels.append(accuracyclassification(truth,predicted))
		# print accuracyclassification(truth,predicted)

	print xlabels
	print ylabels
	plt.plot(xlabels,ylabels)
	plt.savefig("classificationbasic1.png")
		# plt.clf()
		# plt.plot(xlabels,y1labels)
		# plt.savefig("v1basic.png")
		# plt.clf()
		# plt.plot(xlabels,y2labels)
		# plt.savefig("v2basic.png")
		# plt.clf()
		# plt.plot(xlabels,y3labels)
		# plt.savefig("v3.png")
		# plt.clf()
		# plt.plot(xlabels,y4labels)
		# plt.savefig("v4.png")
		# plt.clf()


main()