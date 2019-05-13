from sklearn.svm import LinearSVC, SVC
import numpy as np
from random import shuffle

def normalized(X,Xdash):

	mean = np.mean(X,axis = 0)
	std = np.std(X,axis = 0)

	print(mean)
	print(std)
	TrainSetX = (X - mean)/std
	TestX = (Xdash - mean)/std

	return TrainSetX,TestX

if __name__ == '__main__':

	fileRead = open('Train.txt','r')

	listVector = list()
	for line in fileRead:

		line = line.strip()
		line = line.split()

		vector = [float(sub) for sub in line]
		listVector.append(line)


	shuffle(listVector)
	Tot = len(listVector)

	train = int(0.8*Tot)
	completeFeature = np.array(listVector, dtype=np.float32)


	X = completeFeature[:train, :9]
	TrainY = completeFeature[:train,-1]

	Test = completeFeature[train:, :9]
	TestY = completeFeature[train:,-1]

	#X,Test = normalized(X,Test)
	#clf = LinearSVC(max_iter = 100000,random_state=0, tol=1e-5)
	clf = SVC(max_iter = 200000,random_state = 0,tol = 1e-9)
	clf.fit(X, TrainY)
	#print(clf.coef_)


	acc = 0
	cor1 = 0
	actual1 = 0
	tp=0
	for index,row in enumerate(X):

		row = np.reshape(row, (1,9))
		pred = clf.predict(row)		
		if(pred[0] == TrainY[index]):
			acc += 1
			if(pred[0] == 1):
				tp += 1

		if(TrainY[index] == 1):
			actual1 += 1
		if(pred[0] == 1):
			cor1 += 1

	print("For Train Set :- ")
	print("Acuuracy : "+str(acc) +" TotExamples : "+str(len(X)))
	print("Total 1 Pred : "+str(cor1)+" Actual 1 :"+str(actual1)+"True Positive : "+str(tp))

	print("---------------------------------------------------------")
	print("For Test Set : ")

	acc = 0
	cor1 = 0
	actual1 = 0
	tp=0
	for index,row in enumerate(Test):

		row = np.reshape(row, (1,9))
		pred = clf.predict(row)		
		#print(pred[0])
		#print(TestY[index])
		
		if(pred[0] == TestY[index]):
			acc += 1
			if(pred[0] == 1):
				tp += 1

		if(TestY[index] == 1):
			actual1 += 1
		if(pred[0] == 1):
			cor1 += 1
		
	print("Acuuracy : "+str(acc) +" TotExamples : "+str(len(X)))
	print("Total 1 Pred : "+str(cor1)+" Actual 1 :"+str(actual1)+"True Positive : "+str(tp))
