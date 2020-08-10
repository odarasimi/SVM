import numpy as np
from sklearn import svm

test_data = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}


word = [80,80]

#data1, data2 and data3 are used to generate different possible combinations for the decision vectors including negative values to make it more general, 
#however, it needs to be more computationally inexpensive
data1 = []
for i in range(word[0]+1):
	for v in range (word[1]):
		data1.append([i, v])		
for c in range(word[1]+1):
	data1.append([c,word[0]])

data2 = [] 
transform = [[1,-1],[-1,-1],[-1,1]]
for i in transform:
	for value in data1:
		new = [value[0]*i[0],value[1]*i[1]]
		data2.append(new)

data = (data1 + data2)
data3 = []
for item in data:
	semedo = [item[0]*0.1,item[1]*0.1]
	data3.append(semedo)
	puig = [item[0]*0.01,item[1]*0.01]
	data3.append(puig)
	frenkie = [item[0]*0.001,item[1]*0.001]
	data3.append(frenkie)
data += data3
data = [np.array(item) for item in data]

#bb defines the first value for the intercept of the equation
bb = 80

data_dict = {}
for array in  data:
	for value in range(-bb,bb):
		found_option = True
		for iv in test_data:
			for xy in test_data[iv]:				
				yy = iv
				krustykrab = np.linalg.norm(array)
				if not yy*(np.dot(array,xy)+value) >= 1:
					found_option = False
					break
		if found_option:		
			data_dict[krustykrab] = [array,value]

norms = sorted([n for n in data_dict])
#||w|| : [w,b]


data_choicee = data_dict[norms[5]]
w = data_choicee[0]
b = data_choicee[1]
print (w,b)


def predictt(features):
    # sign( x.w+b )
    classification = np.sign(np.dot(np.array(features),w)+b)
    return classification

#test
predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

for p in predict_us:
    print(predictt(p))


print ("###############")

#now with the scikit-learn classifier
datab = np.array([[1,7],[2,8],[3,8],[5,1],[6,-1],[7,3]])

y = [-1,-1,-1,1,1,1]
clf = svm.SVC(kernel="linear", C = 1.0)
clf.fit(datab,y)
for value in predict_us:
	print (clf.predict([value]))


