import matplotlib.pyplot as plt
plt.scatter(xtrain,ytrain,color='red')
plt.plot(xtest,polypred,color='blue')
plt.scatter(xtest,ytest,color='green')
