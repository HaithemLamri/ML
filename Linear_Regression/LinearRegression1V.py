import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

class Linear_Regression(object):
    def __init__(self,X,y,theta,learning_rate,n_iters):
        self.X=X
        self.y=y
        self.theta=theta
        self.alpha=learning_rate
        self.n=n_iters
    classmethod
    def mean_squared_error(self):
        z = np.power(np.dot(self.X, self.theta.T) - self.y, 2)
        return np.sum(z) / (2 * len(self.X))

    def gradient_gescent(self):
        tmp = np.zeros(self.theta.shape)
        parameters = int(self.theta.shape[1])  # number of features +1
        cost = np.zeros(self.n)
        for i in range(self.n):
            error = np.dot(self.X, self.theta.T) - self.y
            for j in range(parameters):
                term = np.multiply(error, self.X[:, j])
                tmp[0, j] = self.theta[0, j] - ((self.alpha * np.sum(term)) / len(self.X))
            self.theta = tmp
            cost[i] = self.mean_squared_error()
        return theta, cost
    def normal_equation(self):
         return (np.dot(np.linalg.inv(np.dot(self.X.T, self.X)), self.X.T).dot((self.y))).T



#Read the Data
data=pd.read_csv("/LinearRegression/Data.txt", header=None, names=['Population', 'Notes'])

#Show Data details
print("This is the new format of the Data:  \n",data.head(10))
print("*"*50)
print(" Data.describe: \n",data.describe())
print("*"*50)
#Drow Data
plt.style.use('classic')
plt.scatter(data['Population'],data['Notes'])
#plt.show()


#Adding a new col to the Data for the Gradient Descent
data.insert(0,'Ones',1)
print("the new format is : \n",data.head(10))
print("*"*50)

#Separate X (Training Data) from Y (Target variable)
cols=data.shape[1] #the number of columns in the data
x=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]


print("the training data : \n",x.head(10))
print("the target varibale : \n",y.head(10))

#Convert from data frames to matrices
X=np.mat(x.values)
y=np.mat(y.values)

theta=np.mat([[0,0]])
print(theta)

print(f"theta dim={theta.shape}")
print("X in matrix format: \n",X)
print("X.shape= ",X.shape)
print("*"*50)
print("Y in matrix format : \n",y)
print("Y.shape= ",y.shape)

print("*"*50)


LR=Linear_Regression(X,y,theta,learning_rate=0.01,n_iters=2000)

print(" The intitial cost =",LR.mean_squared_error())
print("*"*50)

#fit the model paramaters
h,J=LR.gradient_gescent()
#Calculating theta using the normal equation
h1=LR.normal_equation()

print("the predection function by the gradient descent h(x):",h[0,0]," +" ,h[0,1],"*x")
print("the predection function by the normal equation h(x):",h1[0,0]," +" ,h1[0,1],"*x")
#print("the cost values : ", j)
print("the minimum cost = ",LR.mean_squared_error())
print("*"*50)
###############################"
#the best fit line
x1=np.linspace(data['Population'].min(),data['Population'].max(),100)
h_theta=h[0,0] +h[0,1]*x1

#draw the line
fig,ax=plt.subplots(figsize=(5,5))
ax.plot(x1,h_theta,'r',label='Predection')
ax.scatter(data.Population,data.Notes,label='Data')
ax.set_xlabel('Population')
ax.set_ylabel('Notes')
ax.set_title('Predection of Notes')


#draw  the error graph
fig,ax=plt.subplots(figsize=(5,5))
ax.plot(np.arange(2000),J,'r')
ax.set_xlabel('iterations')
ax.set_ylabel('cost')
ax.set_title('cost vs iterations')
plt.show()






