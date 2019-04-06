import numpy as np
import pandas as pd
from numpy import dot
from numpy import mat
from numpy.linalg import inv
from sklearn import datasets
from argparse import Namespace
import matplotlib.pyplot as plt

args = Namespace(
    seed = 1234,
    data_file = 'sample_data.csv',
    num_samples = 100,
    train_size = 0.75,
    test_size = 0.25,
    num_epochs = 100,
)

np.random.seed(args.seed)

def generate_data(num_samples):
    X = np.array(range(num_samples))
    random_noise = np.random.uniform(-10,10,size = num_samples)
    y = 3.65 * X + 10 + random_noise
    return X,y

X,y = generate_data(args.num_samples)
data = np.vstack([X,y]).T
df = pd.DataFrame(data,columns = ['x1','y'])
# plt.title('Generated data')
# plt.scatter(x=df['X'], y=df['y'])
# plt.show()
df['x0'] = 1
X = df.iloc[:,[2,0]]
Y = df.iloc[:,1].values.reshape(100,1)

theta = dot(dot(inv(dot(X.T,X)),X.T),Y)
print(theta)

theta = np.array([10.,1.]).reshape(2,1)
alpha = 0.0001
temp = theta
lenY = len(Y)
x0 = X.iloc[:,0].values.reshape(lenY,1)
x1 = X.iloc[:,1].values.reshape(lenY,1)
# print(np.sum((Y - dot(X,theta)) * x0) / lenY)
# print(np.sum((Y - dot(X,theta)) * x1) / lenY)

for i in range(1000):
    temp[0] = theta[0] - alpha * np.sum((dot(X,theta) - Y) * x0) / lenY
    temp[1] = theta[1] - alpha * np.sum((dot(X,theta) - Y) * x1) / lenY
    theta = temp
print(theta)
