import fair_mass
import numpy as np

#create some sample data to work with
np.random.seed(0)

#non-protected attributes
X = np.random.random((100, 3))
#the protected attribute
Z = np.random.randint(2, size=100).reshape(-1,1)
#combine the attributes and generate Y using this data
XZ = np.hstack([X,Z])
w = np.array([1.0, -2.0, 3.0, 2.0])
y = 1/(1 + np.exp(-XZ @ w))
Y = np.array(y >= 0.5, dtype=np.int32)

#create an optimization model with the generated data
#note that we already had X and Z seperated
model = fair_mass.Optimization(X, Z, Y, influence='shap')

#generate some more data for non-protected attribute
Xt = np.random.random((100, 3))
print(model.predict(X))
