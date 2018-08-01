# Simple Linear Regression

# Importing the libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas

sys.path.insert(0, "../DataPreprocessing")
from preprocess import Preprocess
preprocess = Preprocess()
preprocess.doPreprocess("Salary_Data.csv", -1, 1)
dataset = preprocess.dataset
dependant_matrix = pandas.DataFrame(preprocess.dependent_matrix)
dependant_train_matrix = pandas.DataFrame(preprocess.dependent_train_matrix)
dependant_test_matrix = pandas.DataFrame(preprocess.dependent_test_matrix)
independant_matrix = pandas.DataFrame(preprocess.independent_matrix)
independant_train_matrix = pandas.DataFrame(preprocess.independent_train_matrix)
independant_test_matrix = pandas.DataFrame(preprocess.independent_test_matrix)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(preprocess.independent_train_matrix, preprocess.dependent_train_matrix)

# Predicting the Test set results
dependant_prediction = regressor.predict(preprocess.independent_test_matrix)

# Visualising the Training set results
plt.scatter(preprocess.independent_train_matrix, preprocess.dependent_train_matrix, color = 'red')
plt.plot(preprocess.independent_train_matrix, regressor.predict(preprocess.independent_train_matrix), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(preprocess.independent_test_matrix, preprocess.dependent_test_matrix, color = 'red')
plt.plot(preprocess.independent_train_matrix, regressor.predict(preprocess.independent_train_matrix), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
