# # Boston Housing Linear Regression
# ### Import Necessary Modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ### Stylistics
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("dark_background")
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 100)
pd.set_option("precision", 2)

# ### Import the Dataset
boston_dataset = load_boston()
# boston is a bunch data type so we are presented with a dictionary like structure with keys mentioned below.
boston_dataset.keys()

# #### Data Definition
# Let's look at the definition of these columns to better understand the data.
print(boston_dataset.DESCR)
boston = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)
boston.head()
boston["MEDV"] = boston_dataset.target

# ### Data Cleaning
# #### Remove NULL Values 
boston.isnull().sum()

# ### Feature Engineering
ax, figure = plt.subplots(figsize=(10, 8))
correlation_matrix = boston.corr().round(2)
_ = sns.heatmap(correlation_matrix, annot=True, cmap="YlOrRd")

# ### Linear Regression
# #### Data Splitting
X = pd.DataFrame(boston["RM"])
y = boston_dataset.target
X.head()
y[:10]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# #### Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# #### Model Traning
regressor = LinearRegression().fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# #### Model Visualization
_ = plt.scatter(X_train, y_train, c="Red")
_ = plt.plot(X_train, regressor.predict(X_train), c="Blue")
_ = plt.xlabel("RM")
_ = plt.ylabel("MEDV")
