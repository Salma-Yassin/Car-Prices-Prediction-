import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import opendatasets as od
od.download("https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho/data")
##############################################################################################
#reading data
df = pd.read_csv('vehicle-dataset-from-cardekho/car data.csv')
df.head()

#Preprocessing
df.info()
df.describe()
df.drop_duplicates()
df.isnull().sum()
df.nunique()
########################################################
##############Visualization#####################
import pandas as pd
import matplotlib.pyplot as plt
# Plotting
import pandas as pd
import matplotlib.pyplot as plt
# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(df['Present_Price'], df['Selling_Price'], color='blue', alpha=0.5)
plt.title('Selling Price vs Present Price')
plt.xlabel('Present Price')
plt.ylabel('Selling Price')
plt.grid(True)
plt.show()

########################################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Assuming you have a pandas DataFrame named 'data'
X = df.drop('Selling_Price', axis=1)  # Assuming you have a target column
y = df['Selling_Price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Consider adjusting test_size

# Identify categorical columns in the training data
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

# Create a LabelEncoder object for each categorical column and fit on training data
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    le_dict[col] = le  # Store the encoder for later use

# Apply the fitted encoders to both training and test sets
for col, le in le_dict.items():
    X_test[col] = X_test[col].map(lambda s: -1 if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, -1)
    X_test[col] = le.transform(X_test[col])

# Now X_train and X_test have label-encoded categorical features with handling of unseen categories

X_train.Car_Name.nunique()
X_test.Car_Name.nunique()
x11=X_train.Car_Name.unique()
x22=X_test.Car_Name.unique()
np.intersect1d(x11, x22).__len__()

#############################################################################33
#Model building
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  r2_score, max_error,mean_squared_error

rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

# rmse_rf = rmse(y_test, y_pred_rf)
r2_score_rf = r2_score(y_test, y_pred_rf)
max_error_rf = max_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

# print(f"rmse : {rmse_rf}")
print(f"r2_score : {r2_score_rf}")
print(f"max_error : {max_error_rf}")
print(f"mean_squared_error : {mse_rf}")





