import pandas as pd
# Display all the columns
pd.set_option("display.max_columns", None)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn import linear_model
import missingno as msno
from statsmodels.tsa.arima.model import ARIMA

def cleaning(df):

    # Drop all the duplicates
    df = df.drop_duplicates()

    # Create a dictionary with the number of "NaN" per column
    dict_nans = dict(df.isnull().sum())

    # Initialize the columns_to_drop list
    columns_to_drop = []
    
    # Add to the above mentioned list all the columns that contain "id" or "ID"
    for column in df.columns:
        if "id" in column:
            columns_to_drop.append(column)
        elif "ID" in column:
            columns_to_drop.append(column)
            
    # Drop all the columns that contain "id" or "ID" in their name
    df = df.drop(columns = columns_to_drop)
    
    # Deal the outliers in the columns "num_room", "material", "full_sq", "state", "build_year"
    df["num_room"].replace(to_replace = 0, value = np.nan, inplace = True)
    df["material"].replace(to_replace = 3, value = 1, inplace = True)
    df["full_sq"].replace(to_replace = 0, value = np.nan, inplace = True)
    df["state"].replace(to_replace = 33, value = 3, inplace = True)
    df["build_year"].replace(to_replace = {20052009:2005, 0:np.nan, 1:np.nan, 2:np.nan, 3:np.nan, 20:2000, 215:2015, 4965:1965, 71:1971}, inplace = True)

    return df

def preparing(df):
    
    # Create several date features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["Date"] = df["timestamp"].dt.strftime('%d.%m.%Y')
    df["year"] = pd.DatetimeIndex(df["timestamp"]).year
    df["month"] = pd.DatetimeIndex(df["timestamp"]).month
    df["day"] = pd.DatetimeIndex(df["timestamp"]).day
    
    # Create the "build_year_old" feature
    df["build_year_old"] = 2020 - df["build_year"]
    
    # Drop the "timestamp" column as it is not useful anymore
    df = df.drop(columns = ["timestamp"])
    
    return df

def mode_fill_na(df):  
    
    # Fill "NaN" with the mode of the column
    df_final = pd.DataFrame()
    df_mode = df.mode()
    for column in df:
        df_final[column] = df[column].fillna(df[column].mode()[0])
        
    return df_final

def ordinal_encoding(df):
    
    # Transform categorical variables into numeric variables
    enc = OrdinalEncoder()
    columns = list(df.columns)
    for column in columns:
        column_numpy = df[column].to_numpy()
        column_numpy = column_numpy.reshape(-1, 1)
        df[column] = enc.fit_transform(column_numpy)
        
    return df
    
# Read the train.csv
df_train = pd.read_csv(r"train.csv", sep = ",", parse_dates = ["timestamp"])

df_train.describe()

# Print the shape of the dataset
print("The shape of the dataset is:", df_train.shape)

# Visualize the number of missing values of df_train as a bar chart
msno.bar(df_train)

# Show the distribution of prices
plt.hist(df_train["price_doc"], bins = 30)
plt.show()

# Clean the dataset
df_train_cleaned = cleaning(df_train)

# Prepare the dataset
df_train_prepared = preparing(df_train_cleaned)

# Fill all the "NaN" by the mode of the column
df_train_filled = mode_fill_na(df_train_prepared)

# Transform all the categorical variables into numeric variables
df_train_transformed = ordinal_encoding(df_train_filled)

# Prepare a dataframe with all the features
df_train_data = df_train_transformed.drop(columns = ["price_doc"])

# Prepare a dataframe with the class
df_train_target = df_train_transformed["price_doc"]

# Create correlation matrix
corr_matrix = df_train_data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

# Drop all the correlated features
df_train_data = df_train_data.drop(df_train_data[to_drop], axis=1)

# Split the dataset into X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(df_train_data, df_train_target, test_size=0.9)

# Build a linear regression to better understand which features are important
regr = linear_model.LinearRegression()
regr.fit(df_train_data, df_train_target)

# Show the features importance
importance = regr.coef_
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# Score the test dataset with the linear regression model
predictions = regr.predict(X_test)
predictions = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

# Calculate the coefficient of determination
r2_lr = regr.score(predictions, y_test)
print("The coefficient of determination of the LR model is:", r2_lr)

# Calculate the mean absolute error
predictions = regr.predict(X_test)
mae_lr = mean_absolute_error(y_test, predictions)
print("The mean absolute error of the LR model is:", mae_lr)

# Retrieve the features
features = np.array(df_train_data.columns)

# Create a dataframe composed of both the features and their coefficients in the linear regression model
df_linear_regression = pd.DataFrame({"coefficients": importance, "features": features}, columns=["coefficients", "features"])

# 5 highest coefficients
df_linear_regression_first_5 = df_linear_regression.sort_values(by = "coefficients", ascending = False).reset_index(drop = True).head(5)

df_linear_regression_first_5["coefficients"] = df_linear_regression_first_5["coefficients"].apply(lambda x: round(x, 3))

print(df_linear_regression_first_5)

# 5 lowest coefficients
df_linear_regression_last_5 = df_linear_regression.sort_values(by = "coefficients", ascending = True).reset_index(drop = True).head(5)

df_linear_regression_last_5["coefficients"] = df_linear_regression_last_5["coefficients"].apply(lambda x: round(x, 3))

print(df_linear_regression_last_5)

# Read the train.csv
df_train = pd.read_csv(r"train.csv", sep = ",", parse_dates = ["timestamp"])

series_time = df_train.groupby(["timestamp"])["price_doc"].mean()

# Set "year_month" as index

print(series_time)

# Create an autocorrelation plot
auto = pd.plotting.autocorrelation_plot(series_time)
  
# Plot the Curve
auto.plot()
  
# Display
plt.show()

# Split into train and test sets
X = series_time.values
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]

# Create a list to append all predictions
predictions = list()

# Perform a walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    
# Calculate the coefficient of determination
r2_ar = r2_score(test, predictions)
print("The coefficient of determination of the AMIRA model is:", r2_ar)

# Calculate the mean absolute error
mae_ar = mean_absolute_error(test, predictions)
print("The mean absolute error of the AMIRA model is:", mae_ar)

print(model_fit.summary())

# A random forest could be interesting with so many features
clf = RandomForestRegressor(n_estimators = 100, max_depth= 25)
clf.fit(X_train, y_train)
                             
# Use the forest's predict method on the test data
predictions = clf.predict(X_test)

# Calculate the coefficient of determination
r2_rf = r2_score(y_test, predictions)
print("The coefficient of determination of the RF model is:", r2_rf)

# Calculate the mean absolute error
mae_rf = mean_absolute_error(y_test, predictions)
print("The mean absolute error of the RF model is:", mae_rf)

print(accuracy_score(y_test, predictions))

# List of columns to build a dataframe with all the models results
columns_df = ["Linear regression", "ARIMA", "Random forest"]

# Index of the dataframe
index_df = ["coefficient of determination", "mean absolute error"]

# Create a dataframe with all the models results
df_scores = pd.DataFrame(list(zip([round(r2_lr, 2), round(mae_lr, 2)] , [round(r2_ar, 2), round(mae_ar, 2)], [round(r2_rf, 2), round(mae_rf, 2)])),columns = columns_df)
df_scores.index = index_df

print(df_scores)s
