import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.stats import zscore
import pandas as pd
import matplotlib.pyplot as plt



columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

df = pd.read_csv('/home/julian/Desktop/stats/data/abalone.data', header=None, names=columns)

word_to_number = {
    'M':0,
    'F':1,
    'I':2
}

# cleaning data in Sex column
df['Sex'] = df['Sex'].map(word_to_number)

# assign features
feature1 = 'Diameter'
feature2 = 'Height'
feature3 = 'Shucked weight'

#print(df)

def graph(df,feature):

    X = df[[feature]]
    y = df['Rings']
    
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.5, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_val = model.predict(X_val)

    mse = mean_squared_error(y_test, y_pred)
    mseVal = mean_squared_error(X_val, y_pred_val)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("Mean Squared Error for validation:", mseVal)
    print("R-squared:", r2)

    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.title('Linear Regression')
    plt.show()

    comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(comparison.head(10))

    variance = np.var(y_pred)

    print("Variance:", variance)

graph(df, feature1)
graph(df, feature2)
graph(df, feature3)

print("MEAN: ",df['Rings'].mean())

#transformation using z-scale
def normalizeData(df, features):
    #Clean data
    df_cleaned = df.copy()

    columns_to_process = features
    points = 0

    for column in columns_to_process:
        z_scores = np.abs(stats.zscore(df[column]))
        threshold = 2
        mean_value = df[column].mean()

        changed = np.sum(z_scores > threshold)
        points += changed

        df_cleaned.loc[z_scores > threshold, column] = mean_value

    print(f"Total data points changed: {points}")
    df_cleaned['Rings'] = df['Rings']
    return(df_cleaned)

columns=['Diameter', 'Height', 'Shucked weight']
newDF= normalizeData(df, columns)

graph(newDF, feature1)
graph(newDF, feature2)
graph(newDF, feature3)