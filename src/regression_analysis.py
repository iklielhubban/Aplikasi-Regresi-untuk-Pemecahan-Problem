import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load dataset from CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """
    Clean dataset by removing missing values.
    """
    data.dropna(inplace=True)
    return data

def split_data(data):
    """
    Split dataset into features and target variable, then split into training and testing sets.
    """
    X = data[['Hours Studied', 'Sample Question Papers Practiced']]
    y = data['Performance Index']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def linear_regression(X_train, X_test, y_train):
    """
    Train linear regression model and make predictions.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def polynomial_regression(X_train, X_test, y_train):
    """
    Train polynomial regression model and make predictions.
    """
    X_train_poly = np.column_stack((X_train['Hours Studied'] ** 2, X_train['Sample Question Papers Practiced'] ** 2))
    X_test_poly = np.column_stack((X_test['Hours Studied'] ** 2, X_test['Sample Question Papers Practiced'] ** 2))
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    return model, y_pred

def calculate_rms(y_test, y_pred):
    """
    Calculate Root Mean Squared Error.
    """
    rms = mean_squared_error(y_test, y_pred, squared=False)
    return rms

def plot_regression_results(X_test, y_test, y_pred, title, save_path=None):
    """
    Plot scatter plot of test data points and regression line.
    """
    plt.scatter(X_test['Hours Studied'], y_test, color='blue')
    plt.plot(X_test['Hours Studied'], y_pred, color='red')
    plt.title(title)
    plt.xlabel('Hours Studied')
    plt.ylabel('Performance Index')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Load dataset
    data = load_data('data/student_performance.csv')

    # Clean dataset
    data = clean_data(data)

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(data)

    # Linear Regression
    model_linear, y_pred_linear = linear_regression(X_train, X_test, y_train)
    rms_linear = calculate_rms(y_test, y_pred_linear)
    print("RMS Linear Regression:", rms_linear)
    plot_regression_results(X_test, y_test, y_pred_linear, 'Linear Regression', save_path='results/linear_regression_plot.png')

    # Polynomial Regression
    model_poly, y_pred_poly = polynomial_regression(X_train, X_test, y_train)
    rms_poly = calculate_rms(y_test, y_pred_poly)
    print("RMS Polynomial Regression:", rms_poly)
    plot_regression_results(X_test, y_test, y_pred_poly, 'Polynomial Regression', save_path='results/polynomial_regression_plot.png')

if __name__ == "__main__":
    main()