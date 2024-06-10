import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.regression_analysis import load_data, clean_data, split_data, linear_regression, polynomial_regression, calculate_rms

class TestRegressionAnalysis(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load and clean the dataset once for all tests
        cls.data = load_data('data/student_performance.csv')
        cls.data = clean_data(cls.data)
    
    def test_load_data(self):
        # Test if data is loaded correctly
        self.assertIsInstance(self.data, pd.DataFrame)
    
    def test_clean_data(self):
        # Test if data is cleaned correctly
        self.assertFalse(self.data.isnull().values.any())
    
    def test_split_data(self):
        # Test if data is split correctly
        X_train, X_test, y_train, y_test = split_data(self.data)
        self.assertEqual(len(X_train) + len(X_test), len(self.data))
        self.assertEqual(len(y_train) + len(y_test), len(self.data))
    
    def test_linear_regression(self):
        # Test linear regression model
        X_train, X_test, y_train, y_test = split_data(self.data)
        model, y_pred = linear_regression(X_train, X_test, y_train)
        self.assertIsInstance(model, LinearRegression)
        self.assertEqual(len(y_pred), len(y_test))
    
    def test_polynomial_regression(self):
        # Test polynomial regression model
        X_train, X_test, y_train, y_test = split_data(self.data)
        model, y_pred = polynomial_regression(X_train, X_test, y_train)
        self.assertIsInstance(model, LinearRegression)
        self.assertEqual(len(y_pred), len(y_test))
    
    def test_calculate_rms(self):
        # Test RMS calculation
        X_train, X_test, y_train, y_test = split_data(self.data)
        _, y_pred_linear = linear_regression(X_train, X_test, y_train)
        rms = calculate_rms(y_test, y_pred_linear)
        self.assertIsInstance(rms, float)
        self.assertGreaterEqual(rms, 0)

if __name__ == '__main__':
    unittest.main()