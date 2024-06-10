import unittest
import pandas as pd
from src.regression_analysis import load_data, clean_data, split_data, linear_regression, polynomial_regression, calculate_rms

class TestRegressionAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load and clean data once for all tests
        cls.data = load_data('data/student_performance.csv')
        cls.cleaned_data = clean_data(cls.data)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = split_data(cls.cleaned_data)
    
    def test_load_data(self):
        data = load_data('data/student_performance.csv')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
    
    def test_clean_data(self):
        cleaned_data = clean_data(self.data)
        self.assertFalse(cleaned_data.isnull().values.any())
    
    def test_split_data(self):
        X_train, X_test, y_train, y_test = split_data(self.cleaned_data)
        self.assertEqual(len(X_train) + len(X_test), len(self.cleaned_data))
        self.assertEqual(len(y_train) + len(y_test), len(self.cleaned_data))
    
    def test_linear_regression(self):
        model, y_pred = linear_regression(self.X_train, self.X_test, self.y_train)
        self.assertEqual(len(y_pred), len(self.y_test))
        rms = calculate_rms(self.y_test, y_pred)
        self.assertGreater(rms, 0)
    
    def test_polynomial_regression(self):
        model, y_pred = polynomial_regression(self.X_train, self.X_test, self.y_train)
        self.assertEqual(len(y_pred), len(self.y_test))
        rms = calculate_rms(self.y_test, y_pred)
        self.assertGreater(rms, 0)

if __name__ == '__main__':
    unittest.main()
