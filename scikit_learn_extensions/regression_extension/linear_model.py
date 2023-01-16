# -*- coding: utf-8 -*-
"""
Created on 2023/1/16 14:08

@author: chenjunhan
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import f
from typing import Any


class LinearRegressionE(LinearRegression):
    """
    Extension class for LinearRegression, add f test and modified r test.

    Dependencies
    ========

    scikit-learn=1.2.0

    numpy=1.24.1

    scipy=1.10.0

    See Also
    ========

    _calculate_check_value
    """

    def __init__(self, *, fit_intercept: bool = True, copy_X: bool = True, n_jobs: Any = None, positive: bool = False):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)

    @staticmethod
    def _calculate_check_value(y: np.ndarray, y_predict: np.ndarray) -> tuple[np.ndarray, ...]:
        r"""
        Calculate SSR, SSE, SST based on the true and predict value of y.

        Explanation
        ========
        .. math::
            $$
            SSR=\sum_{i=1}^n{\left( \hat{y}_i-\bar{y} \right) ^2}
            $$
        .. math::
            $$
            SSE=\sum_{i=1}^n{\left( \hat{y}_i-y_i \right) ^2}
            $$
        .. math::
            $$
            SST=SSR+SSE
            $$

        See Also
        ========
        f_test

        modified_r_test

        :param y: numpy.ndarray, true value
        :param y_predict: numpy.ndarray, predict value
        :return: a tuple of numpy.ndarray
        """
        y_mean = y.mean(axis=0, keepdims=True)

        SSR = ((y_predict - y_mean) ** 2).sum(axis=0, keepdims=True)
        SSE = ((y_predict - y) ** 2).sum(axis=0, keepdims=True)
        SST = SSR + SSE

        return SSR, SSE, SST

    def f_test(self, X: np.ndarray, y: np.ndarray, y_predict: np.ndarray) -> np.ndarray:
        SSR, SSE, SST = self._calculate_check_value(y, y_predict)
        n, m = X.shape
        value = (SSR / m) / (SSE / (n - m - 1))
        F = f(m, n - m - 1)
        prominence = 1 - F.cdf(value)
        return prominence

    def modified_r_test(self, X: np.ndarray, y: np.ndarray, y_predict: np.ndarray) -> np.ndarray:
        SSR, SSE, SST = self._calculate_check_value(y, y_predict)
        n, m = X.shape
        return 1 - (SSE / (n - m - 1)) / (SST / (n - 1))


if __name__ == '__main__':
    # check for LinearRegressionE
    print('-----check for LinearRegressionE-----')

    from sklearn.datasets import load_diabetes

    # load data from sklearn.datasets
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target.reshape(X.shape[0], 1)  # one dimension numpy.ndarray
    y_random = y + np.random.random(y.shape)  # create another random np.ndarray
    y = np.hstack((y, y_random))

    model = LinearRegressionE()
    model.fit(X, y)
    y_predict = model.predict(X)
    f_test = model.f_test(X, y, y_predict)
    r_test = model.modified_r_test(X, y, y_predict)

    print('f_test: {0}'.format(f_test))
    print('modified_r_test: {0}'.format(r_test))
    print('-----check for LinearRegressionE finished-----')
