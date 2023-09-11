import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import mean_squared_error

######################## Layer ########################
class LinearLayer:
    def __init__(self, input_D, output_D):
        np.random.seed(111)
        self._W = np.random.normal(0, 0.1, (input_D, output_D))
        np.random.seed(111)
        self._b = np.random.normal(0, 0.1, (1, output_D))
        self._grad_W = np.zeros((input_D, output_D))
        self._grad_b = np.zeros((1, output_D))

    def forward(self, X):
        return np.matmul(X, self._W) + self._b

    def backward(self, X, grad):
        self._grad_W = np.matmul(X.T, grad)
        self._grad_b = np.matmul(grad.T, np.ones(X.shape[0]))
        return np.matmul(grad, self._W.T)

    def update(self, learn_rate):
        self._W = self._W - self._grad_W * learn_rate


######################## Activation Function ########################
class Relu:
    def __init__(self):
        pass

    def forward(self, X):
        return np.where(X < 0, 0, X)

    def backward(self, X, grad):
        return np.where(X > 0, X, 0) * grad


######################## Hosmer_Lemeshow_test ########################
def Hosmer_Lemeshow_test(data, Q=10):
    '''
    data: dataframe format, with ground_truth label name is y,
                                 prediction value column name is y_hat
    '''
    data = data.sort_values('y_hat')
    data['Q_group'] = pd.qcut(data['y_hat'], Q, duplicates="drop")

    y_p = data['y'].groupby(data.Q_group).sum()
    y_total = data['y'].groupby(data.Q_group).count()
    y_n = y_total - y_p

    y_hat_p = data['y_hat'].groupby(data.Q_group).sum()
    y_hat_total = data['y_hat'].groupby(data.Q_group).count()
    y_hat_n = y_hat_total - y_hat_p

    hltest = (((y_p - y_hat_p) ** 2 / y_hat_p) + ((y_n - y_hat_n) ** 2 / y_hat_n)).sum()
    pval = 1 - chi2.cdf(hltest, Q - 2)

    return hltest, pval


def predict(model, X):
    tmp = X
    for layer in model:
        tmp = layer.forward(tmp)
    res = np.mean(tmp)
    if res < 0:
        res = 0
    if res > 1:
        res = 1
    return res


def calculate_Q(df, q_start, q_end,  step_size=0.005):
    q = q_start
    max_dv = 0
    max_q = None

    while q < q_end:
        group1 = df[df['ma'] < q]
        group2 = df[df['ma'] >= q]
        group1 = group1['es'].mean()
        group2 = group2['es'].mean()
        dv = abs(group1 - group2)
        if dv >= max_dv:
            max_dv = dv
            max_q = q
        q += step_size
    return max_q


def prediction(data_X, x_test,y_test, model):
    y_hat = []
    for bag in range(np.shape(x_test)[0]):
        bag_name = x_test[bag]
        test_X = data_X[data_X['bag_names'] == bag_name].drop(columns=['bag_names'])
        pred = predict(model, test_X)
        y_hat.append(pred)

    # Prediction results
    y_hat = pd.DataFrame(y_hat)
    y = pd.DataFrame(y_test)
    res = pd.concat([y_hat, y], axis=1)
    res.columns = ["y_hat", "y"]
    # print(res)
    test_mse = mean_squared_error(res.iloc[:, 0], res.iloc[:, 1])
    HL_res = Hosmer_Lemeshow_test(res)
    HL_value = HL_res[0]
    p_val = HL_res[1]
    print("mse:{}:hl_value:{},p_val:{}".format(test_mse, HL_value, p_val))
    return y_hat