from loss_function import *
from utils import *
import warnings

warnings.filterwarnings("ignore")


def DFR_MIL(learn_rate, hidden_N, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X, ma_avg):
    learn_rate = learn_rate

    linear1 = LinearLayer(feature_num, hidden_N)
    opt = Relu()
    linear2 = LinearLayer(hidden_N, 1)

    for epoch in range(epochs):
        LOSS = []
        for bag in range(np.shape(x_train)[0]):
            bag_name = x_train[bag]

            # X
            train_X = data_X[data_X['bag_names'] == bag_name].drop(columns=['bag_names'])
            # y
            train_label = y_train[bag]

            o0 = train_X
            a1 = linear1.forward(o0)
            o1 = opt.forward(a1)
            a2 = linear2.forward(o1)
            o2 = np.array(a2)

            for _ in range(np.shape(o2)[0]):
                if o2[_] > 1:
                    o2[_] = 1
                if o2[_] < 0:
                    o2[_] = 0

            loss = HL_loss(o2, train_label)

            grad = np.zeros((np.shape(o2)[0], 1))
            for _ in range(np.shape(o2)[0]):
                if o2[_] == 0:
                    grad[_] = (o2[_] - train_label)
                else:
                    grad[_] = (1 - np.square(train_label) / np.square(o2[_]))
                if grad[_] < -1:
                    grad[_] = -1

            grad = grad.reshape(o2.shape[0], 1)
            grad = linear2.backward(o1, grad)
            grad = opt.backward(a1, grad)
            grad = linear1.backward(o0, grad)

            linear1.update(learn_rate)
            linear2.update(learn_rate)
            LOSS.append(loss)

        model = [linear1, opt, linear2]

        pred_result = []
        for _ in range(np.shape(x_valid)[0]):
            bag_name = x_valid[_]
            test_X = data_X[data_X['bag_names'] == bag_name].drop(columns=['bag_names'])
            pred = predict(model, test_X)
            pred_result.append(pred)

        pred_result = pd.DataFrame(pred_result)
        y_valid = pd.DataFrame(y_valid)
        res = pd.concat([pred_result, y_valid], axis=1)
        res.columns = ["y_pred", "y_true"]
        y_pred = res.iloc[:, 0]
        y_true = res.iloc[:, 1]

        mse = mean_squared_error(y_true, y_pred)

        # Hosmer_Lemeshow_test
        HL_data = res
        HL_data.columns = ["y_hat", "y"]
        HL_value, pval = Hosmer_Lemeshow_test(HL_data)

        # constraint condition
        if HL_value < 0:
            HL_value = 0
        if pval > 1:
            pval = 1

        # Print Results
        print("##### epochs:{} ##### val_mse:{},HL_val:{},p_val为{}".format(epoch, mse, HL_value, pval))
        # early stop
        if mse <= 0.0001:
            break
    # save model
    model = [linear1, opt, linear2]

    # Calculate the threshold
    x_train_ = pd.DataFrame(x_train, columns=['bag_names'])
    bag_ma = pd.merge(x_train_, ma_avg, on='bag_names', how='left').set_index('bag_names')
    pred_result = []
    for _ in range(np.shape(x_train)[0]):
        bag_name = x_train[_]
        test_X = data_X[data_X['bag_names'] == bag_name].drop(columns=['bag_names'])
        pred = predict(model, test_X)
        pred_result.append(pred)

    pred_result = pd.DataFrame(pred_result)
    data_ = pd.concat([bag_ma.reset_index(drop=True), pred_result.reset_index(drop=True)], axis=1)
    data_.columns = ['ma', 'es']
    data_['es'] = data_['es'].apply(lambda x: 1 if x >= 0.5 else 0)

    ma_mean = data_['ma'].mean()
    print('ma_mean:', ma_mean)
    margin = 0.1
    Q = calculate_Q(data_, q_start=ma_mean-margin, q_end=ma_mean+margin)

    return model, Q


def Model_Building_huber_loss(learn_rate, hidden_N, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X, delta=0.1):

    learn_rate = learn_rate
    linear1 = LinearLayer(feature_num, hidden_N)
    opt = Relu()
    linear2 = LinearLayer(hidden_N, 1)

    for epoch in range(epochs):
        LOSS = []
        for bag in range(np.shape(x_train)[0]):
            bag_name = x_train[bag]

            # X
            train_X = data_X[data_X['bag_names'] == bag_name].drop(columns=['bag_names'])
            # y
            train_label = y_train[bag]

            o0 = train_X
            a1 = linear1.forward(o0)
            o1 = opt.forward(a1)
            a2 = linear2.forward(o1)
            o2 = np.array(a2)

            loss = huber_loss(train_label, o2, delta)
            error = train_label - o2

            if np.abs(error).mean() <= delta:
                grad = (o2 - train_label).reshape(o2.shape[0], 1)
            else:
                _ = np.sign(o2 - train_label)
                grad = _ / o2.shape[0]

            grad = linear2.backward(o1, grad)
            grad = opt.backward(a1, grad)
            grad = linear1.backward(o0, grad)

            linear1.update(learn_rate)
            linear2.update(learn_rate)
            LOSS.append(loss)

        model = [linear1, opt, linear2]
        pred_result = []
        for _ in range(np.shape(x_valid)[0]):
            bag_name = x_valid[_]
            test_X = data_X[data_X['bag_names'] == bag_name].drop(columns=['bag_names'])
            pred = predict(model, test_X)
            pred_result.append(pred)

        pred_result = pd.DataFrame(pred_result)
        y_valid = pd.DataFrame(y_valid)
        res = pd.concat([pred_result, y_valid], axis=1)
        res.columns = ["y_pred", "y_true"]
        y_pred = res.iloc[:, 0]
        y_true = res.iloc[:, 1]

        mse = mean_squared_error(y_true, y_pred)

        # Hosmer_Lemeshow_test
        HL_data = res
        HL_data.columns = ["y_hat", "y"]
        HL_value, pval = Hosmer_Lemeshow_test(HL_data)

        # constraint condition
        if HL_value < 0:
            HL_value = 0
        if pval > 1:
            pval = 1

        # Print Results
        # print("##### epochs:{} ##### val_mse:{},HL_val:{},p_val为{}".format(epoch, mse, HL_value, pval))
        # early stop
        if mse <= 0.0001:
            break
    # save model
    model = [linear1, opt, linear2]

    return model

def Model_Building_mse_loss(learn_rate, hidden_N, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X, delta=0.1):

    learn_rate = learn_rate
    linear1 = LinearLayer(feature_num, hidden_N)
    opt = Relu()
    linear2 = LinearLayer(hidden_N, 1)

    for epoch in range(epochs):
        LOSS = []
        for bag in range(np.shape(x_train)[0]):
            bag_name = x_train[bag]

            # X
            train_X = data_X[data_X['bag_names'] == bag_name].drop(columns=['bag_names'])
            # y
            train_label = y_train[bag]

            o0 = train_X
            a1 = linear1.forward(o0)
            o1 = opt.forward(a1)
            a2 = linear2.forward(o1)
            o2 = np.array(a2)

            loss = mse_loss(o2, train_label)

            grad = (o2 - train_label).reshape(o2.shape[0], 1)
            grad = linear2.backward(o1, grad)
            grad = opt.backward(a1, grad)
            grad = linear1.backward(o0, grad)

            linear1.update(learn_rate)
            linear2.update(learn_rate)
            LOSS.append(loss)

        model = [linear1, opt, linear2]
        pred_result = []
        for _ in range(np.shape(x_valid)[0]):
            bag_name = x_valid[_]
            test_X = data_X[data_X['bag_names'] == bag_name].drop(columns=['bag_names'])
            pred = predict(model, test_X)
            pred_result.append(pred)

        pred_result = pd.DataFrame(pred_result)
        y_valid = pd.DataFrame(y_valid)
        res = pd.concat([pred_result, y_valid], axis=1)
        res.columns = ["y_pred", "y_true"]
        y_pred = res.iloc[:, 0]
        y_true = res.iloc[:, 1]

        mse = mean_squared_error(y_true, y_pred)

        # Hosmer_Lemeshow_test
        HL_data = res
        HL_data.columns = ["y_hat", "y"]
        HL_value, pval = Hosmer_Lemeshow_test(HL_data)

        # constraint condition
        if HL_value < 0:
            HL_value = 0
        if pval > 1:
            pval = 1

        # Print Results
        # print("##### epochs:{} ##### val_mse:{},HL_val:{},p_val为{}".format(epoch, mse, HL_value, pval))
        # early stop
        if mse <= 0.0001:
            break
    # save model
    model = [linear1, opt, linear2]

    return model


def Model_Building_mae_loss(learn_rate, hidden_N, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X, delta=0.1):

    linear1 = LinearLayer(feature_num, hidden_N)
    opt = Relu()
    linear2 = LinearLayer(hidden_N, 1)

    for epoch in range(epochs):
        LOSS = []
        for bag in range(np.shape(x_train)[0]):
            bag_name = x_train[bag]

            # X
            train_X = data_X[data_X['bag_names'] == bag_name].drop(columns=['bag_names'])
            # y
            train_label = y_train[bag]

            o0 = train_X
            a1 = linear1.forward(o0)
            o1 = opt.forward(a1)
            a2 = linear2.forward(o1)
            o2 = np.array(a2)

            loss = mae_loss(o2,train_label)

            error = np.sign(o2 - train_label)
            grad = error / o2.shape[0]
            grad = linear2.backward(o1, grad)
            grad = opt.backward(a1, grad)
            grad = linear1.backward(o0, grad)

            linear1.update(learn_rate)
            linear2.update(learn_rate)
            LOSS.append(loss)

        model = [linear1, opt, linear2]
        pred_result = []
        for _ in range(np.shape(x_valid)[0]):
            bag_name = x_valid[_]
            test_X = data_X[data_X['bag_names'] == bag_name].drop(columns=['bag_names'])
            pred = predict(model, test_X)
            pred_result.append(pred)

        pred_result = pd.DataFrame(pred_result)
        y_valid = pd.DataFrame(y_valid)
        res = pd.concat([pred_result, y_valid], axis=1)
        res.columns = ["y_pred", "y_true"]
        y_pred = res.iloc[:, 0]
        y_true = res.iloc[:, 1]

        mse = mean_squared_error(y_true, y_pred)

        # Hosmer_Lemeshow_test
        HL_data = res
        HL_data.columns = ["y_hat", "y"]
        HL_value, pval = Hosmer_Lemeshow_test(HL_data)

        # constraint condition
        if HL_value < 0:
            HL_value = 0
        if pval > 1:
            pval = 1

        # Print Results
        # print("##### epochs:{} ##### val_mse:{},HL_val:{},p_val为{}".format(epoch, mse, HL_value, pval))
        # early stop
        if mse <= 0.0001:
            break
    # save model
    model = [linear1, opt, linear2]

    return model

def Model_Building_logcosh_loss(learn_rate, hidden_N, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X, delta=0.1):

    def log_cosh_loss_gradient(y_true, y_pred):
        return np.tanh(y_pred - y_true)
    linear1 = LinearLayer(feature_num, hidden_N)
    opt = Relu()
    linear2 = LinearLayer(hidden_N, 1)

    for epoch in range(epochs):
        LOSS = []
        for bag in range(np.shape(x_train)[0]):
            bag_name = x_train[bag]

            # X
            train_X = data_X[data_X['bag_names'] == bag_name].drop(columns=['bag_names'])
            # y
            train_label = y_train[bag]

            o0 = train_X
            a1 = linear1.forward(o0)
            o1 = opt.forward(a1)
            a2 = linear2.forward(o1)
            o2 = np.array(a2)

            loss = log_cosh_loss(o2, train_label)

            grad = log_cosh_loss_gradient(train_label, o2).reshape(o2.shape[0], 1)
            grad = linear2.backward(o1, grad)
            grad = opt.backward(a1, grad)
            grad = linear1.backward(o0, grad)

            linear1.update(learn_rate)
            linear2.update(learn_rate)
            LOSS.append(loss)

        model = [linear1, opt, linear2]
        pred_result = []
        for _ in range(np.shape(x_valid)[0]):
            bag_name = x_valid[_]
            test_X = data_X[data_X['bag_names'] == bag_name].drop(columns=['bag_names'])
            pred = predict(model, test_X)
            pred_result.append(pred)

        pred_result = pd.DataFrame(pred_result)
        y_valid = pd.DataFrame(y_valid)
        res = pd.concat([pred_result, y_valid], axis=1)
        res.columns = ["y_pred", "y_true"]
        y_pred = res.iloc[:, 0]
        y_true = res.iloc[:, 1]

        merged_df = pd.concat([y_pred, y_true], axis=1)
        cleaned_df = merged_df.dropna()
        y_p = cleaned_df.iloc[:, 0]
        y_t = cleaned_df.iloc[:, 1]

        mse = mean_squared_error(y_t, y_p)

        # Hosmer_Lemeshow_test
        HL_data = res
        HL_data.columns = ["y_hat", "y"]
        HL_value, pval = Hosmer_Lemeshow_test(HL_data)

        # constraint condition
        if HL_value < 0:
            HL_value = 0
        if pval > 1:
            pval = 1

        # Print Results
        # print("##### epochs:{} ##### val_mse:{},HL_val:{},p_val为{}".format(epoch, mse, HL_value, pval))
        # early stop
        if mse <= 0.0001:
            break
    # save model
    model = [linear1, opt, linear2]

    return model

