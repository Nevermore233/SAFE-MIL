from datetime import datetime
from sklearn.model_selection import train_test_split
from model import *
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset600', type=bool, default=True, help='choose dataset')
parser.add_argument('--compare', type=bool, default=False, help='compare with other loss function')
parser.add_argument('--l', type=float, default=0.001, help='learning rate')
parser.add_argument('--h', type=float, default=50, help='hidden_num')
parser.add_argument('--e', type=float, default=200, help='epochs')
parser.add_argument('--f', type=float, default=5, help='feature_num')
args = parser.parse_args()


def main():
    # file path
    data_path1 = "data_prepare/output/NSCLC1_600.csv.csv"
    data_path2 = "data_prepare/output/NSCLC1_1000.csv.csv"
    pfs_avg_path = "data_prepare/output/pfs_avg.csv"
    ma_avg_path = "data_prepare/output/ma_avg.csv"
    if args.dataset600:
        data = pd.read_csv(data_path1, index_col=0, encoding="utf-8")
    else:
        data = pd.read_csv(data_path2, index_col=0, encoding="utf-8")

    pfs_avg = pd.read_csv(pfs_avg_path, index_col=0, encoding="utf-8")
    ma_avg = pd.read_csv(ma_avg_path, index_col=0, encoding="utf-8")

    data_X = data.drop(columns=['label', 'bag_labels'])
    df = data.loc[:, ["bag_names", "bag_labels"]].drop_duplicates(['bag_names'])
    bag_names, bag_labels = df.iloc[:, 0], df.iloc[:, 1]
    x_train_all, x_test, y_train_all, y_test = train_test_split(
        bag_names, bag_labels)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_all, y_train_all)

    x_train, x_test, y_train, y_test, x_valid, y_valid = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), np.array(x_valid), np.array(y_valid)

    lr = args.l
    hidden_num = args.h
    epochs = args.e
    feature_num = args.f

    ###### hosmer-lemeshow loss

    model, Q = DFR_MIL(lr, hidden_num, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X, ma_avg)
    print('threshold:', Q)

    if args.dataset600:
        print('The resuls of dataset600:')
    else:
        print('The resuls of dataset1000:')
    print('##### hosmer-lemeshow loss #####')
    y_hat = prediction(data_X, x_test, y_test, model)

    if args.dataset600:
        # main_result
        y = pd.DataFrame(y_test)
        x_test_ = pd.DataFrame(x_test, columns=['bag_names'])
        test_pfs_avg = pd.merge(x_test_, pfs_avg, on='bag_names', how='left').set_index('bag_names')
        test_ma_avg = pd.merge(x_test_, ma_avg, on='bag_names', how='left').set_index('bag_names')
        main_result = pd.concat([test_pfs_avg.reset_index(drop=True), y.reset_index(drop=True), y.reset_index(drop=True), y_hat.reset_index(drop=True), test_ma_avg.reset_index(drop=True),  test_ma_avg.reset_index(drop=True)], axis=1)
        main_result.columns = ['pfs', 'status', 'y', 'y_pred', 'mutation abundance', 'group']
        main_result['status'] = main_result['status'].apply(lambda x: 0 if x >= 0.5 else 1) # Patients with efficacy greater than 0.5 have a status of 0.
        main_result['group'] = main_result['group'].apply(lambda x: 1 if x >= Q else 0)
        main_result['dfr'] = 1 - main_result['y']
        main_result['dfr_pred'] = 1 - main_result['y_pred']
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_result.to_csv(f'data_prepare/output/main_result_{current_time}.csv')
        # Then, 'main_result.csv' will be used for survival analysis using GraphPad Prism 8 software.

    #################################### compare ####################################
    if args.compare:
        ###### huber loss
        print('##### huber loss #####')
        model_huber = Model_Building_huber_loss(lr, hidden_num, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X)
        _ = prediction(data_X, x_test, y_test, model_huber)
        ##### mse loss
        print('##### mse loss #####')
        model_mse = Model_Building_mse_loss(lr, hidden_num, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X)
        _ =prediction(data_X, x_test, y_test, model_mse)
        ###### mae loss
        print('##### mae loss #####')
        model_mae = Model_Building_mae_loss(lr, hidden_num, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X)
        _ =prediction(data_X, x_test, y_test, model_mae)
        ##### logcosh loss
        print('##### logcosh loss #####')
        model_logcosh = Model_Building_logcosh_loss(lr, hidden_num, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X)
        _ =prediction(data_X, x_test, y_test, model_logcosh)

if __name__ == '__main__':
    main()
