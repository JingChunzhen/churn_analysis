from data_analysis import Additional_Features_Extractor
from data_analysis import Action_Feature_Extractor
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import sklearn
from sklearn import metrics
import numpy as np

# X, Y = Additional_Features_Extractor(file_in='./data/kzby.db').load(file_in=['./output/fc_train.txt', './output/fc_label.pkl'])
# print(np.shape(X)) # (7142)
X, Y, _ = Action_Feature_Extractor(file_in='./data/kzby.db').load(file_in=['./output/op_feature/sc_train.txt', './output/op_feature/sc_label.pkl'])
print(np.shape(X)) # (7142)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
X_train, X_validate, Y_train, Y_validate = train_test_split(
    X_train, Y_train, test_size=0.2)

# Additional_Features_Extractor(file_in='./data/kbzy.db').check(file_in=['./output/fc_train.txt', './output/fc_label.pkl'])
# X, Y = Additional_Features_Extractor(file_in='./data/kbzy.db').load(file_in=['./output/fc_train.txt', './output/fc_label.pkl'])
# #X, Y, _ = Action_Feature_Extractor(file_in='./data/kbzy.db').load(file_in=['./output/op_feature/fc_train.txt', './output/op_feature/fc_label.pkl'])
# print(np.shape(X))  # -> 17965 2069
# print(len(Y)) # -> 17965 

def xgb_model():
    '''
    '''
    model = xgb.XGBClassifier(
        learning_rate=0.1, n_estimators=100, max_depth=10, subsample=1)
    eval_set = [(X_validate, Y_validate)]
    model.fit(X_train, Y_train, early_stopping_rounds=20,
                eval_metric="logloss", eval_set=eval_set, verbose=True)

    Y_pred = model.predict(X_train)  # Y_pred -> np.ndarray Y_train -> list
    print('training score {}'.format(accuracy_score(Y_train, Y_pred)))
    Y_pred = model.predict(X_validate)
    print('validate score {}'.format(accuracy_score(Y_validate, Y_pred)))
    Y_pred = model.predict(X_test)
    print('test score {}'.format(accuracy_score(Y_test, Y_pred)))
    print(precision_recall_fscore_support(Y_test, Y_pred, average=None))
    pass

if __name__ == '__main__':
    xgb_model()
    pass