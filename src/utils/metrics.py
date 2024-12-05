from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def binarize_target(train_df, test_df, category_col_name, return_binarizer=False):
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit([train_df[category_col_name].to_list()])
    train_multi = [[el] for el in train_df[category_col_name].to_list()]
    test_multi = [[el] for el in test_df[category_col_name].to_list()]
    y_train = multilabel_binarizer.transform(train_multi)
    y_test = multilabel_binarizer.transform(test_multi)
    class_names = multilabel_binarizer.classes_
    if return_binarizer:
        return y_train, y_test, class_names, multilabel_binarizer
    else:
        return y_train, y_test, class_names


def eval_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return f1, precision, recall
