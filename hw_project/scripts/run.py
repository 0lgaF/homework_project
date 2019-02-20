import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from simple.feature_extractor import Feature_extractor
from simple.learner import Train
from simple.preds import make_preds
from simple.score import precision_score


if __name__ == '__main__':
    df = pd.read_csv('../../HR.csv')
    FEATURES_NAMES = [
    'last_evaluation', 'number_project', 
    'average_montly_hours', 'time_spend_company', 
    'Work_accident', 'promotion_last_5years'
    ]
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES_NAMES], df.left, test_size=0.33, random_state=42)
    model = make_pipeline(Feature_extractor(), LogisticRegression())
    train = Train()
    preds = make_preds(train(model, X_train, y_train), X_test)
    print(precision_score(y_test, preds))







