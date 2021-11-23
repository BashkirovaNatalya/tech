from sklearn import model_selection, datasets, linear_model, metrics
# TODO ask why those??^^^
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('one\Customer_Churn.csv')

del df['customerID']

def change_obj_to_code(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype('category').cat.codes

change_obj_to_code(df)

X = df[['gender', 'SeniorCitizen', 'PhoneService', 'MultipleLines', 'InternetService', 'Partner', 'Dependents', 'PaymentMethod']]

y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)



def use_classificator(X_train, X_test, y_train, y_test, classificator):
    classificator.fit(X_train, y_train)
    y_pred = classificator.predict(X_test)


    result_score = metrics.f1_score(y_test, y_pred)

    print('nf1 score: ', result_score)
    print('acc score:', accuracy_score(y_test, y_pred))

use_classificator(X_train, X_test, y_train, y_test,  MultinomialNB())
use_classificator(X_train, X_test, y_train, y_test,  DecisionTreeClassifier())
use_classificator(X_train, X_test, y_train, y_test,  MLPClassifier())


