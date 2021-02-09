import pandas as  pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


df=pd.read_csv('Data/streamlit_data.csv')


# Model Training

X=df[['Crop_Year','Avg_Temp', 'Avg_Rain','states_id', 'crop_id', 'Area_10000',]].values

df.Prod_1000000 = df.Prod_1000000.astype(int)

y=df[['Prod_1000000']].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.70)

clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train,y_train.ravel())


# Pickling the model

pickle.dump(clf, open('rf_clf.pkl', 'wb'))
