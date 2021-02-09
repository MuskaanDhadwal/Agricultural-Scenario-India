import pandas as  pd
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split

df=pd.read_csv('Data/streamlit_data.csv')

df_test=df.sample(frac = 0.1)
# Model Training

X=df_test[['Crop_Year','Avg_Temp', 'Avg_Rain','states_id', 'crop_id', 'Area',]].values

df_test.Production = df_test.Production.astype(int)

y=df_test[['Production']].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.70)

clf = svm.SVC(kernel='linear') # Linear Kernel

clf.fit(X_train,y_train.ravel())

# Pickling the model

pickle.dump(clf, open('svm_clf.pkl', 'wb'))
