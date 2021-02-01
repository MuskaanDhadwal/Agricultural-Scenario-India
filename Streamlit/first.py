import streamlit as st
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Dataset import
df=pd.read_csv('Data/streamlit_data.csv')

State_id={'Andhra Pradesh': 2,
 'Arunachal Pradesh': 3,
 'Assam': 4,
 'Bihar': 5,
 'Chandigarh': 6,
 'Chhattisgarh': 7,
 'Dadra and Nagar Haveli': 8,
 'Daman and Diu': 9,
 'Delhi': 10,
 'Goa': 11,
 'Gujarat': 12,
 'Haryana': 13,
 'Himachal Pradesh': 14,
 'Jharkhand': 16,
 'Karnataka': 17,
 'Kerala': 18,
 'Lakshadweep': 19,
 'Madhya Pradesh': 20,
 'Maharashtra': 21,
 'Manipur': 22,
 'Meghalaya': 23,
 'Mizoram': 24,
 'Nagaland': 25,
 'Puducherry': 27,
 'Punjab': 28,
 'Rajasthan': 29,
 'Sikkim': 30,
 'Tamil Nadu': 31,
 'Tripura': 32,
 'Uttar Pradesh': 33,
 'West Bengal': 35,
 'Andaman and Nicobar Islands': 1,
 'Jammu and Kashmir ': 15,
 'Odisha': 26,
 'Uttarakhand': 34}

crop_id={'Arecanut': 1,
 'Other Kharif pulses': 2,
 'Rice': 3,
 'Banana': 4,
 'Cashewnut': 5,
 'Coconut ': 6,
 'Dry ginger': 7,
 'Sugarcane': 8,
 'Sweet potato': 9,
 'Tapioca': 10,
 'other oilseeds': 11,
 'Arhar/Tur': 12,
 'Bajra': 13,
 'Castor seed': 14,
 'Cotton(lint)': 15,
 'Dry chillies': 16,
 'Gram': 17,
 'Groundnut': 18,
 'Horse-gram': 19,
 'Jowar': 20,
 'Korra': 21,
 'Maize': 22,
 'Moong(Green Gram)': 23,
 'Onion': 24,
 'other misc. pulses': 25,
 'Ragi': 26,
 'Samai': 27,
 'Sesamum': 28,
 'Small millets': 29,
 'Sunflower': 30,
 'Urad': 31,
 'Linseed': 32,
 'Safflower': 33,
 'Wheat': 34,
 'Coriander': 35,
 'Potato': 36,
 'Tobacco': 37,
 'Turmeric': 38,
 'Mesta': 39,
 'Other  Rabi pulses': 40,
 'Rapeseed &Mustard': 41,
 'Niger seed': 42,
 'Varagu': 43,
 'Oilseeds total': 44,
 'Pulses total': 45,
 'Jute': 46,
 'Barley': 47,
 'Khesari': 48,
 'Masoor': 49,
 'Peas & beans (Pulses)': 50,
 'Garlic': 51,
 'Soyabean': 52,
 'Sannhamp': 53,
 'Moth': 54,
 'Guar seed': 55,
 'Other Cereals & Millets': 56,
 'Black pepper': 57,
 'Cardamom': 58,
 'Kapas': 59,
 'Tea': 60,
 'Jute & mesta': 61,
 'Rubber': 62,
 'Coffee': 63,
 'Beans & Mutter(Vegetable)': 64,
 'Bhindi': 65,
 'Brinjal': 66,
 'Citrus Fruit': 67,
 'Cucumber': 68,
 'Grapes': 69,
 'Mango': 70,
 'Orange': 71,
 'other fibres': 72,
 'Other Fresh Fruits': 73,
 'Other Vegetables': 74,
 'Papaya': 75,
 'Pome Fruit': 76,
 'Tomato': 77,
 'Cabbage': 78,
 'Peas  (vegetable)': 79,
 'Bottle Gourd': 80,
 'Pineapple': 81,
 'Turnip': 82,
 'Carrot': 83,
 'Redish': 84,
 'Arcanut (Processed)': 85,
 'Atcanut (Raw)': 86,
 'Cashewnut Processed': 87,
 'Cashewnut Raw': 88,
 'Bitter Gourd': 89,
 'Drum Stick': 90,
 'Jack Fruit': 91,
 'Snak Guard': 92,
 'Cauliflower': 93,
 'Water Melon': 94,
 'Ash Gourd': 95,
 'Beet Root': 96,
 'Lab-Lab': 97,
 'Other Citrus Fruit': 98,
 'Pome Granet': 99,
 'Ribed Guard': 100,
 'Yam': 101,
 'Pump Kin': 102,
 'Apple': 103,
 'Peach': 104,
 'Pear': 105,
 'Plums': 106,
 'Ber': 107,
 'Litchi': 108,
 'Ginger': 109,
 'Cowpea(Lobia)': 110,
 'Paddy': 111,
 'Total foodgrain': 112,
 'Blackgram': 113,
 'Cond-spcs other': 114,
 'Lemon': 115,
 'Sapota': 116}



st.write("""
# Agriculture Scenario of India.
A Data Science Approach
""")

# Dataset head
st.write("""
## Dataset Used
""")

st.write(df.head(10))

# Dataset Description
st.write("""
## Dataset Description
""")

st.write(df.describe())

# Dataset Columns
st.write("""
## Dataset Columns
""")

st.write(df.columns)



# Side Bar

st.sidebar.header('User Input Parameters')

option_state=st.sidebar.selectbox('Select State',df.State_Name.unique())
id_state=State_id[option_state]
st.sidebar.write('State selected is: ',option_state)
st.sidebar.write('State Id is: ',id_state)

option_Crop=st.sidebar.selectbox('Select Crop',df.Crop.unique())
id_crop=crop_id[option_Crop]
st.sidebar.write('Crop selected is: ',option_Crop)
st.sidebar.write('Crop Id is: ',id_crop)

option_year=st.sidebar.selectbox('Select Year',[1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012])
st.sidebar.write('Year selected is: ',option_year)

option_Area=st.sidebar.slider('Select Area',0.01,80000.0,1000.0)
st.sidebar.write('Area selected is: ',option_Area)

option_Temp=st.sidebar.slider('Select Temperature',24.10,25.13)
st.sidebar.write('Temperature selected is: ',option_Temp)

option_Rain=st.sidebar.slider('Select Rain Measure',920.80,1243.50)
st.sidebar.write('Rain Measure selected is: ',option_Rain)


# selected Parameters

data={'State_Name':id_state,
'Crop':id_crop,
'Year':option_year,
'Area':option_Area,
'Temp':option_Temp,
'Rain':option_Rain}

df_selected=pd.DataFrame(data,index=[0])

st.subheader('User Input Parameters')
st.write(df_selected)

# Model Training

X=df[['Crop_Year','Avg_Temp', 'Avg_Rain','states_id', 'crop_id', 'Area_10000',]].values

df.Prod_1000000 = df.Prod_1000000.astype(int)

y=df[['Prod_1000000']].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.70)

clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train,y_train.ravel())

y_pred = clf.predict(X_test)

st.subheader('Random Forest Classifier')
st.write("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

st.write("User Input Predictions")
u_pred=clf.predict(df_selected)
st.write('Amount of Production Predicted : ' , u_pred)
