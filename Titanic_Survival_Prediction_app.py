import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
from tensorflow.keras.models import load_model

# Load model
loaded_model = load_model('Titanic_Ann.keras')

st.title('Titanic Disaster Survival Prediction Using ANN')
name = st.text_input("👤 Passenger Name:", value="Unnamed Passenger")
Pclass = st.selectbox("Passenger Class (Pclass):", [1, 2, 3], help="Proxy for socio-economic status: 1 = Upper, 2 = Middle, 3 = Lower")
Sex = st.selectbox("Sex of the Passenger:", ["male", "female"])
SibSp = st.number_input("Enter the number of family members(Siblings and spouse) of passenger:",min_value=0, step=1)
Age = st.number_input("Enter the age of the passenger (in years):",min_value=0, step=1)
Fare = st.number_input("Enter the passenger fare (in dollars):")
Embarked = st.selectbox("Port of Embarkation (Embarked):", ["C", "Q", "S"], help="C = Cherbourg, Q = Queenstown, S = Southampton")

input_dict = {'Sex': Sex,'Embarked': Embarked,'Pclass': Pclass,'SibSp': SibSp,'Age': Age,'Fare': Fare}

input_df = pd.DataFrame([input_dict])

# Manually encode (we're not using saved scaler/encoder)
# Encode Sex
if input_df.loc[0, 'Sex'] == 'male':
    input_df.loc[0, 'Sex_male'] = 1
else:
    input_df.loc[0, 'Sex_male'] = 0

# Encode Embarked
if input_df.loc[0, 'Embarked'] == 'Q':
    input_df.loc[0, 'Embarked_Q'] = 1
else:
    input_df.loc[0, 'Embarked_Q'] = 0

if input_df.loc[0, 'Embarked'] == 'S':
    input_df.loc[0, 'Embarked_S'] = 1
else:
    input_df.loc[0, 'Embarked_S'] = 0

# Drop original categorical columns
input_df.drop(['Sex', 'Embarked'], axis=1, inplace=True)

scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(input_df[['Pclass','Age','SibSp','Fare']]),columns=['Pclass','Age','SibSp','Fare'])

input_df[['Pclass','Age','SibSp','Fare']] = scaled_df[['Pclass','Age','SibSp','Fare']]

# Prediction only after button is clicked
if st.button("Predict Survival"):
    pred_prob = loaded_model.predict(input_df)
    pred_class = (pred_prob >= 0.5).astype(int)[0]

    if pred_class == 1:
        st.success(f"{name} is survived from this disaster.")
    else:
        st.error(f"{name} is not survived from this disaster.")
