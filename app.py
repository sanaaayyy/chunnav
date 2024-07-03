import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression

# Function to encode categorical variables
def My_Encoder(idx, dict_pc_name):
    if dict_pc_name.get(idx) is not None:
        return dict_pc_name.get(idx)
    else:
        temp = len(dict_pc_name) + 1
        dict_pc_name[idx] = temp
        return temp

# Function to clean and preprocess the data
def preprocess_data(df):
    dict_pc_name = {}
    df['New_Pc_name'] = df['Pc_name'].apply(lambda i: My_Encoder(i, dict_pc_name))
    df1 = df.drop(['Pc_name','no','Turnout','margin','margin%','year'],axis=1)

    regex = r'(?<!\[)[^\[\]]+(?!\])'
    def Changer(str1):
        matches = re.findall(regex, str1)
        temp = matches[0]
        temp = re.sub(r'^\s+|\s+$', '', temp)
        return temp

    df1['state'] = df['state'].apply(lambda i: Changer(i))
    encoder = LabelEncoder()
    df1['New_State'] = encoder.fit_transform(df1['state'])
    df1['New_type'] = encoder.fit_transform(df1['type'])
    df1['New_candidate_name'] = encoder.fit_transform(df1['candidate_name'])
    df1['New_party'] = encoder.fit_transform(df1['party'])

    df2 = df1.drop(['state','type','candidate_name','party'],axis=1)

    def Converting(i):
        i = i.replace(",","")
        try:
            i = float(i)
        except:
            i = 0
        return i    

    df2['votes_new'] = df2['votes'].apply(lambda i: Converting(i))
    df2['electors_new'] = df2['electors'].apply(lambda i: Converting(i))

    df3 = df2.drop(['electors','votes'],axis=1)

    scaler = MinMaxScaler()
    df3['electors_new'] = scaler.fit_transform(df3[['electors_new']])
    df3['votes_new'] = scaler.fit_transform(df3[['votes_new']])
    
    return df3

# Streamlit App
st.title('Prediction Model of Exit Polls and Opinion Polls in India')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.write(df.head())

    df_processed = preprocess_data(df)
    st.write("Preprocessed data preview:")
    st.write(df_processed.head())

    X = df_processed.drop(['votes_new'], axis=1)
    Y = df_processed['votes_new']

    # Train the model
    regressor = LinearRegression()
    regressor.fit(X, Y)

    # Save the model
    filename = 'PredExitPoll.pkl'
    pickle.dump(regressor, open(filename, 'wb'))

    st.success("Model trained and saved as PredExitPoll.pkl")

    # Load the model and make predictions
    loaded_model = pickle.load(open(filename, 'rb'))

    st.header("Make a Prediction")
    new_data = {
        "New_Pc_name": st.number_input("New_Pc_name", min_value=0, key="New_Pc_name"),
        "New_State": st.number_input("New_State", min_value=0, key="New_State"),
        "New_type": st.number_input("New_type", min_value=0, key="New_type"),
        "New_candidate_name": st.number_input("New_candidate_name", min_value=0, key="New_candidate_name"),
        "New_party": st.number_input("New_party", min_value=0, key="New_party"),
        "electors_new": st.number_input("electors_new", min_value=0.0, key="electors_new"),
    }

    new_data_df = pd.DataFrame([new_data])

    if st.button("Predict"):
        prediction = loaded_model.predict(new_data_df)
        st.write(f"Predicted votes: {prediction[0]}")
