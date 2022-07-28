import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np

st.header("Fish Weight Prediction App")
#st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("https://raw.githubusercontent.com/gurokeretcha/WishWeightPredictionApplication/master/Fish.csv")
#load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

tab1, tab2 = st.tabs(["Prediction", "Data"])


# You can also use "with" notation:
with tab1:
    st.subheader("Please select values of fish weight prediction!")
    # left_column, right_column = st.columns(2)

    # with left_column:
    #     inp_species = st.radio(
    #         'Name of the fish:',
    #         np.unique(data['Species']))
    inp_species = st.selectbox("Name of fish:", np.unique(data['Species']))

    input_Length1 = st.slider('Vertical length(cm)', min(data["Length1"]), max(data["Length1"]), 1.0)
    input_Length2 = st.slider('Diagonal length(cm)', min(data["Length2"]), max(data["Length2"]), 1.0)
    input_Length3 = st.slider('Cross length(cm)', min(data["Length3"]), max(data["Length3"]), 1.0)
    input_Height = st.slider('Height(cm)', min(data["Height"]), max(data["Height"]), 1.0)
    input_Width = st.slider('Diagonal width(cm)', min(data["Width"]), max(data["Width"]), 1.0)


    if st.button('Make Prediction'):
        input_species = encoder.transform(np.expand_dims(inp_species, -1))
        inputs = np.expand_dims(
            [int(input_species), input_Length1, input_Length2, input_Length3, input_Height, input_Width], 0)
        prediction = best_xgboost_model.predict(inputs)
        print("final pred", np.squeeze(prediction, -1))
        st.write(f"Your fish weight is: {np.squeeze(prediction, -1):.2f}g")

        #st.write(f"Thank you {st.session_state.name}! I hope you liked it.")

with tab2:
    data
    


