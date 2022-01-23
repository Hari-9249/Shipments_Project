import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
st.title('Product Shipment')
st.sidebar.header('User Input Parameters')

def user_input_features():
    Warehouse_block=st.sidebar.selectbox('Warehouse Block',('0','1','2','3','4'))
    Mode_of_Shipment=st.sidebar.selectbox('Mode Of Shipment',('0','1','2'))
    Gender=st.sidebar.selectbox('Gender',('0','1'))
    Prior_purchases=st.sidebar.number_input("Insert The Number Of Prior Purchase")
    Discount_offered=st.sidebar.number_input("Insert The Discount Offered")
    Weight_in_gms=st.sidebar.number_input("Insert The Product Weight")
    data={'Warehouse_block':Warehouse_block,
          'Mode_of_Shipment':Mode_of_Shipment,
          'Gender':Gender,
          'Prior_purchases':Prior_purchases,
          'Discount_offered':Discount_offered,
          'Weight_in_gms':Weight_in_gms}
    feature=pd.DataFrame(data,index = [0])
    return feature

df=user_input_features()
st.subheader('User Input Parameter')
st.write(df)

shipment=pd.read_csv("D:\shipments3.csv")
shipment=shipment.dropna()

X=shipment.iloc[:,1:7]
Y=shipment.iloc[:,7]
clf=RandomForestClassifier()
clf.fit(X,Y)


prediction=clf.predict(df)


st.subheader('Predicted Result')
st.write('Will Reach On Time'if prediction == 0 else 'Will Not Reach On Time')

