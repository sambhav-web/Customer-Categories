import dill
import streamlit as st
import numpy as np

with open('model.pkl','rb') as f:
    model=dill.load(f)
with open('scaler.pkl','rb') as f:
    scaler=dill.load(f)
with open('pca.pkl','rb') as f:
    pca=dill.load(f)

st.title('Customer Clustering App')
st.write('Enter the customer details: ')

st.header('Customer Age')
age=st.number_input('Age',10,65,step=1)
st.header('Customer Spent Time')
time=st.number_input('Average Spend',5,50)
st.header('Visits per week')
visits=st.number_input('Visits per week',1,7,step=1)
st.header('Promotion Interest')
interest=st.number_input('Promotion Interest',1,11,step=1)

def predict(age,time,visits,interest):
    input=(age,time,visits,interest)
    input_array=np.array(input).reshape(1,-1)
    input_x=scaler.transform(input_array)
    input_x=pca.transform(input_x)
    return model.predict(input_x)

def class_cluster(output):
    if output==0:
        return 'Daily'
    elif output==1:
        return 'Weekend'
    else:
        return 'Promotion'

if st.button('Predict Cluster'):
    prediction=predict(age,time,visits,interest)
    st.success(f'The customer belongs to the {class_cluster(prediction[0])} cluster')