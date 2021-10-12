from pandas._config.config import options
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error




header = st.container()
dataset = st.container()
features = st.container()
model_traning = st.container()


st.markdown(

    '''
    <style>
    .main{
        background-color: #F5F5F5;
    }
    </style>
    ''',
    unsafe_allow_html= True
)

@st.cache
def get_data(filename):
    dataset=pd.read_csv(filename)

    return dataset




with header:
    st.title('Hello Hello Hello!!!!!!!!!!!!!!!!!!!!1')
    st.text('In this project i look into the transitions of texies in new york......')


with dataset:
    st.header(' New York city texy dataset')
    #dataset=pd.read_csv('housing.csv')

    dataset = get_data('data/housing.csv')
    st.write(dataset.head(5))
    
    Price = dataset['PRICE']
    
    st.bar_chart(Price)


with features:
    st.header('The features I created')
    st.markdown('* **First Feature** I created this feature because of this.....I calculated it using this logic.....*')
    st.markdown('* **Second Feature** I created this feature because of this.....I calculated it using this logic.....*')



with model_traning:
    st.header('Time to create model')

    sel_col, disp_col = st.columns(2)
    random_state = sel_col.slider('What should be the max dept of the model', min_value = 10, max_value = 100, value = 20, step = 10 )

    n_estimators = sel_col.selectbox('How many trees should be there', options = [100,200,300,400,'No Limit'],index = 0)

    # Fitting Random Forest Regression to the Training set
    from sklearn.ensemble import RandomForestRegressor

    

    if n_estimators == 'No Limit':
        regressor = RandomForestRegressor(random_state = random_state)
    else:
        regressor = RandomForestRegressor(n_estimators = n_estimators, random_state = random_state)

    #x = dataset.drop('PRICE', axis = 1)
    #y = dataset['PRICE']

    sel_col.text('Here is a list of features in my data:')
    sel_col.write(dataset.columns)

    input_feature = sel_col.text_input('Which feature should be use as input feature','ZN')

    #x = dataset.drop('PRICE', axis = 1)
    x = dataset[[input_feature]]
    y = dataset['PRICE']

    


    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state =  random_state)


 
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    # Evaluating the Algorithm
    from sklearn import metrics


    disp_col.subheader('Mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(y_test, y_pred))


    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write(mean_squared_error(y_test,y_pred))


    disp_col.subheader('R square score of the model is:')
    disp_col.write(r2_score(y_test, y_pred))
