from pandas._config.config import options
import streamlit as st
import pandas as pd
import numpy as np
import os 
import sys
from io import BytesIO, StringIO

import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer

import io


#Wordcloud
import re
import operator
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords

set(stopwords.words('english'))
from wordcloud import WordCloud
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import streamlit.components.v1 as stc




st.markdown(

    '''
    <style>
    .main{
        background-color: #FFFFFF;
    }
    </style>
    ''',
    unsafe_allow_html= True
)


def main():

    st.title(" ---->Jobshie<----")
    st.title("")
    st.title("Find out the most important keywords in the description of a job.")
    st.title("")
    st.subheader("Copy the full job description and paste it down below:")
    sel_col, disp_col = st.columns(2)
    job_description = sel_col.text_input('',' Paste your full job description here ')

    if st.button("--->SHOW ME THE RESULTS<---"):

	



        def tokenizer(text):
            ''' a function to create a word cloud based on the input text parameter'''
            ## Clean the Text
            # Lower
            clean_text = text.lower()
            # remove punctuation
            clean_text = re.sub(r'[^\w\s]', '', clean_text)
            # remove trailing spaces
            clean_text = clean_text.strip()
            # remove numbers
            clean_text = re.sub('[0-9]+', '', clean_text)
            # tokenize 
            clean_text = word_tokenize(clean_text)
            # remove stop words
            stop = stopwords.words('english')
            clean_text = [w for w in clean_text if not w in stop] 
            return(clean_text)


        job_description = tokenizer(job_description)
        job_description1 = ' '.join(job_description)
    

        # initializing dict to store frequency of each element
        elements_count = {}
        # iterating over the elements for frequency
        for element in job_description:
        # checking whether it is in the dict or not
            if element in elements_count:
                # incerementing the count by 1
                elements_count[element] += 1
            else:
                # setting the count to 1
                elements_count[element] = 1
            # printing the elements frequencies

        for key, value in elements_count.items():
        
            print(f"{key}: {value}")


        df =pd.DataFrame(elements_count.items(), columns=['Word', 'Frequency'])
        df.index += 1
        st.write(df)

        Frequency = df['Frequency']
        Word = df['Word']
        
        if job_description is not None:

            def create_word_cloud(jd):
                corpus = jd
                fdist = FreqDist(corpus)
                #print(fdist.most_common(100))
                words = ' '.join(corpus)
                words = words.split()
                    
                    # create a empty dictionary  
                data = dict() 
                    #  Get frequency for each words where word is the key and the count is the value  
                for word in (words):     
                    word = word.lower()     
                    data[word] = data.get(word, 0) + 1 
                # Sort the dictionary in reverse order to print first the most used terms
                dict(sorted(data.items(), key=operator.itemgetter(1),reverse=True)) 
                word_cloud = WordCloud(width = 800, height = 800, 
                background_color ='white',max_words = 500) 
                word_cloud.generate_from_frequencies(data) 


            #st.bar_chart(df["Frequency","Words"])

            import plotly.express as px


            fig = px.bar(df, x="Word", y="Frequency", barmode='group', height=400)
            # st.dataframe(df) # if need to display dataframe
            st.plotly_chart(fig)


            text1 = job_description1
            # Create some sample text
            

            # Create and generate a word cloud image:
            wordcloud1 = WordCloud().generate(text1)
            
            # Display the generated image:
            st.set_option('deprecation.showPyplotGlobalUse', False)

            st.subheader("Word frequency in the job description:")
            plt.imshow(wordcloud1, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot()

            st.title("5 most frequent keywords are:")
            st.write(df.sort_values(by=['Frequency']).head(5))

            st.subheader("")
            st.subheader("")
            st.subheader("")

            st.subheader("-->If you want full guidance on Resume building,contact us on contact.jobshie@gmail.com.")

            

            st.subheader("-->We will provide you the full resume building guidance and hold your hand till you got your dream job.")

            st.subheader("-->Contact us now at contact.jobshie@gmail.com")


if __name__ == '__main__':
	main()

