import streamlit as st
import time

st.set_page_config(
    page_title='Home Page - Drilling Data Automation System',
    page_icon='⛑',
    menu_items={
        'Get help': 'https://www.linkedin.com/in/berat-tu%C4%9Fberk-g%C3%BCnel-928460173/',
        'About': "# Make sure to *cite* while using!"
    }
)

with st.sidebar:
    st.spinner("Loading...")
    time.sleep(2)
    st.info("Please start with *Data Preprocessing*")

st.title("Drilling Data Automation System")
st.info(
    "This webb app is created by Berat Tuğberk Günel. And, the datasets that are used during the development process are acquired from TU Bergakademie Freiberg.")

st.info(
    "This project is developed under the supervision of Prof. Dr. Matthias Reich and Dr. Silke Röntzsch"
)

st.markdown(
    """
    This project is created without expectation of economical purposes
    **👈 Please select an option from the sidebar** 

    ### What does this project include?
    - Data Preprocessing
    - Statistical Automation for Identification of Drilling Data
    - Visualizations
    - Correlations
    - Machine Learning Model to Predict -here come to objective-
    ### To contact me or professors:
    ##### Berat Tuğberk Günel
    - Use my e-mail --> gunel18@itu.edu.tr
    - Use my LinkedIn --> https://www.linkedin.com/in/berat-tu%C4%9Fberk-g%C3%BCnel-928460173/
    ##### Prof Dr. Matthias Reich
    - Use e-mail --> Matthias.Reich@tbt.tu-freiberg.de
    ##### Dr. Silke Röntzsch
    - Use e-mail --> Silke.Roentzsch@tbt.tu-freiberg.de
"""
)




