def main():
    import streamlit as st
    import time

    st.set_page_config(
        page_title='Home Page - Drilling Data Automation System',
        page_icon='⛑',
        menu_items={
            'Get help': 'https://www.linkedin.com/in/berat-tu%C4%9Fberk-g%C3%BCnel-928460173/',
            'About': "# Make sure to *cite* while using!"
        },
        layout="wide"
    )

    st.markdown(
        """
        <style>
        .stTitle {
            font-size: 40px;
            font-weight: bold;
            color: #FF9933;
            margin-bottom: 20px;
        }

        .stHeader {
            font-size: 30px;
            font-weight: bold;
            color: #FF9933;
            margin-bottom: 5px;
        
        }

        .stInfo {
            font-size: 18px;
            line-height: 1.6;
            margin-bottom: 20px;
        }

        .stMarkdown {
            font-size: 16px;
            line-height: 1.6;
            color: #ffffff;
        }
        </style>
        """
        , unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="stTitle">Drilling Data Automation System</div>
        """
        , unsafe_allow_html=True
    )
    
    with st.sidebar:
        st.spinner("Loading...")
        time.sleep(2)
        st.info("Please start with *Data Preprocessing*")

    st.info(
        "This web app is created by Berat Tuğberk Günel. The datasets used during the development process are acquired from TU Bergakademie Freiberg."
    )

    st.info(
        "This project is developed under the supervision of Prof. Dr. Matthias Reich and Dr. Silke Röntzsch."
    )


    st.warning("This project is created without the expectation of economic purposes.", icon="❗")
    
    st.markdown(
        """
        <div class="stMarkdown">
            <br><br>
            <strong>👈 Please select an option from the sidebar</strong>
            <br><br>
            <div class="stHeader">What does this project include?</div>
            <ul>
                <li>Data Preprocessing</li>
                <li>Statistical Automation for Identification of Drilling Data</li>
                <li>Visualizations</li>
                <li>Machine Learning Model to Predict -Still developing-</li>
            </ul>
            <div class="stHeader">To contact me or the professors:</div>
            <h4>Berat Tuğberk Günel</h4>
            <ul>
                <li>Use my e-mail: gunel18@itu.edu.tr</li>
                <li>Use my LinkedIn: <a href="https://www.linkedin.com/in/berat-tu%C4%9Fberk-g%C3%BCnel-928460173/">LinkedIn Profile</a></li>
            </ul>
            <h4>Prof Dr. Matthias Reich</h4>
            <ul>
                <li>Use e-mail: Matthias.Reich@tbt.tu-freiberg.de</li>
            </ul>
            <h4>Dr. Silke Röntzsch</h4>
            <ul>
                <li>Use e-mail: Silke.Roentzsch@tbt.tu-freiberg.de</li>
            </ul>
        </div>
        """
        , unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
