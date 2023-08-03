import streamlit as st

st.set_page_config(
        page_title='Data Uploading and Preprocessing',
        page_icon='💹'
    )


class Upload:
    def __init__(self, data_source, sheet_name):
        self.df = None
        self.sheet_name = sheet_name
        self.data_source = data_source

    @st.cache_data(ttl=3600)
    def read_file(self):
        import pandas as pd
        import streamlit as st

        if self.data_source.name.endswith(".xlsx"):
            # Read Excel file
            try:
                self.df = pd.read_excel(self.data_source, sheet_name=self.sheet_name)
            except:
                st.warning('**Please enter a correct sheet name!**', icon="✅")

        elif self.data_source.name.endswith(".csv"):
            # Read CSV file
            self.df = pd.read_csv(self.data_source)
        else:
            # Unsupported file format
            self.df = None

            raise ValueError("Unsupported file format. Please upload an Excel or CSV file.")

        return self.df


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/TUBAF_Logo.svg/300px-TUBAF_Logo.svg.png);
                background-repeat: no-repeat;
                padding-top: 95px;
                background-position: 25px 50px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Institut für Bohrtechnik und Fluidbergbau";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 15px;
                position: relative;
                top: 100px;
                text-align: center;
                font-weight: bold;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ConfirmedUpload():
    st.session_state.confirm_upload = True


def ConfirmedProcess():
    st.session_state.confirm_process = True


@st.cache_data(ttl=3600)
def processing(well_data):
    import pandas as pd
    import numpy as np

    zero_nan_col = []

    # Find the columns containing only zero or NaN values
    if isinstance(well_data, pd.DataFrame):
        df = well_data.copy()

        measurement_units = df.iloc[0].tolist()
        df.columns = [f"{col} {unit}" for col, unit in zip(df.columns, measurement_units)]

        df = df.iloc[1:].reset_index(drop=True)

        for col in df.columns:
            if np.all(pd.isna(df[col])) or np.all(df[col] == 0):
                zero_nan_col.append(col)

        df = df.drop(zero_nan_col, axis=1)

        df = df.apply(pd.to_numeric, errors='coerce')

        return df, len(df['Zeit [s]']), len(df.columns)

    return None


@st.cache_data(ttl=3600)
def dropNaN(dropped):
    import pandas as pd

    global dropped_copy

    if isinstance(dropped, pd.DataFrame):
        dropped_copy = dropped.dropna()

        if dropped_copy.empty:
            return None

    return dropped_copy


def main():
    import time
    import pandas as pd

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
    
        .stMarkdown {
            font-size: 16px;
            line-height: 1.6;
            color: #ffffff;
        }

        </style>
        """
        , unsafe_allow_html=True
    )

    add_logo()

    st.markdown("""
        <div class='stTitle'><center>Data Preprocessing</center></div>
    """, unsafe_allow_html=True)

    st.info("**Uploaded dataset's feature must be in the SI unit system. Furthermore, the structure of the dataset must be as follows:**")

    data = [['[s]', '[s]', '[m]', '[m]', '[m/h]', '[rad]', '[U/min]',
             '[bar]', '[bar]', '[bar]', '[Nm3/min]', '[Nm]', '[t]'], ['Observation', 'Observation', 'Observation', 'Observation', 'Observation', 'Observation', 'Observation',
             'Observation', 'Observation', 'Observation', 'Observation', 'Observation', 'Observation']]

    columns = ['Zeit', 'Delta Zeit', 'Teufe', 'Delta Teufe', 'vB', 'RotWinkel', 'DZ', 'Andruck', 'Drehdruck', 'p Luft',
               'Q Luft', 'DM', 'WOB']

    sample = pd.DataFrame(data, columns=columns)
    st.table(data=sample)

    st.info("**If your dataset is in the format of .xlsx, you must enter a valid sheet name. Otherwise, you do not need to specify a sheet name.**")
    uploaded_data = st.file_uploader("**Please upload your data file**", type=['csv', 'xlsx'],
                                     accept_multiple_files=False)
    sheet = st.text_input("**Enter the sheet name**", placeholder='Sheet1')
    st.info('**After uploading the dataset and entering the name of the sheet, please hit *Upload* button**', icon='🎯')

    st.button('Upload', on_click=ConfirmedUpload, type='primary')

    if 'confirm_upload' not in st.session_state:
        st.session_state.confirm_upload = False

    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None

    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None


    if st.session_state.confirm_upload:
        if uploaded_data is not None:
                file_name = uploaded_data.name

                if file_name.endswith(".csv"):
                        st.session_state.uploaded_data = uploaded_data
                        st.success('**Dataset has been uploaded!**', icon="✅")

                elif file_name.endswith(".xlsx"):
                    if sheet.strip():
                        st.session_state.uploaded_data = uploaded_data
                        st.success('**Dataset has been uploaded!**', icon="✅")

                else:
                    st.error('**Please enter a sheet name!**')
        else:
            st.error('**Please upload a dataset!**')

    new_form = 0

    if st.session_state.uploaded_data:
        with st.form('Processing'):
            st.markdown("""
                <div class='stHeader'>Processing Data</div>
            """, unsafe_allow_html=True)

            if 'confirm_process' not in st.session_state:
                st.session_state.confirm_process = False

            st.form_submit_button('Process', on_click=ConfirmedProcess, type='primary')

            if st.session_state.confirm_process:
                progress_text = "**Operation in progress. Please wait.**"
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1, text=progress_text)

                    if percent_complete + 1 == 100:
                        my_bar.progress(percent_complete + 1, text='**Dataset has been uploaded!**')
                        st.success('**Dataset has been processed!**', icon="✅")
                    elif percent_complete > 100 or percent_complete < 0:
                        st.error(st.error('**Something went wrong, try again!**', icon="🚨"))

                refresh = None

                if uploaded_data is not None:
                    file_name = uploaded_data.name

                    if file_name.endswith(".xlsx"):
                        file_name_without_extension = file_name[:-5]
                        st.markdown(
                            f"""<div class='stHeader'><center>Name of the Well: <i>{file_name_without_extension}</i></center></div>""",
                            unsafe_allow_html=True)

                        refresh = True

                    elif file_name.endswith(".csv"):
                        file_name_without_extension = file_name[:-4]
                        st.markdown(
                            f"""<div class='stHeader'><center>Name of the Well: <i>{file_name_without_extension}</i></center></div>""",
                            unsafe_allow_html=True)

                        refresh = True

                    else:
                        st.error('**Something went wrong, try again!**', icon="🚨")

                else:
                    refresh = False

                if refresh:
                    try:
                        data_frame = Upload(data_source=uploaded_data, sheet_name=sheet).read_file()
                        processed_data = processing(data_frame)[0]

                        st.session_state.processed_data = processed_data

                        st.table(data=st.session_state.processed_data.describe())

                        num_features = processing(data_frame)[2]
                        num_obs = processing(data_frame)[1]

                        st.caption(f'**Number of Features:** *{num_features}*')
                        st.caption(f'**Number of Observations:** *{num_obs}*')
                        st.divider()

                        st.markdown("""
                        <div class='stHeader'><center>Number of Missing Values</center></div>
                        """, unsafe_allow_html=True)
                        missing_values = st.session_state.processed_data.isna().sum()

                        st.table(data=missing_values)
                        new_form = 1

                    except:
                        st.session_state.processed_data = pd.DataFrame([])
                        st.warning('**Please supply all necessary informations.**', icon="✅")

                if not refresh:
                    st.warning('**Please refresh the page and re-upload the dataset.**', icon='💹')

        if new_form == 1:
            if not st.session_state.processed_data.empty:
                if missing_values.sum() > 0:
                    with st.form('MissingData'):
                        st.markdown("""
                        <div class='stHeader'><center>Handling with Missing Data</center></div>
                        """, unsafe_allow_html=True)
                        option = st.selectbox(
                            '**How would you like to manipulate the data?**',
                            ('Drop NaN', 'Impute NaN'))

                        st.write(f'**You have selected:** *{option}*')

                        confirmation = st.form_submit_button('Confirm Choice', type='primary')

                        if 'dropped_data' not in st.session_state:
                            st.session_state.dropped_data = None

                        if confirmation:
                            if option == 'Drop NaN':
                                dropped_data = dropNaN(st.session_state.processed_data)
                                st.session_state.dropped_data = dropped_data

                                if st.session_state.dropped_data is not None and isinstance(st.session_state.dropped_data, pd.DataFrame):
                                    st.success('**All missing values are dropped!**', icon="✅")

                                    st.markdown("""
                                    <div class='stHeader'>Number of Missing values</div>
                                    """, unsafe_allow_html=True)

                                    st.table(data=st.session_state.dropped_data.isna().sum())

                                elif st.session_state.dropped_data is None:
                                    st.warning('**Please re-upload the dataset.**', icon='💹')

                                else:
                                    st.error('**Something went wrong, try again!**', icon="🚨")

                                st.divider()

                            elif option == 'Impute NaN':
                                st.info('Still developing')

                            else:
                                st.error('**Something went wrong, try again!**', icon="🚨")
                else:
                    dropped_data = st.session_state.processed_data

                    st.session_state.dropped_data = dropped_data
                        
                    st.info('**There are no missing values to handle.**', icon='💹')

            else:
                st.warning('**Please upload a proper dataset!**', icon='💹')


    elif not st.session_state.uploaded_data:
        st.warning('**Please complete data uploading part carefully!**', icon='💹')


if __name__ == '__main__':
    main()
