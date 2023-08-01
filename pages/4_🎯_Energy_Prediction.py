import streamlit as st

st.set_page_config(
    page_title="Energy Prediction",
    page_icon="🎯"
    )


@st.cache_resource(ttl=3600)
def precitionData(length, target, depth_intervals, formations):
    import pandas as pd
    import numpy as np

    interval_depth = np.arange(0+length, target+1, length)
    prediction_data = pd.DataFrame(data=interval_depth, columns=['Depth'])

    interval_formations = []

    for depth in interval_depth:
        assigned_formation = None

        for i, interval in enumerate(depth_intervals):
            if interval[0] <= depth <= interval[1]:
                assigned_formation = formations[i]
                break

        interval_formations.append(assigned_formation)

    prediction_data.loc[:, 'Formation'] = interval_formations

    return prediction_data


@st.cache_resource(ttl=3600)
def mlModel(model_data):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LinearRegression, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, r2_score

    X = model_data[['Teufe [m] Mean', 'Formation']]
    y = model_data['MSE [bar]']

    onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_encoded = onehot_encoder.fit_transform(X.loc[:, 'Formation'].values.reshape(-1, 1))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.loc[:, 'Teufe [m] Mean'].values.reshape(-1, 1))

    X_final = np.hstack((X_scaled, X_encoded))

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    model_params = [
        {
            'model': LinearRegression(),
            'params': {}
        },
        {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 5, 10, 15]
            }
        },
        {
            'model': XGBRegressor(),
            'params': {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.01, 0.001]
            }
        },
        {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        },
        {
            'model': GradientBoostingRegressor(),
            'params': {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.01, 0.001]
            }
        },
        {
        'model': ElasticNet(),
        'params': {
            'alpha': [0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.5, 0.9],
            'fit_intercept': [True, False],
            'max_iter': [1000, 2000]
            }
        }
    ]

    scores = []
    best_model = None
    best_score = -np.inf
    for mp in model_params:
        model = mp['model']
        params = mp['params']
        clf = GridSearchCV(model, params, cv=5)
        clf.fit(X_train, y_train)
        scores.append({
            'model': model.__class__.__name__,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_
        })

        if clf.best_score_ > best_score:
            best_score = clf.best_score_
            best_model = model

    if best_model is not None:
        best_params = next((s['best_params'] for s in scores if s['model'] == best_model.__class__.__name__), None)
        if best_params:
            best_model.set_params(**best_params)
            best_model.fit(X_train, y_train)

    if best_model is not None:
        model = best_model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r_squared = r2_score(y_test, y_pred)

    return best_model, best_model.__class__.__name__, mse, rmse, r_squared, best_params, onehot_encoder, scaler


@st.cache_resource(ttl=3600)
def prediction(_best_model, dataset, _onehot_encoder, _scaler):
    import numpy as np
    import pandas as pd

    input_depth = dataset.loc[:, 'Depth'].values.reshape(-1, 1)
    input_formation = dataset.loc[:, 'Formation'].values.reshape(-1, 1)

    input_depth_scaled = _scaler.transform(input_depth)
    input_formation_encoded = _onehot_encoder.transform(input_formation)
    input_data_final = np.hstack((input_depth_scaled, input_formation_encoded))

    predicted_mse = _best_model.predict(input_data_final)

    data = []
    for depth, formation, mse in zip(input_depth, input_formation, predicted_mse):
        for d, f in zip(depth, formation):
            data.append({
                "Depth [m]": d,
                "Probable Formation": f,
                "Predicted MSE [bar]": round(mse, 2)
            })

    df = pd.DataFrame(data)

    return df

@st.cache_resource(ttl=3600)
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
    st.session_state.confirm_upload_ml = True

def Predict():
    st.session_state.predict = True

def main():
    import pandas as pd
    import numpy as np
    import time

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
        text-align: justify;
        font-size: 20px;
        line-height: 1.6;
        color: #ffffff;
    }
    </style>
    """
    , unsafe_allow_html=True
    )

    add_logo()

    st.markdown("""
        <div class='stTitle'>Energy Prediction</div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='stMarkdown'>In this page, you can develop a model which predicts the corresponding mechanical specific energy of the target depth' intervals. To be able to achieve this, you must have been completed the <i>Interval Automation</i> part. If your resulted <i>Interval Datasets</i> are ready, please upload them at the same time by using below section.</div>

    """, unsafe_allow_html=True)

    st.markdown('                            ')
    st.markdown('                            ')

    st.markdown("""
        <div class='stMarkdown'>Additionally, make sure that your datasets are convinient with the below format. If not, the software will be raising an error. If the data uploading is found to be successful, you will need to specifiy <b><i>Target Depth</i></b>, <b><i>Drill Pipe Lenght</i></b>, <b><i>Probable Formations</i></b>.</div>

    """, unsafe_allow_html=True)
    st.markdown('                            ')

    st.info('**Please make sure that all of the features are in the SI unit system.**', icon='💹')
    st.markdown('                            ')

    data = [['Observation', 'Observation', 'Observation']]

    columns = ['Teufe [m] Mean', 'Formation', 'MSE [bar]']

    sample = pd.DataFrame(data, columns=columns)
    st.table(data=sample)

    st.info("**If your dataset is in the format of .xlsx, you must enter a valid sheet name. Thus, the sheet name must be the same with other Excel files. Otherwise, you do not need to specify a sheet name.**")
    uploaded_data_ml = st.file_uploader("**Please upload your data file**", type=['csv', 'xlsx'], accept_multiple_files=True)

    sheet = st.text_input("**Enter the sheet name**", placeholder='Sheet1')

    st.button('Upload', on_click=ConfirmedUpload, type='primary')

    if 'confirm_upload_ml' not in st.session_state:
        st.session_state.confirm_upload_ml = False

    conact_data = None

    if st.session_state.confirm_upload_ml:
        try:
            file_names = [dataset.name for dataset in uploaded_data_ml]
            extension = [file_name.endswith('.csv') for file_name in file_names]
        
            if np.all(extension):
                        conact_data = pd.concat([pd.read_csv(dataset, encoding='utf-8') for dataset in uploaded_data_ml])
                        st.success('**Dataset has been uploaded!**', icon="✅")
                        

            elif not np.all(extension):
                if sheet.strip():
                    conact_data = pd.concat([pd.read_excel(dataset, sheet_name=sheet) for dataset in uploaded_data_ml])
                    st.success('**Dataset has been uploaded!**', icon="✅")
        except:
             st.warning('Please upload convinient dataset(s).')
    
    if conact_data is not None:
        st.markdown('                            ')
        st.markdown('                            ')
        st.markdown("""<div class='stHeader'>Entire Dataset</div>""",
                        unsafe_allow_html=True)
        st.table(data=conact_data.sample(15))
        st.info(f"**The resulted dataset consists of *{len([dataset.name for dataset in uploaded_data_ml])}* distinct datasets.**")

        st.markdown(f"""
        <div class='stHeader'>Parameters</div>
        """, 
        unsafe_allow_html=True)
        st.markdown('                            ')

        pipe_length = st.number_input('**Drill Pipe Length, in meters**', min_value=1.0, step=0.01)
        target_depth = st.number_input('**Target Depth, in meters**', min_value=1.0, step=0.01)

        depth_intervals = []
        formations = []

        st.markdown(f"""
        <div class='stHeader'>Probable Formations</div>
        """, 
        unsafe_allow_html=True)
        st.markdown('                            ')

        num_intervals = st.number_input('**Number of Formations Interval**', min_value=1, value=3, step=1)

        for i in range(num_intervals):
            interval_start = st.number_input(f'**Interval {i + 1} Start, in m**', value=0.0, step=0.1)
            interval_end = st.number_input(f'**Interval {i + 1} End, in m**', value=0.0, step=0.1)
            formation = st.text_input(f'**Formation for Interval {i + 1}**', value='', placeholder='Gneiss')

            depth_intervals.append((interval_start, interval_end))
            formations.append(formation)
        
        if 'predict' not in st.session_state:
            st.session_state.predict = None

        st.button('Predict', on_click=Predict, type='primary')

        if st.session_state.predict:
            if target_depth == depth_intervals[-1][1]:
                if pipe_length > 0:
                    if target_depth > 1:
                        prediction_data = precitionData(pipe_length, target_depth, depth_intervals, formations)
                        st.markdown('                            ')

                        results = mlModel(conact_data)

                        progress_text = "**Operation in progress. Please wait.**"
                        my_bar = st.progress(0, text=progress_text)

                        for percent_complete in range(100):
                            time.sleep(0.1)
                            my_bar.progress(percent_complete + 1, text=progress_text)

                            if percent_complete + 1 == 100:
                                my_bar.progress(percent_complete + 1, text='**Prediction has been successful!**')
                                st.success('**Prediction has been performed!**', icon="✅")
                                continue

                            elif percent_complete > 100 or percent_complete < 0:
                                st.error(st.error('**Something went wrong, try again!**', icon="🚨"))
                                break
                        
                        
                        st.markdown(f"""
                            <div class='stHeader'>Machine Learning Model Results</div>
                            """, 
                            unsafe_allow_html=True)
                        
                        best_param = results[5]

                        st.markdown('                            ')
                        st.markdown('                            ')

                        st.markdown("""
                                    <div class='stHeader'><center>Model Metrics</center>
                                    """, unsafe_allow_html=True)
                        
                        if not best_param:
                            best_param = 'None'

                            st.markdown(f"""
                                <center>
                                    <table class='justify-content-center'>
                                        <tr>
                                            <th><center>Best Model</center></th>
                                            <th><center>MSE</center></th>
                                            <th><center>RMSE</center></th>
                                            <th><center>R-Squared</center></th>
                                            <th><center>Best Parameter</center></th>
                                        </tr>
                                        <tr>
                                            <td><center>{results[1]}</center></td>
                                            <td><center>{round(results[2], 2)}</center></td>
                                            <td><center>{round(results[3], 2)}</center></td>
                                            <td><center>{round(results[4], 2)}</center></td>
                                            <td><center>{best_param}</center></td>
                                        </tr>
                                    </table>
                                </center>
                                """, unsafe_allow_html=True)

                        else:
                            best_param = best_param
                            
                            st.markdown(f"""
                                <center>
                                    <table class='justify-content-center'>
                                        <tr>
                                            <th><center>Best Model</center></th>
                                            <th><center>MSE</center></th>
                                            <th><center>RMSE</center></th>
                                            <th><center>R-Squared</center></th>
                                        </tr>
                                        <tr>
                                            <td><center>{results[1]}</center></td>
                                            <td><center>{round(results[2], 2)}</center></td>
                                            <td><center>{round(results[3], 2)}</center></td>
                                            <td><center>{round(results[4], 2)}</center></td>
                                        </tr>
                                    </table>
                                </center>
                                """, unsafe_allow_html=True)
                            
                            st.markdown('                            ')

                            st.markdown(f"""
                                <center>
                                    <table class='justify-content-center'>
                                        <tr>
                                            <th><center>Hyperparameter</center></th>
                                            <th><center>Value</center></th>
                                        </tr>
                                        {"".join(f"<tr><td><center>{key}</center></td><td><center>{value}</center></td></tr>" for key, value in best_param.items())}
                                    </table>
                                </center>
                                """, unsafe_allow_html=True)
                            
                        st.markdown('                            ')
                        st.markdown('                            ')
                        st.markdown('                            ')
                        st.markdown('                            ')
                        st.markdown('                            ')

                        st.markdown("""
                            <div class='stHeader'><center>Model Prediction</center></div>
                        
                        """, unsafe_allow_html=True)
                        
                        predicted_data = prediction(results[0], prediction_data, results[6], results[7])
                        st.table(data=predicted_data)
                    
                    else:
                        st.warning("**Please make sure that your target depth is greater than zero.**", icon="✅")
                    
                else:
                    st.warning("**Please make sure that your pipe length is greater than zero.**", icon="✅")
            
            else:
                st.warning("**Please make sure that your last interval depth equals to the target depth.**", icon="✅")
                
                   
if __name__ == '__main__':
    main()
