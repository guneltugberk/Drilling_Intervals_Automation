import streamlit as st

st.set_page_config(
    page_title="Energy Prediction",
    page_icon="🎯"
    )


@st.cache_data(ttl=3600)
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

    return best_model, best_model.__class__.__name__, mse, rmse, r_squared, best_params, onehot_encoder, scaler, X_test, y_test


@st.cache_resource(ttl=3600)
def prediction(_best_model, dataset, _onehot_encoder, _scaler, num_bootstraps=100):
    import numpy as np
    import pandas as pd

    input_depth = dataset.loc[:, 'Depth'].values.reshape(-1, 1)
    input_formation = dataset.loc[:, 'Formation'].values.reshape(-1, 1)

    input_depth_scaled = _scaler.transform(input_depth)
    input_formation_encoded = _onehot_encoder.transform(input_formation)
    input_data_final = np.hstack((input_depth_scaled, input_formation_encoded))

    predicted_mse = _best_model.predict(input_data_final)

    predicted_mse_bootstraps = []

    for _ in range(num_bootstraps):
        # Resample input data
        resampled_indices = np.random.choice(len(input_data_final), size=len(input_data_final), replace=True)
        resampled_input_data = input_data_final[resampled_indices]

        # Predict MSE for resampled data
        bootstraps_mse = _best_model.predict(resampled_input_data)
        predicted_mse_bootstraps.append(bootstraps_mse)
    
    mse_mean = np.mean(predicted_mse_bootstraps, axis=0)
    mse_std = np.std(predicted_mse_bootstraps, axis=0)
    lower_bound = mse_mean - 1.96 * mse_std
    upper_bound = mse_mean + 1.96 * mse_std

    data = []
    for depth, formation, mean_mse, lb, ub in zip(input_depth, input_formation, predicted_mse, lower_bound, upper_bound):
        for d, f in zip(depth, formation):
            data.append({
                "Depth [m]": d,
                "Probable Formation": f,
                "Predicted Mean MSE [bar]": round(mean_mse, 2),
                "Lower Bound [bar]": round(lb, 2),
                "Upper Bound [bar]": round(ub, 2)
            })


    df = pd.DataFrame(data)

    return df


def confirmation(data_to_control):

    required_columns = ['Teufe [m] Mean', 'Formation', 'MSE [bar]']

    control_columns = data_to_control.columns

    confirmed = None

    if all(col in control_columns for col in required_columns):
        confirmed = True

    else:
        confirmed = False

    return confirmed


@st.cache_resource(ttl=3600)
def uncertainty_plot(predicted):
    import plotly.graph_objects as go
    import seaborn as sns

    unique_formations = predicted['Probable Formation'].unique()

    # Generate a color palette with a number of colors matching the unique formations
    color_palette = sns.color_palette("husl", n_colors=len(unique_formations))

    # Create a dynamic color mapping dictionary
    color_mapping = {formation: f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
                    for formation, (r, g, b) in zip(unique_formations, color_palette)}
    

    lower_bound = go.Scatter(
    x=predicted['Lower Bound [bar]'],
    y=predicted['Depth [m]'],
    mode='lines',
    name='Lower Bound',
    line=dict(color='lightskyblue'),
    showlegend=True,
    opacity=0.3,
    hovertemplate="Depth: %{y} meter<br>Lower Bound MSE: %{x} bars<br><extra></extra>"
    )

    upper_bound = go.Scatter(
    x=predicted['Upper Bound [bar]'],
    y=predicted['Depth [m]'],
    mode='lines',
    name='Upper Bound',
    fill='tonexty',  
    line=dict(color='lightskyblue'),
    fillcolor='rgba(0, 0, 255, 0.1)', 
    showlegend=True,
    opacity=0.3,
    hovertemplate="Depth: %{y} meters<br>Upper Bound MSE: %{x} bars<br><extra></extra>"
    )

    layout = go.Layout(
    title=dict(text='Predicted MSE with Uncertainty Intervals'),
    xaxis=dict(title='Mechanical Specific Energy [bar]'),
    yaxis=dict(title='Depth [m]'),
    showlegend=True,
    legend=dict(
    tracegroupgap=0,  
    itemsizing='trace'  
            )
    )

    fig = go.Figure(data=[lower_bound, upper_bound], layout=layout)

    for formation, color in color_mapping.items():
        formation_data = predicted[predicted['Probable Formation'] == formation]
        hover_template = (
            "Formation: %{text}<br>" 
            "Predicted Mean MSE: %{x} bars<br>" 
            "Depth: %{y} meters<extra></extra>"
        )

        scatter_trace = go.Scatter(
            x=formation_data['Predicted Mean MSE [bar]'],
            y=formation_data['Depth [m]'],
            mode='markers',
            name=formation,
            marker=dict(color=color),
            legendgroup=formation,
            text=[formation] * len(formation_data),  
            hovertemplate=hover_template
        )

        fig.add_trace(scatter_trace)
    
    fig.update_layout(showlegend=True)            
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    fig.update_yaxes(range=[predicted['Depth [m]'].min(), predicted['Depth [m]'].max()+10])
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(height=600, hovermode='closest',
                        title=dict(
                        xref='paper',
                        x=0.5,
                        font=dict(size=16),
                        xanchor='center'
                    ))
    
    return fig


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
    import plotly.graph_objects as go
    import seaborn as sns

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
        <div class='stTitle'><center>Energy Prediction</center></div>
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

    file_names = [dataset.name for dataset in uploaded_data_ml]
    extension = [file_name.endswith('.csv') for file_name in file_names]

    if st.session_state.confirm_upload_ml:
        control = []

        try:
            for data in uploaded_data_ml:
                if np.all(extension):
                    data_Frame = pd.read_csv(data, encoding='utf-8')

                    control.append(confirmation(data_Frame))
                elif not np.all(extension):
                    data_Frame = pd.read_excel(data, sheet_name=sheet)

                    control.append(confirmation(data_Frame))

            if np.all(extension):
                if np.all(control):
                    conact_data = pd.concat([pd.read_csv(dataset, encoding='utf-8') for dataset in uploaded_data_ml])
                    st.success('**Dataset has been uploaded!**', icon="✅")

                elif not np.all(control):
                    st.warning("**Please make sure that your datasets contain the following features:** *Teufe [m] Mean*, *Formation*, *MSE [m] Mean*", icon='⚠️')

            elif not np.all(extension):
                if sheet.strip():
                    if np.all(control):
                        conact_data = pd.concat([pd.read_excel(dataset, sheet_name=sheet) for dataset in uploaded_data_ml])
                        st.success('**Dataset has been uploaded!**', icon="✅")
                    
                    elif not np.all(control):
                        st.warning("**Please make sure that your datasets contain the following features:** *Teufe [m] Mean*, *Formation*, *MSE [m] Mean*", icon='⚠️')
        except:
             st.warning('Please upload convinient dataset(s).', icon='⚠️')
    
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
        flag = confirmation(conact_data)
        number_flag = len(conact_data['Well Name'].unique())
        length_flag = len(conact_data)

        if st.session_state.predict:
            if target_depth == depth_intervals[-1][1]:
                if pipe_length > 0:
                    if target_depth > 1:
                        if number_flag > 3:
                            if length_flag > 200:
                                if flag:
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

                                    st.markdown("                   ")

                                    st.markdown("""
                                                <div class='stHeader'>Uncertainty Analysis</div>            
                                                
                                    """
                                    , unsafe_allow_html=True
                                    )
                                    
                                    fig = uncertainty_plot(predicted=predicted_data)

                                    st.plotly_chart(fig)
                        
                                else:
                                    st.error('**Please make sure that your dataset contain the following features:** *Teufe [m] Mean*, *Formation*, *MSE [bar]*', icon='❗')
                            
                            else:
                                st.warning('**Please make sure that your datasets contain more than 200 observations.**', icon="⚠️")

                        else:
                            st.warning('**Please make sure that you have been uploaded more than 3 different well data.**', icon="⚠️")

                                        
                    else:
                        st.warning("**Please make sure that your target depth is greater than zero.**", icon="⚠️")
                    
                else:
                    st.warning("**Please make sure that your pipe length is greater than zero.**", icon="⚠️")
            
            else:
                st.warning("**Please make sure that your last interval depth equals to the target depth.**", icon="⚠️")
                
                   
if __name__ == '__main__':
    main()