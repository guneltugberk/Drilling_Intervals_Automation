import streamlit as st

st.set_page_config(
    page_title="Drilling Intervals Automation",
    page_icon="📐"
    )


class Intervals:
    @staticmethod
    @st.cache_resource(ttl=3600)
    def train_test_split(data, test_size=0.2, random_seed=None):
        import numpy as np

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Shuffle the data
        np.random.shuffle(data)

        # Number of elements to include in the test set
        num_test = int(len(data) * test_size)

        # Use the first 'num_test' elements for the test set
        test = data[:num_test]

        # Use the rest for the training set
        train = data[num_test:]

        return train, test
    
    @staticmethod
    @st.cache_resource(ttl=3600)
    def custom_pr_auc_score(_classifier, X_train):
        from sklearn.metrics import precision_recall_curve, auc
        from sklearn.ensemble import IsolationForest
        import numpy as np

        # Fit the classifier to the training data
        _classifier.fit(X_train)

        # Get the outlier scores for X_train using the decision_function
        if hasattr(_classifier, 'kneighbors_graph'):
            outlier_scores = -_classifier.negative_outlier_factor_
        else:
            outlier_scores = -_classifier.score_samples(X_train)

        # Calculate precision-recall curve and AUC
        precision, recall, _ = precision_recall_curve(np.ones(len(X_train)), outlier_scores)
        pr_auc = auc(recall, precision)

        return pr_auc

    @staticmethod
    @st.cache_resource(ttl=3600)
    def feature_wise_isolation_forest(scaled, real, n, c):
        from sklearn.ensemble import IsolationForest

        best_iforest = IsolationForest(n_estimators=n,
                                    contamination=c)

        # Fit the best Isolation Forest model to the entire training set
        best_iforest.fit(scaled)

        # Predict outliers on the test set using both models
        outlier_scores_if = best_iforest.decision_function(scaled)

        # Extract the outliers and the inliers (non-outliers) from the test set for both models
        outliers_if = real[outlier_scores_if < 0]
        inliers_if = real[outlier_scores_if >= 0]


        return inliers_if

    @staticmethod
    @st.cache_resource(ttl=3600)
    def outliers(data, cols, threshold):
        import pandas as pd
        import numpy as np
        from scipy import stats
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler


        data_frame = data.copy()

        delta_teufe = np.array(data.loc[:, 'Delta Teufe [m]'])
        teufe = np.array(data.loc[:, 'Teufe [m]'])

        unsatisfied_indexes_teufe = np.where((teufe < 0) | (delta_teufe <= 0.001))[0]
        removed_data = data.drop(data.index[unsatisfied_indexes_teufe])

        p_luft = np.array(removed_data.loc[:, 'p Luft [bar]'])
        DZ = np.array(removed_data.loc[:, 'DZ [U/min]'])
        
        unsatisfied_indexes_pluft = np.where((p_luft < 2) | (DZ == 0))[0]
        removed_data = removed_data.drop(removed_data.index[unsatisfied_indexes_pluft])

        if isinstance(removed_data, pd.DataFrame):
            # Create a list of the filtered values for each column
            filtered_values_list = []

            for col in cols:
                if col == 'p Luft [bar]' or col == 'DZ [U/min]' or col == 'Q Luft [Nm3/min]':
                    desired_values = removed_data[col]

                    z_score_desired_values = stats.zscore(desired_values)
                    filtered_desired_values = desired_values[np.abs(z_score_desired_values) < threshold]

                    if len(filtered_desired_values) > 0:
                        filtered_values_list.append(filtered_desired_values)

            min_length_z = min(len(l) for l in filtered_values_list)
            sliced_arrays_z = [arr[:min_length_z] for arr in filtered_values_list]

            columns_z = ['p Luft [bar]', 'DZ [U/min]', 'Q Luft [Nm3/min]']

            data_z = {col: arr for col, arr in zip(columns_z, sliced_arrays_z)}
            df_z = pd.DataFrame(data_z)

            data_frame = data_frame.iloc[:len(df_z)]
            data_frame.loc[:, columns_z] = data_frame.loc[:, columns_z].replace(data_z)
            data_frame.reset_index(drop=True, inplace=True)

            print(data_frame)

        scaler = StandardScaler()

        p_luft_if = np.array(data_frame.loc[:, 'p Luft [bar]']).reshape(-1, 1)
        dz_if = np.array(data_frame.loc[:, 'DZ [U/min]']).reshape(-1, 1)

        scaled_p_luft = scaler.fit_transform(p_luft_if)
        scaled_dz = scaler.fit_transform(dz_if)

        X_train_p_luft, X_test_p_luft = Intervals.train_test_split(scaled_p_luft, test_size=0.2, random_seed=42)
        X_train_dz, X_test_dz = Intervals.train_test_split(scaled_dz, test_size=0.2, random_seed=42)

        n_neighbors_list = [50, 75, 100, 125, 150, 175, 200]
        contamination_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        
        best_score_p_luft = 0
        best_params_p_luft = {}
        
        best_score_dz = 0
        best_params_dz = {}

        for n in n_neighbors_list:
            for c in contamination_list:
                iforest = IsolationForest(n_estimators=n, contamination=c)

                score_p_luft = Intervals.custom_pr_auc_score(iforest, X_train_p_luft)
                score_dz = Intervals.custom_pr_auc_score(iforest, X_train_dz)

                if score_p_luft > best_score_p_luft:
                    best_score_p_luft = score_p_luft
                    best_params_p_luft = {'n_estimators': n, 'contamination': c}

                if score_dz > best_score_dz:
                    best_score_dz = score_dz
                    best_params_dz = {'n_estimators': n, 'contamination': c}
        
        inliers_p_luft = Intervals.feature_wise_isolation_forest(scaled_p_luft, np.array(data_frame.loc[:, 'p Luft [bar]']), best_params_p_luft['n_estimators'], best_params_p_luft['contamination'])
        inliers_dz = Intervals.feature_wise_isolation_forest(scaled_dz, np.array(data_frame.loc[:, 'DZ [U/min]']), best_params_dz['n_estimators'], best_params_dz['contamination'])

        len_list = [inliers_p_luft, inliers_dz]

        columns = ['p Luft [bar]', 'DZ [U/min]']

        min_length = min(len(ml) for ml in len_list)
        sliced_arrays = [arr[:min_length] for arr in len_list]

        data = {col: arr for col, arr in zip(columns, sliced_arrays)}
        df = pd.DataFrame(data)

        existing_df = data_frame.iloc[:len(df)]
        existing_df.loc[:, columns] = data_frame.loc[:, columns].replace(data)
        existing_df.reset_index(drop=True, inplace=True)
        print(existing_df)
        existing_df = existing_df[existing_df['p Luft [bar]'] != 0]

        return existing_df

    @staticmethod
    def Energy(data):
        energy = []

        for p, q in zip(data.loc[:, 'p Luft [bar]'], data.loc[:, 'Q Luft [Nm3/min]']):
            energy_value = (p * 14.503 * q * 264.172052) / 1714
            energy_value = energy_value * 0.745699872
            energy.append(energy_value)

        data.loc[:, 'Hydraulic Power [kW]'] = energy

        return data

    @staticmethod
    @st.cache_resource(ttl=3600)
    def CalculateIntervals(data, cols, interval_size, error):
        import pandas as pd
        import numpy as np

        global interval_count

        # Get the column names dynamically
        statistical_columns = ['{} Mean'.format(col) for col in cols]
        statistical_columns += ['{} Standard Dev.'.format(col) for col in cols]

        statistical_df = pd.DataFrame(columns=statistical_columns)

        for col in cols:
            if col == 'Zeit [s]' or col == 'Teufe [m]':
                if np.isnan(data[col]).any() or len(data[col]) == 0:
                    st.warning('Dataset contains missing or empty values in the critical features.')

        if 'p Luft [bar]' in cols and 'Q Luft [Nm3/min]' in cols:
            cols.append('Hydraulic Power [kW]')

        if 'DZ [U/min]' in cols and 'Andruck [bar]' in cols and 'vB [m/h]' in cols:
            cols.append('Drillibility Index [kN/mm]')

        depth = np.array(data['Teufe [m]'].values)
        deltaTime = np.array(data['Delta Zeit [s]'].values)

        interval_count = round((max(depth) - min(depth)) / interval_size)
        start_index = 0
        depth_interval = []
        time_interval = []

        for i in range(interval_count):
            end_value = depth[start_index] + interval_size
            end_index = depth.searchsorted(end_value, side='right')

            if end_index >= len(depth):
                end_index = len(depth) - 1

            diff = depth[end_index] - depth[start_index]

            if (diff <= interval_size or diff <= ((interval_size * error / 100) + interval_size) * 1.96) and deltaTime.size:
                interval_values_data_frame = pd.DataFrame(columns=cols)
                interval_values_depth = np.array(data['Teufe [m]'].values)[start_index:end_index + 1]
                interval_values_time = np.array(data['Zeit [s]'].values)[start_index:end_index + 1]

                if i == 0:
                    for col in cols:
                        interval_values_data_frame[col] = data[col].values[start_index:end_index + 1]
                        interval_values_data_frame[col] = interval_values_data_frame[col].astype(float)

                elif 0 < i <= interval_count:
                    for col in cols:
                        column_to_operate = col + '_' + f'{i}'
                        interval_values_data_frame[column_to_operate] = data[col].values[start_index:end_index + 1]
                        interval_values_data_frame[column_to_operate] = interval_values_data_frame[
                            column_to_operate].astype(float)

                else:
                    st.error('Huge Error on indexing')

                if interval_values_data_frame.empty:
                    continue

                else:
                    if i == 0:
                        if len(interval_values_data_frame['Zeit [s]'].values) >= 3 and len(
                                interval_values_data_frame['Teufe [m]'].values) >= 3:
                            if np.all(np.gradient(interval_values_data_frame['Zeit [s]'].values) != 0) and np.all(
                                    np.gradient(interval_values_data_frame['Teufe [m]'].values) != 0):

                                # Calculate the differences between consecutive values
                                time_diff = np.diff(interval_values_data_frame['Zeit [s]'].values)
                                depth_diff = np.diff(interval_values_data_frame['Teufe [m]'].values)

                                # Calculate the mean and standard deviation of the differences
                                time_mean = time_diff.mean()
                                time_std = time_diff.std()

                                depth_mean = depth_diff.mean()
                                depth_std = depth_diff.std()

                                # Define the threshold as a certain number of standard deviations from the mean
                                time_threshold = time_mean + 3 * time_std
                                depth_threshold = depth_mean + 3 * depth_std

                                # Find the indices where the differences exceed the thresholds
                                high_diff_indices = np.where(time_diff > time_threshold)[
                                    0]  # Indices where time difference exceeds the threshold
                                high_diff_indices = np.append(high_diff_indices,
                                                              np.where(depth_diff > depth_threshold)[
                                                                  0])  # Indices where depth difference exceeds the threshold

                                interval_values_depth = np.delete(interval_values_depth, high_diff_indices + 1)
                                interval_values_time = np.delete(interval_values_time, high_diff_indices + 1)

                                depth_interval.append(interval_values_depth)
                                time_interval.append(interval_values_time)

                                for col in cols:
                                    feature = interval_values_data_frame[col]
                                    feature = np.array(feature)
                                    feature = feature[~np.isnan(feature)]
                                    feature = np.delete(feature, high_diff_indices + 1)
                                    feature = feature.astype(float)

                                    mean_col_label = col + ' ' + 'Mean'
                                    std_col_label = col + ' ' + 'Standard Dev.'

                                    feature = feature[~np.isnan(feature)]

                                    statistical_df.loc[i, mean_col_label] = np.mean(feature)
                                    statistical_df.loc[i, std_col_label] = np.std(feature)

                    elif 0 < i <= interval_count:
                        if len(interval_values_data_frame[f'Teufe [m]_{i}'].values) >= 3 and len(
                                interval_values_data_frame[f'Zeit [s]_{i}'].values) >= 3:
                            if np.all(np.gradient(
                                    interval_values_data_frame[f'Zeit [s]_{i}'].values) != 0) and np.all(np.gradient(interval_values_data_frame[f'Teufe [m]_{i}'].values) != 0):

                                # Calculate the differences between consecutive values
                                time_diff = np.diff(interval_values_data_frame[f'Zeit [s]_{i}'].values)
                                depth_diff = np.diff(interval_values_data_frame[f'Teufe [m]_{i}'])

                                # Calculate the mean and standard deviation of the differences
                                time_mean = time_diff.mean()
                                time_std = time_diff.std()

                                depth_mean = depth_diff.mean()
                                depth_std = depth_diff.std()

                                # Define the threshold as a certain number of standard deviations from the mean
                                time_threshold = time_mean + 3 * time_std
                                depth_threshold = depth_mean + 3 * depth_std

                                # Find the indices where the differences exceed the thresholds
                                high_diff_indices = np.where(time_diff > time_threshold)[
                                    0]  # Indices where time difference exceeds the threshold
                                high_diff_indices = np.append(high_diff_indices,
                                                              np.where(depth_diff > depth_threshold)[
                                                                  0])  # Indices where depth difference exceeds the threshold

                                interval_values_depth = np.delete(interval_values_depth, high_diff_indices + 1)
                                interval_values_time = np.delete(interval_values_time, high_diff_indices + 1)

                                depth_interval.append(interval_values_depth)
                                time_interval.append(interval_values_time)

                                for col in cols:
                                    new_column = col + '_' + f'{i}'
                                    feature = interval_values_data_frame[new_column]
                                    feature = np.array(feature)
                                    feature = feature[~np.isnan(feature)]
                                    feature = np.delete(feature, high_diff_indices + 1)
                                    feature = feature.astype(float)

                                    mean_col_label = col + ' ' + 'Mean'
                                    std_col_label = col + ' ' + 'Standard Dev.'

                                    feature = feature[~np.isnan(feature)]

                                    statistical_df.loc[i, mean_col_label] = np.mean(feature)
                                    statistical_df.loc[i, std_col_label] = np.std(feature)

            start_index = end_index

        return statistical_df, statistical_df.shape[0], interval_count, depth_interval, time_interval

    @staticmethod
    def Formations(data_set, depth_intervals, formations, data_type):
        # Create a list to store the formations for each interval
        interval_formations = []

        if data_type == 'prior data':
            for depth in data_set.loc[:, 'Teufe [m]']:
                assigned_formation = None

                # Check if the depth falls within any interval
                for i, interval in enumerate(depth_intervals):
                    if interval[0] <= depth <= interval[1]:
                        assigned_formation = formations[i]
                        break

                interval_formations.append(assigned_formation)
                
            data_set.loc[:, 'Formation'] = interval_formations

        elif data_type == 'stats data':
            for depth in data_set.loc[:, 'Teufe [m] Mean']:
                assigned_formation = None

                # Check if the depth falls within any interval
                for i, interval in enumerate(depth_intervals):
                    if interval[0] <= depth <= interval[1]:
                        assigned_formation = formations[i]
                        break

                interval_formations.append(assigned_formation)

            data_set.loc[:, 'Formation'] = interval_formations

        return data_set


st.cache_data(ttl=3600)
def Excel(df, sheet):
    import pandas as pd
    from io import BytesIO
    from pyxlsb import open_workbook as open_xlsb
    import xlsxwriter

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name=sheet)
    workbook = writer.book
    worksheet = writer.sheets[sheet]

    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processed_data = output.getvalue()

    return processed_data


@st.cache_data(ttl=3600)
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
    

def DI(dataframe):
    import math
    
    dataframe.loc[:, 'Andruck [kPa]'] = dataframe.loc[:, 'Andruck [bar]'] * 100
    dataframe.loc[:, 'Gravitational Force [kN]'] = dataframe.loc[:, 'Andruck [kPa]'] * math.pi * (0.152**2) / 4

    drillibility = []

    for pr, n, w in zip(dataframe.loc[:, 'vB [m/h]'], dataframe.loc[:, 'DZ [U/min]'], dataframe.loc[:, 'Gravitational Force [kN]']):
        if pr == 0:
            di = 0
            
        else:
            di = (3.35 * n * w) / (15.2 * pr)
            
        drillibility.append(di)

    dataframe.loc[:, 'Drillibility Index [kN/mm]'] = drillibility

    return dataframe
    

def Start():
    st.session_state.start = True


def Display():
    st.session_state.display = True


def Control(data, cols, length, error, int_depth=None, rocks=None, number_of_intervals=None):
    required_columns = ["Zeit [s]", "Delta Zeit [s]", "Teufe [m]", "Delta Teufe [m]", "p Luft [bar]", "DZ [U/min]"]
    flag_control = -3


    
    if int_depth is None and number_of_intervals is None and rocks is None:
        if all(col in cols for col in required_columns) and all(col in data.columns for col in required_columns):
            flag_control = 1

        
    if cols and length and error and int_depth and number_of_intervals and rocks:
        if all(col in cols for col in required_columns) and all(col in data.columns for col in required_columns):
            if all(value[0] < value[1] for value in int_depth) and all(rocks):
                if number_of_intervals >= 1:
                    flag_control = 1

                else:
                    flag_control = -1  # Number of intervals are missing

            else:
                flag_control = -2  # Formation and depth missing
                
        else:
            flag_control = 0  # Required columns are missing

    st.session_state.flag = flag_control

    return flag_control


@st.cache_resource(ttl=3600)
def plot_intervals(interval_depths, interval_times, available_data):
    import plotly.graph_objects as go
    import numpy as np

    fig = go.Figure()

    for depth, time in zip(interval_depths, interval_times):
        fig.add_trace(go.Scatter(x=time, y=depth, mode='markers', marker=dict(color='blue'),
                                 showlegend=False, name='Interval Measurements', texttemplate='presentation'))

        last_time = time[-1]
        fig.add_shape(
            type="line",
            x0=last_time,
            y0=min(depth),
            x1=last_time,
            y1=max(depth),
            line=dict(color="red", width=1, dash='dash'),
            name='Interval Boundary'
        )

        fig.add_trace(go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([depth - (2.0 - 1.96 * np.std(depth)),
                              depth[::-1] + (2.0 - 1.96 * np.std(depth[::-1]))]),
            fill='toself',
            fillcolor='gray',
            opacity=0.5,
            mode='none',
            showlegend=False
        ))

    fig.update_layout(
        xaxis=dict(title="Measured Time, s"),
        yaxis=dict(title="Measured Depth, m",
                   range=[max(available_data['Teufe [m]']), min(available_data['Teufe [m]'])]),
        title="Intervals Based on Math Algorithm",
        showlegend=True
    )

    st.plotly_chart(fig)


def main():

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
    <div class='stTitle'>Drilling Intervals Automation</div>
    """, unsafe_allow_html=True)
    
    st.divider()

    if 'dropped_data' not in st.session_state:
        st.session_state.dropped_data = None  # Initialize the dropped_data attribute

    if st.session_state.dropped_data is None:
        st.warning('**To be able to proceed on Automation part, please complete the step one:** *Data Preprocessing*',
                   icon='💹')

    elif not st.session_state.dropped_data.empty:
        file_name = st.session_state.uploaded_data.name
        if file_name.endswith(".xlsx"):
            file_name_without_extension = file_name[:-5]
        elif file_name.endswith(".csv"):
            file_name_without_extension = file_name[:-4]

        if 'file_name' not in st.session_state:
            st.session_state.file_name = file_name_without_extension

        st.markdown(f"""
        <div class='stHeader'>Drilling Interval Properties for <i>{file_name_without_extension}</i></div>
        """, unsafe_allow_html=True)
        
        st.info(
            'The following columns that are chosen by default, are necessary to select before proceeding the calculations. If these columns are not found, the algorithm will be giving an error.')

        pipe_length = st.number_input('**Drill Pipe Length, in meters**', min_value=1)
        error_rate = st.number_input('**Error Rate, in %**', min_value=0, max_value=100)
        threshold = st.number_input('**Threshold, an integer**', min_value=1, max_value=15)

        check_formation = st.checkbox('**Do you have formation informations?**')

        if 'check_formation' not in st.session_state:
            st.session_state.check_formation = check_formation

        depth_intervals = []
        formations = []
        
        if check_formation:
            st.markdown(f"""
            <div class='stHeader'>Encountered Formations for <i>{file_name_without_extension}</i></div>
            """, 
            unsafe_allow_html=True)
        
            num_intervals = st.number_input('**Number of Formation Interval**', min_value=1, value=3, step=1)

            for i in range(num_intervals):
                interval_start = st.number_input(f'**Interval {i + 1} Start, in m**', value=0.0, step=0.1)
                interval_end = st.number_input(f'**Interval {i + 1} End, in m**', value=0.0, step=0.1)
                formation = st.text_input(f'**Formation for Interval {i + 1}**', value='', placeholder='Gneiss')

                depth_intervals.append((interval_start, interval_end))
                formations.append(formation)

        else:
            num_intervals = None
            depth_intervals = None
            formations = None

        
        check_water = st.checkbox('**Do you have formartion water information?**')

        if 'check_water' not in st.session_state:
            st.session_state.check_water = check_water

        if check_water:
            formation_water = st.number_input('**Encountered Formation Water Depth, in m**', min_value=0)

            if 'formation_water' not in st.session_state or st.session_state.formation_water != formation_water:
                st.session_state.formation_water = formation_water
        
        columns = st.multiselect(
            label='**Columns to be used**',
            options=st.session_state.dropped_data.columns,
            default=['Zeit [s]', 'Delta Zeit [s]', 'Teufe [m]', 'Delta Teufe [m]', 'p Luft [bar]', 'DZ [U/min]']
        )


        st.button('Start', on_click=Start, type='primary')
        display_chart = False

        if 'start' not in st.session_state:
            st.session_state.start = False

        if st.session_state.start:
            flag = Control(st.session_state.dropped_data, columns, pipe_length, error_rate, depth_intervals, formations,
                           num_intervals)

            if 'flag' not in st.session_state:
                st.session_state.flag = flag

            if st.session_state.flag == 1:
                available_data = Intervals.outliers(st.session_state.dropped_data, columns, threshold)

                if 'p Luft [bar]' in columns and 'Q Luft [Nm3/min]' in columns:
                    a = Intervals.Energy(available_data)
                
                    if 'DZ [U/min]' in columns and 'Andruck [bar]' in columns and 'vB [m/h]' in columns:
                        prior_data = DI(a)
                        
                    else:
                        prior_data = available_data
                        
                elif 'DZ [U/min] Mean' in columns and 'Andruck [bar] Mean' in columns and 'vB [m/h] Mean' in columns:
                    prior_data = DI(available_data)
                    
                else:
                    prior_data = available_data

                if 'prior_data' not in st.session_state:
                    st.session_state.prior_data = prior_data

                stats_data = \
                    Intervals.CalculateIntervals(st.session_state.prior_data, columns, pipe_length, error_rate)[0]

                if 'stats_data' not in st.session_state:
                    st.session_state.stats_data = stats_data

                intervals = Intervals.CalculateIntervals(st.session_state.prior_data, columns, pipe_length, error_rate)[1]
                counted_interval = Intervals.CalculateIntervals(st.session_state.prior_data, columns, pipe_length, error_rate)[2]

                st.markdown(f"""
                <div class='stHeader'><i>{file_name_without_extension}</i> After Outlier Removal Statistics</div>
                """, unsafe_allow_html=True)
                
                st.table(data=st.session_state.prior_data.describe())
                st.divider()

                if check_formation:
                
                    prior_data_rocks = Intervals.Formations(st.session_state.prior_data, depth_intervals, formations, 'prior data')
                    stats_data_rocks = Intervals.Formations(st.session_state.stats_data, depth_intervals, formations, 'stats data')

                if not check_formation:
                    prior_data_rocks = st.session_state.prior_data
                    stats_data_rocks = st.session_state.stats_data

                if not prior_data_rocks.empty and not stats_data_rocks.empty:
                    if 'prior_data_rocks' not in st.session_state:
                        st.session_state.prior_data_rocks = prior_data_rocks
    
                    if 'stats_data_rocks' not in st.session_state:
                        st.session_state.stats_data_rocks = stats_data_rocks
    
                    st.markdown(f"""
                    <div class='stHeader'><i>{file_name_without_extension}</i> Intervals Data</div>
                    """, unsafe_allow_html=True)

                    if 'DZ [U/min] Mean' in columns and 'Andruck [bar] Mean' in columns and 'vB [m/h] Mean' in columns:
                        st.session_state.stats_data_rocks = DI(st.session_state.stats_data_rocks)

                    st.table(data=st.session_state.stats_data_rocks)
                    st.caption(f'**Calculated Number of Intervals:** {counted_interval}')
                    st.caption(f'**Number of Intervals in Dataset:** {intervals}')
    
                    display_chart = True
                    st.markdown(f"""
                    <div class='stHeader'>Download the Resulted Datasets for <i>{file_name_without_extension}</i></div>
                    """, unsafe_allow_html=True)
                    col1, col2 = st.columns(2, gap='large')

                    prior_data_rocks_excel = Excel(st.session_state.prior_data_rocks, 'Prior Data')
                    stats_data_rocks_excel = Excel(st.session_state.stats_data_rocks, 'Interval Data')

                    with col1:
                        st.download_button(label='Download Prior Data',
                                          data=prior_data_rocks_excel,
                                          file_name='Prior_data.xlsx')

                    with col2:
                        st.download_button(label='Download Interval Data',
                                          data=stats_data_rocks_excel,
                                          file_name='Interval_data.xlsx')
                    
                else:
                    st.error('According to the algorithm, the dataset is found to be not convenient.', icon='🛑')

            elif st.session_state.flag == 0:
                st.warning('Please do not forget to select the following columns: *Zeit*, *Teufe*, *Delta Zeit*, *p Luft*, *DZ*', icon='💹')

            elif st.session_state.flag == -1:
                st.warning('Number of formation intervals cannot be less than 1!', icon='💹')
            elif st.session_state.flag == -2:
                st.warning(
                    "The following interval's start measurement cannot be less than previous formation's end boundary!",
                    icon='💹')
            elif st.session_state.flag == -3:
                st.warning('Please fill the above entries to proceed into calculations!', icon='💹')
            else:
                st.error('Something went wrong!')


        else:
            st.warning('Please press Start button to proceed!', icon='💹')

        if display_chart:
            with st.form('Chart'):
                if 'display' not in st.session_state:
                    st.session_state.display = False

                st.form_submit_button('Display Plot', on_click=Display, type='primary')

                if st.session_state.display:
                    interval_depths, interval_times = Intervals.CalculateIntervals(st.session_state.prior_data,
                                                                                   columns, pipe_length, error_rate)[3:5]

                    plot_intervals(interval_depths, interval_times, st.session_state.prior_data)

    else:
        st.warning('**This dataset is empty. Probably something went wrong. Please repeat the calculations.', icon='💹')


if __name__ == '__main__':
    main()
