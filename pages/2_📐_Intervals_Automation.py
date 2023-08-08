import streamlit as st

st.set_page_config(
    page_title="Drilling Intervals Automation",
    page_icon="📐"
    )


class Intervals:
    @staticmethod
    @st.cache_resource(ttl=3600)
    def custom_auc_pr_score(y_true, outlier_scores):
        from sklearn.metrics import precision_recall_curve, auc

        precision, recall, _ = precision_recall_curve(y_true, outlier_scores)
        auc_pr = auc(recall, precision)

        return auc_pr
    

    @staticmethod
    @st.cache_resource(ttl=3600)
    def custom_grid_search(X, n_estimators_list, max_samples_list, contamination_list):
        from sklearn.ensemble import IsolationForest
        import numpy as np

        best_params = None
        best_score = None

        for n in n_estimators_list:
            for s in max_samples_list:
                for c in contamination_list:
                    model = IsolationForest(n_estimators=n, max_samples=s, contamination=c)
                    model.fit(X)

                    outliers = model.predict(X)

                    synthetic_labels = np.ones(len(outliers))
                    synthetic_labels[outliers == -1] = 0

                    auc_pr = Intervals.custom_auc_pr_score(synthetic_labels, -model.decision_function(X))

                    if best_score is None or auc_pr > best_score:
                        best_score = auc_pr
                        best_params = {'n_estimators': n, 'max_samples': s, 'contamination': c}

        return best_params, best_score


    @staticmethod
    @st.cache_resource(ttl=3600)
    def outliers(data, cols, threshold):
        import pandas as pd
        import numpy as np
        from scipy import stats
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import precision_recall_curve, auc
        import re

        data_frame = data.copy()

        delta_teufe = np.array(data.filter(regex=r'(?i)^Delta Teufe'))
        teufe = np.array(data.filter(regex=r'(?i)^Teufe'))

        unsatisfied_indexes_teufe = np.where((teufe < 0) | (delta_teufe <= 0.001))[0]
        custom_outlier = len(unsatisfied_indexes_teufe)

        removed_data = data.drop(data.index[unsatisfied_indexes_teufe])

        p_luft = np.array(removed_data.filter(regex=r'(?i)^p Luft'))
        
        unsatisfied_indexes_pluft = np.where(p_luft < 2)[0]
        removed_data = removed_data.drop(removed_data.index[unsatisfied_indexes_pluft])
        custom_outlier = custom_outlier + len(unsatisfied_indexes_pluft)

        DZ = np.array(removed_data.filter(regex=r'(?i)^DZ'))
        dz_mean = DZ.mean()
        dz_std = DZ.std()

        upper = dz_mean + 3 * dz_std
        lower = dz_mean - 3 * dz_std

        unsatisfied = np.where((DZ > upper) | (DZ < lower))[0]
        custom_outlier = custom_outlier + len(unsatisfied)

        removed_data = removed_data.drop(removed_data.index[unsatisfied])

        if isinstance(removed_data, pd.DataFrame):
            # Create a list of the filtered values for each column
            filtered_values_list = []
            z = None

            for col in cols:
                if re.match(r'^(p Luft|DZ|Q Luft)', col, re.IGNORECASE):
                    desired_values = removed_data[col]

                    z_score_desired_values = stats.zscore(desired_values)
                    filtered_desired_values = desired_values[np.abs(z_score_desired_values) < threshold]

                    if len(filtered_desired_values) > 0:
                        if z is None:
                            z = len(desired_values[np.abs(z_score_desired_values) >= threshold])
                        else:
                            if z <  len(desired_values[np.abs(z_score_desired_values) >= threshold]):
                                z = len(desired_values[np.abs(z_score_desired_values) >= threshold])

                        filtered_values_list.append(filtered_desired_values)

            min_length_z = min(len(l) for l in filtered_values_list)
            sliced_arrays_z = [arr[:min_length_z] for arr in filtered_values_list]

            columns_z = []

            for col in cols:
                match = re.match(r'^(p Luft|DZ|Q Luft)', col, re.IGNORECASE)

                if match:
                    columns_z.append(col)

            data_z = {col: arr for col, arr in zip(columns_z, sliced_arrays_z)}
            df_z = pd.DataFrame(data_z)

            data_frame = data_frame.iloc[:len(df_z)]
            data_frame.loc[:, columns_z] = data_frame.loc[:, columns_z].replace(data_z)
            data_frame.reset_index(drop=True, inplace=True)

            X = np.array(data_frame.filter(regex=r'(?i)^vB'))
            X_2 = np.array(data_frame.filter(regex=r'(?i)^Delta Zeit')) 

            n_estimators = [50, 100, 200, 300, 400, 500]
            max_samples = [0.1, 0.2, 0.5, 0.6, 0.7, 0.8]
            contamination = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]

            best_params, best_score = Intervals.custom_grid_search(X, n_estimators, max_samples, contamination)
            best_params_teufe, best_score_teufe = Intervals.custom_grid_search(X_2, n_estimators, max_samples, contamination) 

            best_model = IsolationForest(n_estimators=best_params['n_estimators'], max_samples=best_params['max_samples'], contamination=best_params['contamination'])
            best_model_teufe = IsolationForest(n_estimators=best_params_teufe['n_estimators'], max_samples=best_params_teufe['max_samples'], contamination=best_params_teufe['contamination'])

            best_model.fit(X)
            best_model_teufe.fit(X_2) 

            outliers = best_model.predict(X)
            outliers_teufe = best_model_teufe.predict(X_2)

            combined_outliers = np.logical_or(outliers == -1, outliers_teufe == -1) 
            indices = np.where(combined_outliers)[0]

            ml_outlier = np.sum(combined_outliers)

            total_outlier = None

            if z is not None:
                total_outlier = ml_outlier + custom_outlier + z
            
            else:
                total_outlier = ml_outlier + custom_outlier

            cleaned_data = data_frame.drop(indices)

        return cleaned_data, best_params, best_score, ml_outlier, custom_outlier, z, total_outlier, best_score_teufe, best_params_teufe

    @staticmethod
    def Energy(data):
        energy = []

        for p, q in zip(data.loc[:, 'p Luft [bar]'].values, data.loc[:, 'Q Luft [Nm3/min]'].values):
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
        import re

        global interval_count

        # Get the column names dynamically
        statistical_columns = ['{} Mean'.format(col) for col in cols]
        statistical_columns += ['{} Standard Dev.'.format(col) for col in cols]

        statistical_df = pd.DataFrame(columns=statistical_columns)

        for col in cols:
            if re.match(r'^(Andruck|DZ|vB)', col, re.IGNORECASE):
                cols.append('Pseudo Drillibility Index [kN/mm]')

            if re.match(r'^(p Luft|Q Luft)', col, re.IGNORECASE):
                cols.append('Hydraulic Power [kW]')

        teufe_columns = [col for col in cols if col.strip().lower().startswith('teufe')]
        zeit_columns = [col for col in cols if col.strip().lower().startswith('zeit')]

        depth = np.array(data.loc[:, teufe_columns[0]].values)
        deltaTime = np.array(data.filter(regex=r'(?i)^Delta Zeit').values)

        interval_count = np.around((np.max(depth) - np.min(depth)) / interval_size).astype(int)
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
                interval_values_depth = np.array(data.loc[:, teufe_columns[0]].values)[start_index:end_index + 1]
                interval_values_time = np.array(data.loc[:, zeit_columns[0]].values)[start_index:end_index + 1]

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
                        if len(interval_values_data_frame.loc[:, zeit_columns[0]].values) >= 3 and len(
                                interval_values_data_frame.loc[:, teufe_columns[0]].values) >= 3:
                            if np.all(np.gradient(interval_values_data_frame.loc[:, zeit_columns[0]].values) != 0) and np.all(
                                    np.gradient(interval_values_data_frame.loc[:, teufe_columns[0]].values) != 0):

                                # Calculate the differences between consecutive values
                                time_diff = np.diff(interval_values_data_frame.loc[:, zeit_columns[0]].values)
                                depth_diff = np.diff(interval_values_data_frame.loc[:, teufe_columns[0]].values)

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
                        if len(interval_values_data_frame[f'{teufe_columns[0]}_{i}'].values) >= 3 and len(
                                interval_values_data_frame[f'{zeit_columns[0]}_{i}'].values) >= 3:
                            if np.all(np.gradient(
                                    interval_values_data_frame[f'{zeit_columns[0]}_{i}'].values) != 0) and np.all(np.gradient(interval_values_data_frame[f'{teufe_columns[0]}_{i}'].values) != 0):

                                # Calculate the differences between consecutive values
                                time_diff = np.diff(interval_values_data_frame[f'{zeit_columns[0]}_{i}'].values)
                                depth_diff = np.diff(interval_values_data_frame[f'{teufe_columns[0]}_{i}'])

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
        teufe_column = [col for col in data_set.columns if col.strip().lower().startswith('teufe')]

        if data_type == 'prior data':
            for depth in data_set.loc[:, teufe_column[0]]:
                assigned_formation = None

                # Check if the depth falls within any interval
                for i, interval in enumerate(depth_intervals):
                    if interval[0] < depth <= interval[1]:
                        assigned_formation = formations[i]
                        break

                    else:
                        diff = depth - interval[1]

                        if round(diff, 2) < 3:
                            assigned_formation = formations[i]
                            break

                interval_formations.append(assigned_formation)
                
            data_set.loc[:, 'Formation'] = interval_formations

        elif data_type == 'stats data':
            for depth in data_set.loc[:, teufe_column[0]]:
                assigned_formation = None

                # Check if the depth falls within any interval
                for i, interval in enumerate(depth_intervals):
                    if interval[0] < depth <= interval[1]:
                        assigned_formation = formations[i]
                        break

                    else:
                        diff = depth - interval[1]

                        if round(diff, 2) < 3:
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
    dataframe.loc[:, 'Pseudo Gravitational Force [kN]'] = dataframe.loc[:, 'Andruck [kPa]'] * math.pi * (0.152**2) / 4

    drillibility = []

    for pr, n, w in zip(dataframe.loc[:, 'vB [m/h]'], dataframe.loc[:, 'DZ [U/min]'], dataframe.loc[:, 'Pseudo Gravitational Force [kN]']):
        if pr == 0:
            di = 0
            
        else:
            di = (3.35 * n * w) / (15.2 * pr)
            
        drillibility.append(di)

    dataframe.loc[:, 'Pseudo Drillibility Index [kN/mm]'] = drillibility
    

    return dataframe


@st.cache_resource(ttl=3600)
def WOB(interval_df, Wp=None, Wh=None, Wb=None, Axial=None):
    interval_df_copy = interval_df.copy()
    all_params = 0

    if Wp is not None and Wh is not None and Wb is not None and Axial is not None:
        for i in range(len(interval_df_copy)):
            if i == 0:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = Wp + Wh + Wb + (Axial[i] * 1000)
            
            else:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = interval_df_copy.loc[i-1, 'Section WOB [kg]'] + Wp + (Axial[i] * 1000)

        all_params = 1

    elif Wp is not None and Wh is None and Wb is not None and Axial is not None:
        for i in range(len(interval_df_copy)):
            if i == 0:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = Wp + Wb + (Axial[i] * 1000)
            
            else:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = interval_df_copy.loc[i-1, 'Section WOB [kg]'] + Wp + (Axial[i] * 1000)

    elif Wp is not None and Wb is None and Wh is not None and Axial is not None:
        for i in range(len(interval_df_copy)):
            if i == 0:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = Wp + Wh + (Axial[i] * 1000)
            
            else:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = interval_df_copy.loc[i-1, 'Section WOB [kg]'] + Wp + (Axial[i] * 1000)
            

    elif Wp is not None and Wb is None and Wh is None and Axial is not None:
        for i in range(len(interval_df_copy)):
            if i == 0:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = Wp + (Axial[i] * 1000)
            
            else:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = interval_df_copy.loc[i-1, 'Section WOB [kg]'] + Wp + (Axial[i] * 1000)

    
    elif Wp is not None and Wb is None and Wh is None and Axial is None:
        for i in range(len(interval_df_copy)):
            if i == 0:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = Wp 
            
            else:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = interval_df_copy.loc[i-1, 'Section WOB [kg]'] + Wp

    elif Wp is not None and Wb is not None and Wh is not None and Axial is None:
        for i in range(len(interval_df_copy)):
            if i == 0:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = Wp + Wb + Wh
            
            else:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = interval_df_copy.loc[i-1, 'Section WOB [kg]'] + Wp
    
    elif Wp is not None and Wb is not None and Wh is None and Axial is None:
        for i in range(len(interval_df_copy)):
            if i == 0:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = Wp + Wb 
            
            else:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = interval_df_copy.loc[i-1, 'Section WOB [kg]'] + Wp
        
    elif Wp is not None and Wb is None and Wh is not None and Axial is None:
        for i in range(len(interval_df_copy)):
            if i == 0:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = Wp + Wh
            
            else:
                interval_df_copy.loc[i, 'Section WOB [kg]'] = interval_df_copy.loc[i-1, 'Section WOB [kg]'] + Wp

    return interval_df_copy, all_params


def MSE(dataframe, d):
    import math

    diameter = 0.0393700787 * d

    for i in range(len(dataframe)):
        wob = dataframe.loc[i, 'Section WOB [kg]'] * 2.222
        N = dataframe.loc[i, 'DZ [U/min] Mean']
        T = dataframe.loc[i, 'DM [Nm] Mean'] * 0.737463126
        ROP = dataframe.loc[i, 'vB [m/h] Mean'] * 3.2808399

        mse = ((4 * wob) / (math.pi * diameter**2)) + ((8 * N * T) / (diameter**2 * ROP))
        mse = mse * 0.0689475729

        dataframe.loc[i, 'MSE [bar]'] = round(mse, 3)

    return dataframe

def Start():
    st.session_state.start = True


def Display():
    st.session_state.display = True


def Control(data, cols, length, error, int_depth=None, rocks=None, number_of_intervals=None):
    required_columns = ["Zeit [s]", "Delta Zeit [s]", "Teufe [m]", "Delta Teufe [m]", "p Luft [bar]", "DZ [U/min]", "Q Luft [Nm3/min]"]
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


def plot_intervals(interval_depths, interval_times, available_data):
    import plotly.graph_objects as go
    import numpy as np

    fig = go.Figure()

    
    fig.add_trace(go.Scatter(x=available_data['Zeit [s]'], y=available_data['Teufe [m]'], mode='markers', marker=dict(color='blue'),
                                showlegend=True, name='Interval Measurements', texttemplate='presentation', hovertemplate="Time: %{x} seconds <br>Depth: %{y} meters"))

    for depth, time in zip(interval_depths, interval_times):
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
            showlegend=False,
            name='Confidence Interval',
            hovertemplate="Time: %{x} seconds <br>Depth: %{y} meters"
        ))
    
    fig.add_trace(go.Scatter(x=[None, None], y=[None, None], mode='lines+text',
                                    line=dict(color='red', width=2, dash='dash'),
                                    text=['Interval Boundary'],
                                    textposition='top center',
                                    name='Interval Boundary', 
                                    opacity=0.8, showlegend=True))

    fig.update_layout(
        xaxis=dict(title="Measured Time, s"),
        yaxis=dict(title="Measured Depth, m",
                   range=[max(available_data['Teufe [m]']), min(available_data['Teufe [m]'])]),
        title="Intervals Based on Math Algorithm"
    )

    fig.update_layout(height=600, hovermode='closest',
                        title=dict(
                        xref='paper',
                        x=0.5,
                        font=dict(size=16),
                        xanchor='center'
                    ))

    st.plotly_chart(fig)


def main():
    import pandas as pd
    import numpy as np
    import re

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
    <div class='stTitle'><center>Drilling Intervals Automation</center></div>
    """, unsafe_allow_html=True)
    
    st.markdown('                            ')
    st.markdown('                            ')

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
            '**The following columns that are chosen by default, are necessary to select before proceeding the calculations. If these columns are not found, the algorithm will be giving an error.**')

        pipe_length = st.number_input('**Drill Pipe Length, in meters**', min_value=1.0, step=0.01)
        error_rate = st.number_input('**Error Rate, in %**', min_value=0.0, max_value=100.0, step=0.01)
        threshold = st.number_input('**Threshold, an integer**', min_value=1.0, max_value=15.0, step=1.0)

        pipe_check = st.checkbox('**Do you have the drill pipe weight information?**')

        if 'pipe_info' not in st.session_state:
            st.session_state.pipe_info = False

        if pipe_check:
            pipe_weight = st.number_input('**Please specify the weight of one drill pipe, in kg**', min_value=0.0, step=0.01)

            st.session_state.pipe_info = True
            st.session_state.pipe_weight = pipe_weight

        if 'hammer_weight' not in st.session_state:
            st.session_state.hammer_weight = None

        hammer_check = st.checkbox('**Do you have the hammer weight information?**')

        if hammer_check:
            hammer_weight = st.number_input('**Please specify the weight of the hammer, in kg**', min_value=0.0, step=0.01)

            st.session_state.hammer_weight = hammer_weight

        if 'bit_weight' not in st.session_state:
            st.session_state.bit_weight = None
        
        if 'bit_diameter' not in st.session_state:
            st.session_state.bit_diameter = None

        bit_check = st.checkbox('**Do you have the bit informations?**')

        if bit_check:
            bit_weight = st.number_input('**Please specify the weight of the bit, in kg**', min_value=0.0, step=0.01)
            bit_diameter = st.number_input('**Please specify the diameter of the bit, in mm**', min_value=0.0, step=0.01)

            st.session_state.bit_weight = bit_weight
            st.session_state.bit_diameter = bit_diameter

        check_formation = st.checkbox('**Do you have formation informations?**')

        depth_intervals = []
        formations = []
        
        if 'formation_info' not in st.session_state:
            st.session_state.formation_info = False

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
            
            st.session_state.formation_info = True

        else:
            num_intervals = None
            depth_intervals = None
            formations = None

            st.session_state.formation_info = False

        
        check_water = st.checkbox('**Do you have formartion water information?**')
        
        if 'water_info' not in st.session_state:
            st.session_state.water_info = False

        if 'formation_water' not in st.session_state:
            st.session_state.formation_water = None

        if check_water:
            formation_water = st.number_input('**Encountered Formation Water Depth, in m**', min_value=0.0, value=0.0, step=0.01)
            st.session_state.formation_water = formation_water

            st.session_state.water_info = True
        
        else:
            st.session_state.water_info = False

        columns_used = []

        for col in st.session_state.dropped_data.columns:
            match = re.match(r'^(p Luft|DZ|Q Luft|Zeit|Delta Zeit|Teufe|Delta Teufe|vB|Andruck|Drehdruck|WOB|DM)', col, re.IGNORECASE)

            if match:
                columns_used.append(col)

        columns = st.multiselect(
            label='**Columns to be used**',
            options=st.session_state.dropped_data.columns,
            default=columns_used,
            disabled=True
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
                outlier = Intervals.outliers(st.session_state.dropped_data, columns, threshold)
                available_data = outlier[0]

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

                stats_data = \
                    Intervals.CalculateIntervals(prior_data, columns, pipe_length, error_rate)[0]
                
                if 'axial_force' not in st.session_state:
                    st.session_state.axial_force = None
                
                if 'WOB [t] Mean' in stats_data.columns:
                    st.session_state.axial_force = stats_data.loc[:, 'WOB [t] Mean']

                if 'torque' not in st.session_state:
                    st.session_state.torque = None

                if 'DM [Nm] Mean' in stats_data.columns:
                    st.session_state.torque = stats_data.loc[:, 'DM [Nm] Mean'].values


                intervals = Intervals.CalculateIntervals(prior_data, columns, pipe_length, error_rate)[1]
                counted_interval = Intervals.CalculateIntervals(prior_data, columns, pipe_length, error_rate)[2]
                st.divider()
                st.markdown('                            ')

                st.markdown("""
                            <div class='stHeader'><center>Outlier Detection Model</center>
                            """, unsafe_allow_html=True)
                        

                st.markdown(f"""
                            <center>
                                <table class='justify-content-center'>
                                    <tr>
                                        <th><center>ML Model</center></th>
                                        <th><center>ML Model Score on vB</center></th>
                                        <th><center>ML Model Score on Delta Zeit</center></th>
                                        <th><center>Custom Model</center></th>
                                        <th><center>Statistics Model</center></th>
                                    </tr>
                                    <tr>
                                        <td><center>Isolation Forest</center></td>
                                        <td><center>{round(outlier[2], 2)}</center></td>
                                        <td><center>{round(outlier[7], 2)}</center></td>
                                        <td><center>Checkout Article</center></td>
                                        <td><center>Z-Score</center></td>
                                    </tr>
                                </table>
                            </center>   
                            """, unsafe_allow_html=True)
                
                st.markdown('                            ')
                st.markdown('                            ')

                vb = outlier[1]
                delta_zeit = outlier[8]

                st.markdown("""
                            <div class='stHeader'><center>Outlier Detection on vB Model Hyperparameters</center>
                            """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <center>
                        <table class='justify-content-center'>
                            <tr>
                                <th><center>Hyperparameter</center></th>
                                <th><center>Value</center></th>
                            </tr>
                            {"".join(f"<tr><td><center>{key}</center></td><td><center>{value}</center></td></tr>" for key, value in vb.items())}
                        </table>
                    </center>
                    """, unsafe_allow_html=True)
                
                st.markdown('                            ')

                st.markdown("""
                            <div class='stHeader'><center>Outlier Detection on Delta Zeit Model Hyperparameters</center>
                            """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <center>
                        <table class='justify-content-center'>
                            <tr>
                                <th><center>Hyperparameter</center></th>
                                <th><center>Value</center></th>
                            </tr>
                            {"".join(f"<tr><td><center>{key}</center></td><td><center>{value}</center></td></tr>" for key, value in delta_zeit.items())}
                        </table>
                    </center>
                    """, unsafe_allow_html=True)


                st.markdown('                            ')
                st.markdown('                            ')

                st.markdown("""
                            <div class='stHeader'><center>Observed Number of Outliers</center>
                            """, unsafe_allow_html=True)

                st.markdown(f"""
                            <center>
                                <table class='justify-content-center'>
                                    <tr>
                                        <th><center>ML Model</center></th>
                                        <th><center>Custom Model</center></th>
                                        <th><center>Z-Score</center></th>
                                        <th><center>Total</center></th>
                                    </tr>
                                    <tr>
                                        <td><center>{outlier[3]}</center></td>
                                        <td><center>{outlier[4]}</center></td>
                                        <td><center>{outlier[5]}</center></td>
                                        <td><center>{outlier[6]}</center></td>
                                    </tr>
                                </table>
                            </center>
                            """, unsafe_allow_html=True)
                
                st.markdown('                            ')
                st.markdown('                            ')
                
                st.markdown(f"""
                    <div class='stHeader'><center><i>{file_name_without_extension}</i>After Outlier Removal Statistics</center></div>
                    """, unsafe_allow_html=True)
                
                st.table(data=prior_data.describe())
                st.divider()

                if 'prior_data_rocks' not in st.session_state:
                    st.session_state.prior_data_rocks = None

                if 'stats_data_rocks' not in st.session_state:
                    st.session_state.stats_data_rocks = None

                if st.session_state.formation_info and st.session_state.pipe_info:
                    
                    prior_data_rocks = Intervals.Formations(prior_data, depth_intervals, formations, 'prior data')
                    stats_data_rocks = Intervals.Formations(stats_data, depth_intervals, formations, 'stats data')

                        
                    stats_data_rocks = WOB(stats_data_rocks, st.session_state.pipe_weight, st.session_state.hammer_weight, st.session_state.bit_weight, st.session_state.axial_force)[0]
                    param_check = WOB(stats_data_rocks, st.session_state.pipe_weight, st.session_state.hammer_weight, st.session_state.bit_weight, st.session_state.axial_force)[1]

                    if param_check == 1 and isinstance(st.session_state.torque, np.ndarray):
                        stats_data_rocks = MSE(stats_data_rocks, bit_diameter)

                elif not st.session_state.formation_info and st.session_state.pipe_info:
                    stats_data_rocks = WOB(stats_data, st.session_state.pipe_weight, st.session_state.hammer_weight, st.session_state.bit_weight, st.session_state.axial_force)[0]
                    param_check = WOB(stats_data, st.session_state.pipe_weight, st.session_state.hammer_weight, st.session_state.bit_weight, st.session_state.axial_force)[1]

                    if param_check == 1 and isinstance(st.session_state.torque, np.ndarray):
                        stats_data_rocks = MSE(stats_data_rocks, bit_diameter)
                    
                    prior_data_rocks = prior_data
                    
                elif not st.session_state.pipe_info and st.session_state.formation_info:
                    prior_data_rocks = Intervals.Formations(prior_data, depth_intervals, formations, 'prior data')
                    stats_data_rocks = Intervals.Formations(stats_data, depth_intervals, formations, 'stats data')
                            
                elif not st.session_state.formation_info and not st.session_state.pipe_info:         
                    prior_data_rocks = prior_data
                    stats_data_rocks = stats_data
                
                if prior_data_rocks is not None and stats_data_rocks is not None:
                    if not prior_data_rocks.empty and not stats_data_rocks.empty:
                        st.markdown(f"""
                        <div class='stHeader'><center><i>{file_name_without_extension}</i> Intervals Data</center></div>
                        """, unsafe_allow_html=True)

                        unit_check = False

                        if 'DZ [U/min] Mean' in columns and 'Andruck [bar] Mean' in columns and 'vB [m/h] Mean' in columns:
                            if isinstance(DI(stats_data_rocks), pd.DataFrame):
                                stats_data_rocks = DI(stats_data_rocks)

                            elif DI(stats_data_rocks) is None:
                                st.error('**Please make sure that your units are in SI System. Please go back to the first part, and upload a new data set with the consistent units!**', icon='🛑')
                                unit_check = True
                            
                        if not unit_check:
                            stats_data_rocks.loc[:, 'Well Name'] = st.session_state.file_name
                            
                            st.table(data=stats_data_rocks)
                            st.caption(f'**Calculated Number of Intervals:** {counted_interval}')
                            st.caption(f'**Number of Intervals in Dataset:** {intervals}')
            
                            display_chart = True
                            st.markdown(f"""
                            <div class='stHeader'><center>Download the Resulted Datasets for <i>{file_name_without_extension}</i></center></div>
                            """, unsafe_allow_html=True)
                            col1, col2 = st.columns(2, gap='large')

                            prior_data_rocks_excel = Excel(prior_data_rocks, 'Prior Data')
                            stats_data_rocks_excel = Excel(stats_data_rocks, 'Interval Data')

                            st.session_state.prior_data_rocks = prior_data_rocks
                            st.session_state.stats_data_rocks = stats_data_rocks

                            with col1:
                                st.download_button(label='Download Prior Data',
                                                data=prior_data_rocks_excel,
                                                file_name=f'Prior data for {st.session_state.file_name}.xlsx')

                            with col2:
                                st.download_button(label='Download Interval Data',
                                                data=stats_data_rocks_excel,
                                                file_name=f'Interval data for {st.session_state.file_name}.xlsx')
                        
                    else:
                        st.error('**According to the algorithm, the dataset is found to be not convenient.**', icon='🛑')

            elif st.session_state.flag == 0:
                st.error('**Please make sure that your dataset features are in SI unit system!**', icon='🛑')

            elif st.session_state.flag == -1:
                st.warning('**Number of formation intervals cannot be less than 1!**', icon='💹')
            elif st.session_state.flag == -2:
                st.warning(
                    "**The following interval's start measurement cannot be less than previous formation's end boundary!**",
                    icon='💹')
            elif st.session_state.flag == -3:
                st.error('**Please make sure that your dataset features are in SI unit system!**', icon='🛑')
            else:
                st.error('**Something went wrong!**')


        else:
            st.warning('**Please press Start button to proceed!**', icon='💹')

        if display_chart:
            with st.form('Chart'):
                if 'display' not in st.session_state:
                    st.session_state.display = False

                st.form_submit_button('Display Plot', on_click=Display, type='primary')

                if st.session_state.display:
                    interval_depths, interval_times = Intervals.CalculateIntervals(st.session_state.prior_data_rocks,
                                                                                   columns, pipe_length, error_rate)[3:5]

                    plot_intervals(interval_depths, interval_times, st.session_state.prior_data_rocks)

    else:
        st.warning('**This dataset is empty. Probably something went wrong. Please repeat the calculations.', icon='💹')


if __name__ == '__main__':
    main()
