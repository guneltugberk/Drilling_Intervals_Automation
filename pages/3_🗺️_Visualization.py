import streamlit as st

st.set_page_config(
    page_title="Visualization",
    page_icon="🗺️"
    )


@st.cache_data(ttl=3600)
def correlation_correction(copy_df):
    import pandas as pd

    # Make a copy of the DataFrame
    df = copy_df.copy()

    rock_types = df['Formation'].unique()

    mapping = {formation: i for i, formation in enumerate(rock_types)}
    df_numeric = pd.DataFrame(columns=rock_types)

    if 'None' in df_numeric.columns:
        df_numeric.drop('None', axis=1, inplace=True)

    for rock_type, numeric_value in mapping.items():
        df_numeric[rock_type] = [numeric_value]

    # Modify the copied DataFrame
    df['Formation'] = df['Formation'].replace(mapping)

    return df, df_numeric


@st.cache_resource(ttl=3600)
def correlation_matrix_plot(data_plot):
    import plotly.express as px

    corr_matrix = data_plot.corr(numeric_only=True)

    fig = px.imshow(corr_matrix,
                    color_continuous_scale='RdBu',
                    labels=dict(color="Correlation"),
                    title="Correlation Matrix Plot")

    fig.update_layout(width=600, height=600)

    return fig


@st.cache_resource(ttl=3600)
def feature_investigation_plot(selected_columns_x, selected_columns_y, formation_option, water_depth, data_feature, add_linear_curve, control_formation, control_water):
    import plotly.express as px
    import plotly.graph_objects as go
    import statsmodels.api as sm


    if control_formation:
        if formation_option == 'Include Formations' and add_linear_curve:

            fig = px.scatter(data_feature, x=selected_columns_x, y=selected_columns_y, color='Formation',
                            hover_name='Formation',
                            title='Feature Investigation with Formations', color_discrete_sequence=px.colors.qualitative.G10, trendline='ols', trendline_color_override='red', trendline_scope='overall')
            
        elif formation_option == 'Include Formations':
            fig = px.scatter(data_feature, x=selected_columns_x, y=selected_columns_y, color='Formation',
                            hover_name='Formation',
                            title='Feature Investigation with Formations', color_discrete_sequence=px.colors.qualitative.G10)
            
        elif add_linear_curve:
            fig = px.scatter(data_feature, x=selected_columns_x, y=selected_columns_y,
                            title='Feature Investigation without Formations', trendline='ols', trendline_color_override='red', trendline_scope='overall')
            
    elif not control_formation:    
        if add_linear_curve:
            fig = px.scatter(data_feature, x=selected_columns_x, y=selected_columns_y,
                            title='Feature Investigation without Formations', trendline='ols', trendline_color_override='red', trendline_scope='overall')
            
        else:
            fig = px.scatter(data_feature, x=selected_columns_x, y=selected_columns_y,
                            title='Feature Investigation without Formations')
    
    
    if control_water:
        if selected_columns_y == 'Teufe [m] Mean' or selected_columns_y == 'Teufe [m]':
            fig.update_yaxes(autorange="reversed")
            fig.add_trace(go.Scatter(x=[None, None], y=[None, None], mode='lines+text',
                                    line=dict(color='blue', width=2, dash='dash'),
                                    text=['Formation Water'],
                                    textposition='top center',
                                    name='Formation Water Depth', 
                                    opacity=0.5))

            fig.add_hline(y=water_depth, line=dict(color='blue', width=1, dash='dash'))

    else:
        if selected_columns_y == 'Teufe [m] Mean' or selected_columns_y == 'Teufe [m]':
            fig.update_yaxes(autorange="reversed")

    fig.update_layout(showlegend=True)            
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    fig.update_yaxes(range=[data_feature[selected_columns_y].min(), data_feature[selected_columns_y].max()+10])
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(height=600, hovermode='closest')

    return st.plotly_chart(fig)


def Select():
    st.session_state.select = True


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
    <div class='stTitle'>Visualization</div>
    """, unsafe_allow_html=True)
    st.divider()
    
    if 'stats_data_rocks' not in st.session_state:
        st.session_state.stats_data_rocks = None

    if 'prior_data_rocks' not in st.session_state:
        st.session_state.prior_data_rocks = None
    
    if 'formation_info' not in st.session_state:
        st.session_state.formation_info = None
    
    if 'water_info' not in st.session_state:
        st.session_state.water_info = None

    if st.session_state.prior_data_rocks is None and st.session_state.stats_data_rocks is None and st.session_state.water_info is None and st.session_state.formation_info is None:
        st.warning('**To be able to proceed the calculations, please complete the step one and two**',
                   icon='❌')
    elif st.session_state.prior_data_rocks is not None and st.session_state.stats_data_rocks is None:
        st.warning('**To be able to proceed the calculations, please complete the step two:** *Intervals Automation*',
                   icon='📐')

    elif st.session_state.prior_data_rocks is not None and st.session_state.stats_data_rocks is not None and st.session_state.water_info is not None and st.session_state.formation_info is not None:
        # Add a slider to select the visualization type
        visualization_type = st.sidebar.selectbox("**Select Visualization Type**",
                                                  ['Correlation Matrix Plot', 'Feature Investigation'])

        # Add radio buttons to select the data type
        data_type = st.sidebar.radio("**Select Data Type**", ['Use Prior Data', 'Use Stats Data'])

        st.sidebar.button('Select', on_click=Select)

        if 'select' not in st.session_state:
            st.session_state.select = False

        if st.session_state.select:
            # Perform actions based on the selected options
            if visualization_type == 'Correlation Matrix Plot' and data_type == 'Use Prior Data':
                st.markdown(f"""
                <div class='stHeader'>The Correlation Matrix Plot with Prior Data of <i>{st.session_state.file_name}</i></div>
                """, unsafe_allow_html=True)

                if st.session_state.formation_info:
                    correct_data = correlation_correction(st.session_state.prior_data_rocks)[0]
                    numeric_data = correlation_correction(st.session_state.prior_data_rocks)[1]

                    figure = correlation_matrix_plot(correct_data)
                    st.plotly_chart(figure)

                    st.markdown("""
                    <div class='stHeader'>Numerical Representation of Formations</div>
                    """, unsafe_allow_html=True)

                    st.table(data=numeric_data)

                elif not st.session_state.formation_info:
                    figure = correlation_matrix_plot(st.session_state.prior_data_rocks)
                    st.plotly_chart(figure)

            elif visualization_type == 'Correlation Matrix Plot' and data_type == 'Use Stats Data':
                st.markdown(f"""
                <div class='stHeader'>The Correlation Matrix Plot with Stats Data of <i>{st.session_state.file_name}</i></div>
                """, unsafe_allow_html=True)
                
                if st.session_state.formation_info:
                    correct_data = correlation_correction(st.session_state.stats_data_rocks)[0]
                    numeric_data = correlation_correction(st.session_state.stats_data_rocks)[1]

                    figure = correlation_matrix_plot(correct_data)
                    st.plotly_chart(figure)

                    st.markdown("""
                    <div class='stHeader'>Numerical Representation of Formations</div>
                    """, unsafe_allow_html=True)

                    st.table(data=numeric_data)
                
                elif not st.session_state.formation_info:
                    figure = correlation_matrix_plot(st.session_state.stats_data_rocks)
                    st.plotly_chart(figure)

            elif visualization_type == 'Feature Investigation' and data_type == 'Use Prior Data':

                st.markdown(f"""
                <div class='stHeader'>The Feature Investigation with Prior Data <i>{st.session_state.file_name}</i></div>
                """, unsafe_allow_html=True)
                st.info('X-axis and y-axis entry arguments accept only one feature at a time.')

                selected_columns_x = st.selectbox("**Select x-axis**", st.session_state.prior_data_rocks.columns)
                selected_columns_y = st.selectbox("**Select y-axis**", st.session_state.prior_data_rocks.columns)

                if selected_columns_y == 'Teufe [m]':
                    is_linear = st.checkbox('**Would you like to add a linear fit?**')

                else:
                    is_linear = False

                if st.session_state.formation_info:
                    formation_option = st.radio("**Formation Option**", ['Include Formations', 'Exclude Formations'])

                else:
                    formation_option = None

                if st.session_state.water_info:
                    if 'formation_water' not in st.session_state:
                        st.session_state.formation_water = None

                    if st.session_state.formation_water is None:
                        st.warning('**Please go back into previous page to specify Formation Water Depth**')

                    display_plot = st.button('Confirm Selection')

                    if display_plot:
                        if 'Formation' in selected_columns_x or 'Formation' in selected_columns_y:
                            st.warning('You cannot plot a categorical variable on a graph! Please re-select another feature', icon='💹')
                            
                        else:
                            if st.session_state.formation_water:
                                feature_investigation_plot(selected_columns_x, selected_columns_y, formation_option,
                                                        st.session_state.formation_water, st.session_state.prior_data_rocks, is_linear, st.session_state.formation_info, st.session_state.water_info)
                
                elif not st.session_state.water_info:
                    display_plot = st.button('Confirm Selection')

                    if display_plot:
                        feature_investigation_plot(selected_columns_x, selected_columns_y, formation_option,
                                                            st.session_state.formation_water, st.session_state.prior_data_rocks, is_linear, st.session_state.formation_info, st.session_state.water_info)


            elif visualization_type == 'Feature Investigation' and data_type == 'Use Stats Data':

                st.markdown(f"""
                <div class='stHeader'>The Feature Investigation with Stats Data of <i>{st.session_state.file_name}</i></div>
                """, unsafe_allow_html=True)
                st.info('X-axis and y-axis entry arguments accept only one feature at a time.')

                selected_columns_x = st.selectbox("**Select x-axis**", st.session_state.stats_data_rocks.columns)
                selected_columns_y = st.selectbox("**Select y-axis**", st.session_state.stats_data_rocks.columns)

                if selected_columns_y == 'Teufe [m] Mean':
                    is_linear = st.checkbox('**Would you like to add a linear fit?**')

                else:
                    is_linear = False

                if st.session_state.formation_info:
                    formation_option = st.radio("**Formation Option**", ['Include Formations', 'Exclude Formations'])

                else:
                    formation_option = None
                
                if st.session_state.water_info:
                    if 'formation_water' not in st.session_state:
                        st.session_state.formation_water = None

                    if st.session_state.formation_water is None:
                        st.warning('**Please go back into previous page to specify Formation Water Depth**')

                    display_plot = st.button('Confirm Selection')

                    if display_plot:
                        if 'Formation' in selected_columns_x or 'Formation' in selected_columns_y:
                            st.warning('You cannot plot a categorical variable on a graph! Please re-select another feature', icon='💹')
                            
                        else:
                            if st.session_state.formation_water:
                                feature_investigation_plot(selected_columns_x, selected_columns_y, formation_option,
                                                        st.session_state.formation_water, st.session_state.stats_data_rocks, is_linear, st.session_state.formation_info, st.session_state.water_info)
                
                elif not st.session_state.water_info:
                    display_plot = st.button('Confirm Selection')

                    if display_plot:
                        feature_investigation_plot(selected_columns_x, selected_columns_y, formation_option,
                                                            st.session_state.formation_water, st.session_state.stats_data_rocks, is_linear, st.session_state.formation_info, st.session_state.water_info)


if __name__ == '__main__':
    main()
