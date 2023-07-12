import streamlit as st

st.set_page_config(
    page_title="Visualization",
    page_icon="🗺️"
)

st.title('Visualization')
st.divider()


@st.cache_resource(ttl=3600)
def correlation_matrix_plot(data):
    import plotly.express as px

    corr_matrix = data.corr(numeric_only=True)

    fig = px.imshow(corr_matrix,
                    color_continuous_scale='RdBu',
                    labels=dict(color="Correlation"),
                    title="Correlation Matrix Plot")

    fig.update_layout(width=600, height=600)

    print(corr_matrix)

    return fig


@st.cache_resource(ttl=3600)
def feature_investigation_plot(selected_columns_x, selected_columns_y, formation_option, water_depth, data):
    import plotly.express as px
    import plotly.graph_objects as go

    if formation_option == 'Include Formations':

        fig = px.scatter(data, x=selected_columns_x, y=selected_columns_y, color='Formation',
                         hover_name='Formation',
                         title='Feature Investigation with Formations')
    else:
        fig = px.scatter(data, x=selected_columns_x, y=selected_columns_y,
                         title='Feature Investigation without Formations')

    if selected_columns_y == 'Teufe [m] Mean' or selected_columns_y == 'Teufe [m]':
        fig.update_yaxes(autorange="reversed")
        fig.add_trace(go.Scatter(x=[None, None], y=[None, None], mode='lines+text',
                                 line=dict(color='blue', width=2, dash='dash'),
                                 text=['Formation Water'],
                                 textposition='top center',
                                 name='Formation Water'))

        fig.add_hline(y=water_depth, line=dict(color='blue', width=1, dash='dash'))

        fig.update_layout(showlegend=True)

    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(height=600, hovermode='closest')

    return fig


def Select():
    st.session_state.select = True


def main():
    import pandas as pd
    
    if 'stats_table' not in st.session_state:
        st.session_state.stats_table = None

    if 'dropped_data' not in st.session_state:
        st.session_state.dropped_data = None

    if st.session_state.dropped_data is None and st.session_state.stats_table is None:
        st.warning('**To be able to proceed the calculations, please complete the step one and two**',
                   icon='❌')
    elif st.session_state.dropped_data is not None and st.session_state.stats_table is None:
        st.warning('**To be able to proceed the calculations, please complete the step two:** *Intervals Automation*',
                   icon='📐')

    elif st.session_state.dropped_data is not None and st.session_state.stats_table is not None:
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
                st.subheader(f'The Correlation Matrix Plot with Prior Data of {st.session_state.file_name}')
                rock_types = st.session_state.dropped_data['Formation'].unique()

                mapping_dropped = {formation: i for i, formation in enumerate(rock_types)}
                df_numeric_dropped = pd.DataFrame(columns=rock_types)

                copy_df_dropped = st.session_state.dropped_data.copy()
            
                if 'None' in df_numeric_dropped.columns:
                    df_numeric_dropped.drop('None', axis=1, inplace=True)
            
                for rock_type, numeric_value in mapping_dropped.items():
                    df_numeric_dropped[rock_type] = [numeric_value]
            
                copy_df_dropped['Formation'] = copy_df_dropped['Formation'].replace(mapping_dropped)

                if 'copy_df_dropped' not in st.session_state:
                    st.session_state.copy_df_dropped = copy_df_dropped

                if 'df_numeric_dropped' not in st.session_state:
                    st.session_state.df_numeric_dropped = df_numeric_dropped
                    
                figure = correlation_matrix_plot(st.session_state.copy_df_dropped)
                st.plotly_chart(figure)

                st.table(data=st.session_state.df_numeric_dropped)

            elif visualization_type == 'Correlation Matrix Plot' and data_type == 'Use Stats Data':
                st.subheader(f'The Correlation Matrix Plot with Stats Data of {st.session_state.file_name}')
                rock_types = st.session_state.stats_table['Formation'].unique()

                mapping_stats = {formation: i for i, formation in enumerate(rock_types)}
                df_numeric_stats = pd.DataFrame(columns=rock_types)

                copy_df_stats = st.session_state.stats_table.copy()
            
                if 'None' in df_numeric_stats.columns:
                    df_numeric_stats.drop('None', axis=1, inplace=True)
            
                for rock_type, numeric_value in mapping_stats.items():
                    df_numeric_stats[rock_type] = [numeric_value]
            
                copy_df_stats['Formation'] = copy_df_stats['Formation'].replace(mapping_stats)

                if 'copy_df_stats' not in st.session_state:
                    st.session_state.copy_df_stats = copy_df_stats

                if 'df_numeric_stats' not in st.session_state:
                    st.session_state.df_numeric_stats = df_numeric_stats
                    
                figure = correlation_matrix_plot(st.session_state.copy_df_stats)
                st.plotly_chart(figure)

                st.table(data=st.session_state.df_numeric_stats)
                st.write(st.session_state.copy_df_stats.columns)


            elif visualization_type == 'Feature Investigation' and data_type == 'Use Prior Data':

                st.subheader(f'The Feature Investigation with Prior Data {st.session_state.file_name}')
                st.info('X-axis and y-axis entry arguments accept only one feature at a time.')

                selected_columns_x = st.selectbox("**Select x-axis**", st.session_state.dropped_data.columns)
                selected_columns_y = st.selectbox("**Select y-axis**", st.session_state.dropped_data.columns)

                formation_option = st.radio("**Formation Option**", ['Include Formations', 'Exclude Formations'])
                print(st.session_state.dropped_data.info())

                if 'formation_water' not in st.session_state:
                    st.session_state.formation_water = None

                if st.session_state.formation_water is None:
                    st.warning('**Please go back into previous page to specify Formation Water Depth**')

                if st.button('Confirm Selection'):

                    if st.session_state.formation_water:
                        figure = feature_investigation_plot(selected_columns_x, selected_columns_y, formation_option,
                                                   st.session_state.formation_water, st.session_state.dropped_data)

                        st.plotly_chart(figure)


            elif visualization_type == 'Feature Investigation' and data_type == 'Use Stats Data':

                st.subheader(f'The Feature Investigation with Stats Data of {st.session_state.file_name}')
                st.info('X-axis and y-axis entry arguments accept only one feature at a time.')

                selected_columns_x = st.selectbox("**Select x-axis**", st.session_state.stats_table.columns)
                selected_columns_y = st.selectbox("**Select y-axis**", st.session_state.stats_table.columns)

                formation_option = st.radio("**Formation Option**", ['Include Formations', 'Exclude Formations'])

                if 'formation_water' not in st.session_state:
                    st.session_state.formation_water = None

                if st.session_state.formation_water is None:
                    st.warning('**Please go back into previous page to specify Formation Water Depth**')

                if st.button('Confirm Selection'):

                    figure = feature_investigation_plot(selected_columns_x, selected_columns_y, formation_option,
                                               st.session_state.formation_water, st.session_state.stats_table)

                    st.plotly_chart(figure)


if __name__ == '__main__':
    main()
