import streamlit as st

st.set_page_config(
    page_title="Visualization",
    page_icon="🗺️"
)

st.title('Visualization')
st.divider()


def correlation_matrix_plot(data):
    import plotly.express as px

    corr_matrix = data.corr()

    fig = px.imshow(corr_matrix,
                    color_continuous_scale='RdBu',
                    labels=dict(color="Correlation"),
                    title="Correlation Matrix Plot")

    fig.update_layout(width=600, height=600)
    st.plotly_chart(fig)


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
    st.plotly_chart(fig)


def Select():
    st.session_state.select = True


def main():
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

                correlation_matrix_plot(st.session_state.dropped_data)

            elif visualization_type == 'Correlation Matrix Plot' and data_type == 'Use Stats Data':
                st.subheader(f'The Correlation Matrix Plot with Stats Data of {st.session_state.file_name}')

                correlation_matrix_plot(st.session_state.stats_table)


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
                        feature_investigation_plot(selected_columns_x, selected_columns_y, formation_option,
                                                   st.session_state.formation_water, st.session_state.dropped_data)


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

                    feature_investigation_plot(selected_columns_x, selected_columns_y, formation_option,
                                               st.session_state.formation_water, st.session_state.stats_table)


if __name__ == '__main__':
    main()
