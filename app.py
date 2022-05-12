import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from plotnine import ggplot, aes, geom_point, geom_smooth

with st.echo(code_location='below'):
    st.write(f"### Formula 1 data analysis")

    @st.cache(allow_output_mutation=True)
    def get_data(data):
        return pd.read_csv(data)

    circuits = get_data("circuits.csv")
    constructor_results = get_data("constructor_results.csv")
    constructor_standings = get_data("constructor_standings.csv")
    constructors = get_data("constructors.csv")
    driver_standings = get_data("driver_standings.csv")
    drivers = get_data("drivers.csv")
    lap_times = get_data("lap_times.csv")
    pit_stops = get_data("pit_stops.csv")
    qualifying = get_data("qualifying.csv")
    races = get_data("races.csv")
    results= get_data("results.csv")
    seasons = get_data("seasons.csv")
    sprint_results = get_data("sprint_results.csv")
    status = get_data("status.csv")

    drivers['fullname'] = drivers['forename'] + ' ' + drivers['surname']

    """You can type the fullname of any driver who have ever participated in Formula 1 Gran Prix to read some 
    information anout him and watch the visualization of his performance."""
    """If you don't know any F1 drivers, you can choose any driver from the following list:\n
    Fernando Alonso, Michael Schumacher, Lewis Hamilton, Daniil Kvyat, Sebastian Vettel, Ayrton Senna"""

    name = st.text_input("Type the full name of the driver:", key='name')
    if name:
        results_by_driver = results.merge(drivers, left_on='driverId', right_on='driverId')[
                                lambda x: x['fullname'] == name][:]
        first_places = results_by_driver[lambda x: x['positionOrder'] == 1]['raceId'].count()
        second_places = results_by_driver[lambda x: x['positionOrder'] == 2]['raceId'].count()
        third_places = results_by_driver[lambda x: x['positionOrder'] == 3]['raceId'].count()
        finishes = results_by_driver['raceId'].count()
        other_places = finishes - first_places - second_places - third_places
        driver_results = pd.DataFrame({'Places': ['1st place', '2nd place', '3rd place', '4th or lower'],
                                       'Number': [first_places, second_places, third_places, other_places]})
        if not results_by_driver.empty:
            drivers[drivers['fullname']==name].reset_index()[['forename', 'surname', 'code', 'number', 'dob', 'nationality']]

            fig = px.pie(driver_results, values='Number', names='Places', title=f'The performance of {name}:',
                         color='Places',
                         color_discrete_map={'1st place': 'gold',
                                                             '2nd place': 'gray',
                                                             '3rd place': 'saddlebrown',
                                                             '4th or lower': 'whitesmoke'})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)
        else:
            """There are no drivers with such full name. Try again!"""

    """Here you can choose the country that held the F1 races and watch the information about the circuits located 
    in this country. In the diagram below, you can see how many races were held in each of the circuits."""

    country = st.selectbox("Choose the country", circuits["country"].unique())
    if country:
        circuits_in_country = circuits[lambda x: x['country'] == country].reset_index()[['name','location','country']]
        circuits_in_country

        races_by_circuits = (races.merge(circuits, left_on='circuitId', right_on='circuitId')
                             [lambda x: x['country']==country][:])
        number_of_races_by_circuits = races_by_circuits.groupby('name_y')['raceId'].count()
        number_of_races_by_circuits_df = (pd.DataFrame({'Track': number_of_races_by_circuits.index,
                                                        'Number of races': number_of_races_by_circuits}))
        fig = px.pie(number_of_races_by_circuits_df, values='Number of races', names='Track', title=f'The number of races on the tracks located in {country}:',
                     color='Track')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig)

    """As you have seen before, there are many drivers who won the races. But how many drivers have won the races from every country?"""

    """You can choose the nationality below and see how many drivers with the chosen nationality have won the races and 
    how many races they won. Note that some countries have no race winners."""

    nationality = st.selectbox("Choose the nationality:", drivers["nationality"].unique())
    victories_by_drivers = (results[lambda x: x['positionOrder']==1]
                                .merge(drivers, left_on='driverId', right_on='driverId')[:])
    number_of_victories = (victories_by_drivers[lambda x: x['nationality'] == nationality]
                           .groupby("fullname")['raceId'].count())
    total_victories = (pd.DataFrame({'Driver': number_of_victories.index, 'Number of victories': number_of_victories})
        .sort_values('Number of victories', ascending =False).reset_index()[['Driver', 'Number of victories']])
    if not total_victories.empty:
        fig, ax = plt.subplots()
        chart = sns.barplot(y=total_victories['Driver'], x=total_victories['Number of victories'] , ax=ax,  palette='crest')
        chart.bar_label(chart.containers[0], fontsize=8.5, color='black')
        st.pyplot(fig)
    else:
        """Nobody from this country has won any races. Thy to choose another nationality!"""


    """As you see, some contries have more race winners than others. You can choose the few nationalities below and 
    compare different countries in term of the number of victories of the drivers of different nationalities."""
    nationalities =  st.multiselect("Choose few nationalities:", drivers["nationality"].unique())
    number_of_victories_by_nations = (victories_by_drivers[lambda x: x['nationality']
                                      .isin(nationalities)].groupby("nationality")['raceId'].count())
    total_victories_by_nations = (pd.DataFrame({'Nationality': number_of_victories_by_nations.reset_index()['nationality'],
                                               'Number of victories': number_of_victories_by_nations.reset_index()['raceId']})
                                  .sort_values('Number of victories', ascending=False))
    for element in nationalities:
        if not total_victories_by_nations['Nationality'].isin([element]).any():
            total_victories_by_nations = (total_victories_by_nations
                                          .append({'Nationality': element, 'Number of victories': 0}, ignore_index=True))
    if not total_victories_by_nations.empty:
        fig, ax = plt.subplots()
        chart = sns.barplot(y=total_victories_by_nations['Nationality'], x=total_victories_by_nations['Number of victories'], ax=ax)
        chart.bar_label(chart.containers[0], fontsize=8.5, color='black')
        st.pyplot(fig)
    else:
        """Nobody has been chosen yet."""

    """You can find three kinds of criteria that can be used to compare the drivers. 
    You can choose one of them and look at top-20 drivers in terms of the chosen criteria."""

    choice = st.radio('Choose:', ['Number of the races', 'Number of the podiums', 'Number of the victories'])

    total_results = results.merge(drivers, left_on='driverId', right_on='driverId')[:]
    all_drivers_races = total_results.groupby('fullname')['raceId'].count()
    all_drivers_races_df = (pd.DataFrame({'Driver': all_drivers_races.index, 'Number of races': all_drivers_races})
                            .sort_values('Number of races', ascending=False).iloc[0:20])

    all_drivers_victories = total_results[lambda x: x['positionOrder']==1].groupby('fullname')['raceId'].count()
    all_drivers_victories_df = (pd.DataFrame({'Driver': all_drivers_victories.index, 'Number of victories': all_drivers_victories})
                                .sort_values('Number of victories', ascending=False).iloc[0:20])

    all_drivers_2nd = total_results[lambda x: x['positionOrder'] == 2].groupby('fullname')['raceId'].count()
    all_drivers_3rd = total_results[lambda x: x['positionOrder'] == 3].groupby('fullname')['raceId'].count()
    all_drivers_podiums = all_drivers_victories + all_drivers_2nd + all_drivers_3rd
    all_drivers_podiums_df = (pd.DataFrame({'Driver': all_drivers_podiums.index, 'Number of podiums': all_drivers_podiums})
        .sort_values('Number of podiums', ascending=False).iloc[0:20])

    if choice == 'Number of the races':
        fig, ax = plt.subplots()
        chart1 = sns.barplot(y=all_drivers_races_df['Driver'], x=all_drivers_races_df['Number of races'],
                            ax=ax,  palette="viridis")
        chart1.bar_label(chart1.containers[0], fontsize=8.5, color='black')
        st.pyplot(fig)

    if choice == 'Number of the podiums':
        fig, ax = plt.subplots()
        chart2 = sns.barplot(y=all_drivers_podiums_df['Driver'], x=all_drivers_podiums_df['Number of podiums'],
                            ax=ax,  palette='flare')
        chart2.bar_label(chart2.containers[0], fontsize=8.5, color='black')
        st.pyplot(fig)

    if choice == 'Number of the victories':
        fig, ax = plt.subplots()
        chart3 = sns.barplot(y=all_drivers_victories_df['Driver'], x=all_drivers_victories_df['Number of victories'],
                            ax=ax,  palette='rocket')
        chart3.bar_label(chart3.containers[0], fontsize=8.5, color='black')
        st.pyplot(fig)

    agree = st.checkbox('Show:')

    if agree:
        data = (results[lambda x: x['positionOrder'] == 1]
                .merge(constructors, left_on='constructorId', right_on='constructorId')
                .merge(races[['year', 'raceId']], left_on='raceId', right_on='raceId')[['name', 'positionOrder', 'year']].sort_values('year'))
        data1 = data.pivot_table(values='positionOrder', index='name',  columns='year', aggfunc='count').fillna(0)

        data1
        data2 = np.cumsum(data1, axis=1).unstack().reset_index()

        data2['victories'] = data2[0]

        fig = px.bar(data2, x='victories', y="name", color="name",
                     animation_frame="year")
        st.plotly_chart(fig)

    """Some analytics"""

    results_by_drivers = results.merge(drivers, left_on='driverId', right_on='driverId')
    first_places1 = results_by_drivers[lambda x: x['positionOrder'] == 1].groupby('fullname')['raceId'].count()
    second_places1 = results_by_drivers[lambda x: x['positionOrder'] == 2].groupby('fullname')['raceId'].count()
    third_places1 = results_by_drivers[lambda x: x['positionOrder'] == 3].groupby('fullname')['raceId'].count()
    finishes = results_by_drivers.groupby('fullname')['raceId'].count()
    podiums1 = first_places1 + second_places1 + third_places1
    total_driver_results = pd.DataFrame()
    total_driver_results['finishes'] = finishes
    total_driver_results['victories'] = first_places1
    total_driver_results['podiums'] = podiums1

    total_driver_results = total_driver_results.fillna(0)

    plot1 = (ggplot(total_driver_results, aes(x='finishes', y='victories'))
             + geom_point(aes(color='finishes')) + geom_smooth(method='lm'))
    st.pyplot(ggplot.draw(plot1))

    plot2 = (ggplot(total_driver_results, aes(x='finishes', y='podiums'))
             + geom_point(aes(color='finishes')) + geom_smooth(method='lm'))
    st.pyplot(ggplot.draw(plot2))

    plot3 = (ggplot(total_driver_results[lambda x: x['podiums'] > 0], aes(x='podiums', y='victories'))
             + geom_point(aes(color='finishes')) + geom_smooth(method='lm'))
    st.pyplot(ggplot.draw(plot3))


    """Some race analytics"""

    a = st.slider('Choose the year:', 2006, 2021)
    races_in_this_year = races[lambda x: x['year'] == a]
    if a:
        gran_prix = st.selectbox('Choose the Gran Prix:', races_in_this_year['name'].unique())


    df1 = qualifying.merge(races[lambda x: x['year'] == a], left_on='raceId', right_on='raceId')
    df = df1[lambda x: x['name'] == gran_prix].merge(drivers, left_on='driverId', right_on='driverId')
    chosen_driver = st.selectbox('Choose the driver:', df['fullname'].unique())

    #number_of_drivers = len(df.index)
    ### https://question-it.com/questions/1146283/pandas-preobrazovanie-vremeni-v-sekundy-dlja-vseh-znachenij-v-stolbtse
    df['q1'] = (df[df['q1'].str[0].isin(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])]
                ['q1'].map(lambda x: sum(x * float(t) for x, t in zip([60.0, 1.0], x.split(':')))))
    df['q2'] = (df[df['q2'].str[0].isin(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])]
                ['q2'].map(lambda x: sum(x * float(t) for x, t in zip([60.0, 1.0], x.split(':')))))
    df['q3'] = (df[df['q3'].str[0].isin(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])]
                ['q3'].map(lambda x: sum(x * float(t) for x, t in zip([60.0, 1.0], x.split(':')))))
    df['type1'] = 'Q1'
    df['type2'] = 'Q2'
    df['type3'] = 'Q3'



    df2 = df[lambda x: x['fullname']==chosen_driver]
    df3 = pd.DataFrame({'time': [df2.iloc[0]['q1'], df2.iloc[0]['q2'], df2.iloc[0]['q3']], 'session': ['Q1', 'Q2', 'Q3']})

    fig1 = (alt.Chart(df).mark_point(size=50, filled=True)
            .encode(alt.X('q1', scale=alt.Scale(zero=False), axis=alt.Axis(title='Lap time')),
                    alt.Y('type1', scale=alt.Scale(zero=False), axis=alt.Axis(title='Session')),
                    color=alt.Color('fullname', legend=alt.Legend(title='The drivers')),
                    tooltip = [alt.Tooltip('fullname'), alt.Tooltip('q1')])
            .properties(height=500, width=500, title='The qualification').interactive())
    fig2 = (alt.Chart(df).mark_point(size=50, filled=True)
            .encode(alt.X('q2', scale=alt.Scale(zero=False), axis=alt.Axis(title='Lap time')),
                    alt.Y('type2', scale=alt.Scale(zero=False), axis=alt.Axis(title='Session')),
                    color=alt.Color('fullname', legend=alt.Legend(title='The drivers')),
                    tooltip = [alt.Tooltip('fullname'), alt.Tooltip('q2')])
            .properties(height=500, width=500, title='The qualification').interactive())
    fig3 = (alt.Chart(df).mark_point(size=50, filled=True)
            .encode(alt.X('q3', scale=alt.Scale(zero=False), axis=alt.Axis(title='Lap time')),
                    alt.Y('type3', scale=alt.Scale(zero=False), axis=alt.Axis(title='Session')),
                    color=alt.Color('fullname', legend=alt.Legend(title='The drivers')),
                    tooltip = [alt.Tooltip('fullname'), alt.Tooltip('q3')])
            .properties(height=500, width=500, title='The qualification').interactive())
    points = (alt.Chart(df3).mark_point(size=150, filled=True, color='black')
            .encode(alt.X('time', scale=alt.Scale(zero=False), axis=alt.Axis(title='Lap time')),
                    alt.Y('session', scale=alt.Scale(zero=False), axis=alt.Axis(title='Session')),
                    tooltip = [alt.Tooltip('time'), alt.Tooltip('session')])
            .properties(height=500, width=500, title='The qualification').interactive())

    st.altair_chart(fig1+fig2+fig3+points)

    question = st.radio('Do you want to watch the graph for another race?', ['No, I want to watch this one',
                                                                             'Yes, I want to choose another one'])

    if question == 'Yes, I want to choose another one':
        a = st.slider('Choose the year:', 1996, 2021)
        races_in_this_year = races[lambda x: x['year'] == a]
        if a:
            gran_prix = st.selectbox('Choose the Gran Prix:', races_in_this_year['name'].unique())

    if question:
        prerace_df = lap_times.merge(races, left_on='raceId', right_on='raceId')[lambda x: x['year'] == a]
        race_df = prerace_df[lambda x: x['name'] == gran_prix].merge(drivers, left_on='driverId', right_on='driverId')[
            ['fullname', 'lap', 'position']]

        df_for_start1 = (results.merge(drivers, left_on='driverId', right_on='driverId')
            .merge(races, left_on='raceId', right_on='raceId')[lambda x: x['year'] == a])
        df_for_start2 = df_for_start1[lambda x: x['name'] == gran_prix][['fullname', 'grid']]

        for i in range(len(df_for_start2.index)):
            if df_for_start2.iloc[i]['grid'] > 0:
                race_df = race_df.append({'fullname': df_for_start2.iloc[i]['fullname'], 'lap': 0,
                                          'position': df_for_start2.iloc[i]['grid']}, ignore_index=True)

        selection = alt.selection_multi(fields=['fullname'], bind='legend')

        fig = alt.Chart(race_df).mark_line(point=True).encode(
            x=alt.X("lap", scale=alt.Scale(zero=False), title="Lap"),
            y=alt.Y("position", scale=alt.Scale(zero=False), sort='descending', axis=alt.Axis(title='Position')),
            color=alt.Color("fullname"), tooltip=[alt.Tooltip('fullname'), alt.Tooltip('lap'), alt.Tooltip('position')],
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
        ).properties(
            title="The race",
            width=600,
            height=600,
        ).add_selection(selection).interactive()

        st.altair_chart(fig)

        new_race_df1 = (race_df.pivot_table(values='lap', index='fullname', columns='position', aggfunc='count')
                       .fillna(0))
        new_race_df2 = (race_df.pivot_table(values='position', index='fullname', aggfunc='mean')
                       .fillna(0))
        new_race_df1['average'] = new_race_df2['position']

        new_race_df1 = new_race_df1.sort_values(by='average', ascending=True)

        fig, ax = plt.subplots()
        chart2 = sns.heatmap(data=new_race_df1.drop('average', axis=1),
                             ax=ax, cbar=True, cmap='Oranges')
        st.pyplot(fig)





        lap_times_gp = (lap_times.merge(races, left_on='raceId', right_on='raceId')[lambda x: x['year'] == a]
            .merge(drivers, left_on='driverId', right_on='driverId')[lambda x: x['name'] == gran_prix])

        the_driver = st.selectbox('Choose the driver:', lap_times_gp['fullname'].unique())

        driver_laps = lap_times_gp[lambda x: x['fullname'] == the_driver][
            ['fullname', 'lap', 'position', 'milliseconds']]

        plot4 = (ggplot(driver_laps, aes(x='lap', y='milliseconds'))
                 + geom_point(aes(color='milliseconds')) + geom_smooth())
        st.pyplot(ggplot.draw(plot4))

    """The championship analysis"""

    b = st.slider('Choose the year', 1950, 2021)

    table_of_standings = driver_standings.merge(races, left_on='raceId', right_on='raceId')[lambda x: x['year']==b]\
        .merge(drivers, left_on='driverId', right_on='driverId')[['fullname', 'round', 'position', 'points']]

    selection1 = alt.selection_multi(fields=['fullname'], bind='legend')

    graph1 = alt.Chart(table_of_standings).mark_line(point=True).encode(
        x=alt.X("round", scale=alt.Scale(zero=False), title="Round"),
        y=alt.Y("position", scale=alt.Scale(zero=False), sort='descending', axis=alt.Axis(title='Position')),
        color=alt.Color("fullname"), tooltip=[alt.Tooltip('fullname'), alt.Tooltip('position'), alt.Tooltip('points')],
        opacity=alt.condition(selection1, alt.value(1), alt.value(0.2))
    ).properties(
        title="The championship battle",
        width=600,
        height=600,
    ).add_selection(selection1).interactive()

    for i in range(len(table_of_standings['fullname'].unique())):
        table_of_standings = table_of_standings.append({'fullname': table_of_standings['fullname'].unique()[i], 'round': 0,
                                      'position': 0, 'points': 0}, ignore_index=True)

    graph2 = alt.Chart(table_of_standings).mark_line(point=True).encode(
        x=alt.X("round", scale=alt.Scale(zero=False), title="Round"),
        y=alt.Y("points", scale=alt.Scale(zero=False), sort='ascending', axis=alt.Axis(title='Points')),
        color=alt.Color("fullname"), tooltip=[alt.Tooltip('fullname'), alt.Tooltip('position'), alt.Tooltip('points')],
        opacity=alt.condition(selection1, alt.value(1), alt.value(0.2))
    ).properties(
        title="The championship battle",
        width=600,
        height=600,
    ).add_selection(selection1).interactive()

    st.altair_chart(graph1 & graph2)






















