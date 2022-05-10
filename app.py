import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import requests
import plotly.express as px

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
            .properties(height=500, width=500).interactive())
    fig2 = (alt.Chart(df).mark_point(size=50, filled=True)
            .encode(alt.X('q2', scale=alt.Scale(zero=False), axis=alt.Axis(title='Lap time')),
                    alt.Y('type2', scale=alt.Scale(zero=False), axis=alt.Axis(title='Session')),
                    color=alt.Color('fullname', legend=alt.Legend(title='The drivers')),
                    tooltip = [alt.Tooltip('fullname'), alt.Tooltip('q2')])
            .properties(height=500, width=500).interactive())
    fig3 = (alt.Chart(df).mark_point(size=50, filled=True)
            .encode(alt.X('q3', scale=alt.Scale(zero=False), axis=alt.Axis(title='Lap time')),
                    alt.Y('type3', scale=alt.Scale(zero=False), axis=alt.Axis(title='Session')),
                    color=alt.Color('fullname', legend=alt.Legend(title='The drivers')),
                    tooltip = [alt.Tooltip('fullname'), alt.Tooltip('q3')])
            .properties(height=500, width=500).interactive())
    points = (alt.Chart(df3).mark_point(size=150, filled=True, color='black')
            .encode(alt.X('time', scale=alt.Scale(zero=False), axis=alt.Axis(title='Lap time')),
                    alt.Y('session', scale=alt.Scale(zero=False), axis=alt.Axis(title='Session')),
                    tooltip = [alt.Tooltip('time'), alt.Tooltip('session')])
            .properties(height=500, width=500).interactive())

    st.altair_chart(fig1+fig2+fig3+points)




