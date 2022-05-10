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
    total_victories = (pd.DataFrame({'driver': number_of_victories.index, 'victories': number_of_victories})
        .sort_values('victories', ascending =False).reset_index()[['driver', 'victories']])
    if not total_victories.empty:
        total_victories
        fig, ax = plt.subplots()
        sns.barplot(y=total_victories['driver'], x=total_victories['victories'] , ax=ax,  palette='crest')
        st.pyplot(fig)
    else:
        """Nobody from this country has won any races. Thy to choose another nationality!"""


    """As you see, some contries have more race winners than others. You can choose the few nationalities below and 
    compare different countries in term of the number of victories of the drivers of different nationalities."""
    nationalities =  st.multiselect("Choose few nationalities:", drivers["nationality"].unique())
    number_of_victories_by_nations = (victories_by_drivers[lambda x: x['nationality']
                                      .isin(nationalities)].groupby("nationality")['raceId'].count())
    total_victories_by_nations = (pd.DataFrame({'nationality': number_of_victories_by_nations.reset_index()['nationality'],
                                               'victories': number_of_victories_by_nations.reset_index()['raceId']})
                                  .sort_values('victories', ascending=False))
    for element in nationalities:
        if not total_victories_by_nations['nationality'].isin([element]).any():
            total_victories_by_nations = (total_victories_by_nations
                                          .append({'nationality': element, 'victories': 0}, ignore_index=True))
    if not total_victories_by_nations.empty:
        total_victories_by_nations
        fig, ax = plt.subplots()
        sns.barplot(y=total_victories_by_nations['nationality'], x=total_victories_by_nations['victories'], ax=ax)
        st.pyplot(fig)
    else:
        """Nobody has been chosen yet."""


