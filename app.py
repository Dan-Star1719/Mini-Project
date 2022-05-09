import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import requests

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

    name = st.text_input("Your name", key='name')
    drivers['fullname'] = drivers['forename'] + ' ' + drivers['surname']
    drivers[drivers['fullname']==name].reset_index()[['number','code','forename','surname','dob','nationality','url']]

    country = st.selectbox("Country", circuits["country"].unique())
    circuits[lambda x: x['country'] == country].reset_index()[['name','location','country','lat','lng','alt','url']]

    nationality = st.selectbox("Nationality", drivers["nationality"].unique())
    victories = results[lambda x: x['positionOrder']==1]
    victories_drivers = victories.merge(drivers, left_on='driverId', right_on='driverId')[:]
    number_of_victories = victories_drivers[lambda x: x['nationality'] == nationality].groupby("fullname")['raceId'].count()
    total_victories = pd.DataFrame({'driver': number_of_victories.index, 'victories':number_of_victories}).sort_values('victories', ascending =False)
    if not total_victories.empty:
        total_victories.reset_index()[['driver', 'victories']]
        fig, ax = plt.subplots()
        sns.barplot(y=total_victories['driver'], x=total_victories['victories'] , ax=ax)
        st.pyplot(fig)
    else:
        """Nobody"""


    nationalities =  st.multiselect("gtgtg", drivers["nationality"].unique())
    victories = results[lambda x: x['positionOrder'] == 1]
    victories_drivers = victories.merge(drivers, left_on='driverId', right_on='driverId')[:]
    number_of_victories = victories_drivers[lambda x: x['nationality'].isin(nationalities)].groupby("fullname")[
        'raceId'].count()
    total_victories = pd.DataFrame({'driver': number_of_victories.index, 'victories': number_of_victories}).sort_values(
        'victories', ascending=False)
    if not total_victories.empty:
        total_victories.reset_index()[['driver', 'victories']]
        fig, ax = plt.subplots()
        sns.barplot(y=total_victories['driver'], x=total_victories['victories'], ax=ax)
        st.pyplot(fig)
    else:
        """Nobody"""







    drivers

    constructors

    results

    constructor_standings[lambda x: x['wins'] > 0].merge(constructors, left_on='constructorId',
                                                         right_on='constructorId')[:].groupby('name')['wins'].count()









