from loading_data import * 
import pandas as pd 
import seaborn as sns 

def cleaning_data():    
    df = load_df()
    df = df.drop(["Land Area(Km2)","Latitude","Longitude",],axis=1)
    df['Access to electricity (% of population)'].fillna(df["Access to electricity (% of population)"].median(),inplace=True)
    df['Access to clean fuels for cooking'].fillna(df["Access to clean fuels for cooking"].median(),inplace=True)
    df['Financial flows to developing countries (US $)'].fillna(df["Financial flows to developing countries (US $)"].median(),inplace=True)
    df['Renewable energy share in the total final energy consumption (%)'].fillna(df["Renewable energy share in the total final energy consumption (%)"].median(),inplace=True)
    df['Electricity from fossil fuels (TWh)'].fillna(df["Electricity from fossil fuels (TWh)"].median(),inplace=True)
    df['Electricity from nuclear (TWh)'].fillna(df["Electricity from nuclear (TWh)"].median(),inplace=True)
    df['Electricity from renewables (TWh)'].fillna(df["Electricity from renewables (TWh)"].median(),inplace=True)
    df['Low-carbon electricity (% electricity)'].fillna(df["Low-carbon electricity (% electricity)"].median(),inplace=True)
    df['Primary energy consumption per capita (kWh/person)'].fillna(df["Primary energy consumption per capita (kWh/person)"].median(),inplace=True)
    df['Energy intensity level of primary energy (MJ/$2017 PPP GDP)'].fillna(df["Energy intensity level of primary energy (MJ/$2017 PPP GDP)"].median(),inplace=True)
    df['Value_co2_emissions_kt_by_country'].fillna(df["Value_co2_emissions_kt_by_country"].median(),inplace=True)
    df['Renewables (% equivalent primary energy)'].fillna(df["Renewables (% equivalent primary energy)"].median(),inplace=True)
    df['gdp_growth'].fillna(df["gdp_growth"].median(),inplace=True)
    df['gdp_per_capita'].fillna(df["gdp_per_capita"].median(),inplace=True)
    df['Renewable-electricity-generating-capacity-per-capita'].fillna(df["Renewable-electricity-generating-capacity-per-capita"].median(),inplace=True)
    return df 

