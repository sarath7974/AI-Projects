"""
Table of Content :

(1) Data Preprocessing
(2) Covid-19 Exploratory Data Analysis and Visualisation
  Covid-19 World Wide :
    - Covid-19 spread Analysis
    - Covid-19 Testing Analysis
    - Covid-19 Vaccination Analysis
    - Covid-19 Stringency Progression across Countries
    
  Covid-19 Continent Data :
    - Most Affected Continents
  
  Covid-19 Country Data :
    - Preprocess Country Data
    - Most Affected Countries
    - Testing Analysis in Countries
    - Vaccination of Countries
    - Sorting Counties Based on various Factors affecting Covid-19

  Covid-19 India :
    - Covid-19 Spread Analysis in India
    - Covid-19 Vaccination in India
    - Covid-19 Testing in India
    - Comparison between India and United States
"""



#Import Packages
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#Load the data
dataset=pd.read_csv("owid-covid-data.csv")
dataset.head()
print("SHAPE OF DATA :",dataset.shape)

print(dataset.columns)

dataset.describe()

dataset.info()

#Data Preprocessing
# Delete Smoothed values of cases and Deaths
delete_columns=["new_cases_smoothed","new_deaths_smoothed","total_cases_per_million",
                "new_cases_smoothed_per_million","total_deaths_per_million","new_deaths_smoothed_per_million",
                "reproduction_rate","weekly_icu_admissions_per_million","weekly_hosp_admissions_per_million",
                "new_tests_smoothed","new_tests_smoothed_per_thousand","new_vaccinations_smoothed",
                "new_vaccinations_smoothed_per_million","excess_mortality_cumulative"]
dataset=dataset.drop(delete_columns,axis=1)
print("SHAPE OF DATA :",dataset.shape)

data = dataset.copy()    #make a copy of dataset
# Remove non country items in Location Column   
data.drop(data[data["location"] == "World"].index,inplace=True)          #remove rows that have location as world 
data.drop(data[data["location"] == "European Union"].index,inplace=True) #remove rows that have location as Union 
data.drop(data[data["location"] == "International"].index,inplace=True)   #remove rows that have location as International 
# Remove Countries with null population
data.drop(data[data["population"].isna()].index,inplace=True)
continent=data["continent"].unique()
for i in continent:                                      #remove rows that have location as continent name
    data=data.drop(data[data["location"] == i].index)
data.reset_index(drop=True,inplace=True)
print("Shape of Data :",data.shape)
data.head()

#Covid-19 EDA & Visualisation
#World Data
world_data = data.groupby(["date"])["total_cases","new_cases","total_deaths","new_deaths",
                                       "icu_patients","new_tests","total_tests","total_tests_per_thousand",
                                       "total_vaccinations","people_vaccinated","people_fully_vaccinated",
                                       "total_boosters","new_vaccinations","total_vaccinations_per_hundred"].sum()
world_data.reset_index(inplace=True)
print("shape of world_data",world_data.shape)
display("COVID DATA OF WORLD :",world_data)

#Worldwide Total Cases and Total Deaths Over Time
fig=px.line(world_data,x="date",y=["total_cases","total_deaths"],
            title="Worldwide Total Cases and Total Deaths Over Time")
fig.show()

#World-wide Daily Covid_19 Cases
fig=px.line(world_data,x="date",y=["new_cases","new_deaths"],labels={"value":"Count"},
            title="Worldwide Cases and Deaths Daily")
fig.show()

#World-wide Vaccination Data from starting of Vaccination
start_date = "2020-12-12"                    #Starting Date of Vaccination
end_date = "2021-09-23"
mask = (world_data['date'] > start_date) & (world_data['date'] <= end_date)
vaccination_start = world_data.loc[mask]
vaccination_start.reset_index(drop=True,inplace=True)
vaccination_start.head()

#Worldwide Daily Vaccinations Over Time
fig=px.line(vaccination_start,x="date",y="new_vaccinations",
            title="Worldwide Daily Vaccinations Over Time")
fig.show()

#World-wide Total Vaccinations Over Time
fig=px.line(vaccination_start,x="date",y="total_vaccinations",
            title="World-wide Total Vaccinations Over Time")
fig.show()

#Vaccination Dose Administered World-wide
fig=px.line(vaccination_start,x="date",y=["people_vaccinated", "people_fully_vaccinated","total_boosters"],
            title="Vaccination Doses Administered World Wide",labels={"value":"Count"})
fig.show()

#Daily Covid_19 Testing vs New Cases
fig=px.line(world_data ,x="date",y=["new_tests", "new_cases"],
            title="Relation Between Daily Testing and Daily Cases",labels={"value":"Count"})
fig.show()

#World-Wide Stringency Index Over Time
figure = px.choropleth(data,locations="location",locationmode="country names",color="stringency_index",
                     animation_frame="date",hover_data=["continent","new_cases","population"])
figure.update_layout(title="World Wide Stringency Index Over Time",template="plotly_dark")
figure.show()

#Continent Data
continent=data["continent"].unique()  
print("List of Continents:",continent)

#Total Cases ,Deaths and Recovered Patients Based on Continent
cases_continent = data.groupby("continent")["total_cases"].sum()    #Acessing total cases for each continent
deaths_continent = data.groupby("continent")["total_deaths"].sum()  #Acessing total deaths for each contient
deaths_con = deaths_continent[continent].values
cases_con = cases_continent[continent].values
recovered_con = np.subtract(cases_con,deaths_con)               #Acessing total recoverd cases for each continent
continent_total=pd.DataFrame({"Continent":continent,"Total_cases":cases_con,"Total_death":deaths_con,
                              "Total_recovered":recovered_con})
continent_total.sort_values("Total_cases",axis = 0, ascending=False, inplace = True)    
continent_total.reset_index(drop=True)
display("Continent Data",continent_total)

#Percentage of Confirmed Cases on each Continent
my_explode = (0.1,0.0,0.0,0.0,0.0,0.0)
my_labels = continent_total["Continent"]
continent_total.plot(kind= "pie",y = 'Total_cases',autopct='%1.0f%%',labels=continent_total["Continent"],
                     figsize=(8,10),explode=my_explode,legend=True,shadow=True,
                     title="Percentage of Confirmed Cases on Each Continent",colormap='plasma')

#Percentage of Confirmed Deaths on each Continent
my_explode = (0.0,0.1,0.0,0.0,0.0,0.0)
my_labels = continent_total["Continent"]
continent_total.plot(kind = "pie",y = 'Total_death',autopct='%1.0f%%',labels = continent_total["Continent"],
                     figsize = (8,10),explode = my_explode,legend=True,shadow = True,
                     title = "Percentage of Confirmed Deaths on Each Continent",colormap='plasma')

#Country Data
#Countries with No Covid Cases
no_total_cases = pd.DataFrame(data.groupby("location")["new_cases"].sum())
covid_free_countries = no_total_cases[no_total_cases["new_cases"] == 0].reset_index()
covid_free_countries.rename(columns={"new_cases":"Total_cases"},inplace=True)
print("NUMBER OF COUNTRIES WITH NO COVID CASES :",len(covid_free_countries))
display(covid_free_countries)

#Preprocess Country Data
#Drop countries with no covid cases
covid_free = covid_free_countries["location"].unique() 
for location in covid_free:
    data = data.drop(data[data["location"] == location].index)
    
country=data["location"].unique()    #unique country locations
location = country.tolist()          #convert to list

#Function to add feature values to dataset
def preprocess (data,label):
    """
    INPUTS :
    data   : input dataframe
    label  : feature name
    _____________________________________________________
    OUTPUTS :
    feature : value of each feature sorted by country
    _____________________________________________________
    """
    feature =[]
    for loc in country:
        out = pd.DataFrame(data[data["location"] == loc][label])
        out = out.sort_values(label,axis=0,ascending=False).head(1)
        feature.append(out[label].values[0])
    return feature

#features to add
values = ["total_cases","total_deaths","population","total_tests","total_tests_per_thousand","total_vaccinations",
          "people_vaccinated", "people_fully_vaccinated", "total_boosters","total_vaccinations_per_hundred",
          "population_density","median_age","aged_65_older","aged_70_older", "gdp_per_capita", "extreme_poverty",
          "cardiovasc_death_rate","diabetes_prevalence", "female_smokers","male_smokers", "handwashing_facilities", 
          "hospital_beds_per_thousand","life_expectancy", "human_development_index"]

processed_dict = {}

#function call
for i in values:
    dict_value=preprocess(data,i)
    processed_dict[i] = dict_value
    
# dictionary for country names
country_dict={"Country":location}
#concat two dictionaries
new_dict = {**country_dict,**processed_dict}
#create a new Dataframe
country_total=pd.DataFrame(new_dict)
display("Covid Data based on Country ::",country_total)

# Add Case fatality ratio for countries to data
case_fatality_ratio = country_total["total_deaths"]/country_total["total_cases"]
# Add Mortality ratio for countries to data
mortality_ratio = country_total["total_deaths"]/country_total["population"]
country_total = country_total.assign(Case_fatality_ratio = case_fatality_ratio,Mortality_ratio = mortality_ratio)

print("Featue Names ::",country_total.columns)

#Sort Countries based on Number of Covid Cases
country_cases_sorted = country_total.sort_values("total_cases", axis = 0, ascending = False)
country_cases_sorted.reset_index(drop=True,inplace=True)
display("COUNTRIES WITH MOST NUMBER OF CASES :",country_cases_sorted)

#Top 10 Countries with Most Cases
top_10_cases = country_cases_sorted.head(10)
fig = px.bar(top_10_cases, x="Country", y=["total_cases","total_deaths"], barmode="group",
             title="Top 10 Countries with Most Cases",labels={"value":"Count"})
fig.show()

#Top 10 Countries with Most Death Counts
country_death_sorted = country_total.sort_values("total_deaths", axis = 0, ascending = False)  
country_death_sorted.reset_index(drop=True,inplace=True)
top_10_deaths = country_death_sorted.head(10)
display("Top 10 Countries with Most Death Counts",top_10_deaths[top_10_deaths.columns[:5]])

fig = px.bar(top_10_deaths, x="Country", y=["total_deaths","total_cases"], barmode="group",
             title="Top 10 Countries with Most Deaths")
fig.show()

#Testing Data of Countries
#Country with Max-Covid Test Rates
#Calculating Covid Test Rate for Each Country
country_total["test_rate"] = country_total["total_tests"]/country_total["population"]
max_test_rates = country_total.sort_values("test_rate",axis = 0, ascending = False)[:10]
max_test_rates.reset_index(drop=True,inplace=True)

fig = px.bar(max_test_rates, x="test_rate", y="Country",
             title="Top 10 Countries with Maximum Covid Test Rates", height=600, text="test_rate",
             orientation="h",color="test_rate",hover_data=["population"])
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()

#Case Fatality Rate vs Median Age of Different Countries
# countries with high median age have low case fatality rate
high_median = country_total.sort_values("median_age",axis=0,ascending=False)
high_median_age = high_median[["Country","median_age","life_expectancy","Case_fatality_ratio"]].dropna().reset_index(drop=True)[:20]

fig = px.scatter(high_median_age, x="median_age", y="Case_fatality_ratio",hover_name="Country",color="Country", 
                  size="Case_fatality_ratio",labels={"median_age":"High median age"},size_max=30,
                 title="Case Fatality Rate in Counries with High Median of Age")
fig.show()

# countries with low median age have high case fatality rates
low_median = country_total.sort_values("median_age",axis=0,ascending=True)
low_median_age = low_median[["Country","median_age","life_expectancy","Case_fatality_ratio"]].dropna().reset_index(drop=True)[:20]

fig = px.scatter(low_median_age, x="median_age", y="Case_fatality_ratio",hover_name="Country",color="Country", 
                  size="Case_fatality_ratio",labels={"median_age":"Low median age"},size_max=30,
                 title="Case Fatality Rate in Counries with Low Median of Age")
fig.show()

#Comparing Cases in Countries with High Hand Washing Facilities and Low hand Washing Facilities
# Top 10 countries with high handwashing_facilities
high_handwash_facility = country_total.sort_values("handwashing_facilities",axis=0,ascending=False).reset_index(drop=True)[:10]
display("Countries With Better Handwashing Facilities",high_handwash_facility[["Country","handwashing_facilities","total_cases"]])

# Top 10 countries with low handwashing_facilities
low_handwash_facility  = country_total.sort_values("handwashing_facilities",axis=0,ascending=True).reset_index(drop=True)[:10]
display("Countries With Low Handwashing Facilities",low_handwash_facility[["Country","handwashing_facilities","total_cases"]])

# Visualize 
fig, axes = plt.subplots(1, 2,figsize=(17, 7))
high_handwash_facility.plot(kind="bar",ax=axes[0],x="Country",y="total_cases",ylabel="Total cases",
                            title="Total Cases in Countries with High Hand Washing Facilities",grid=True)

low_handwash_facility.plot(kind="bar",ax=axes[1],x="Country",y="total_cases",color="red",ylabel="Total cases",
                           title="Total Cases in Countries with Low Hand Washing Facilities",grid=True)

#Top 10 Countries with High Hospital Beds Per Thousand Rate
hospital_beds = country_total.sort_values("hospital_beds_per_thousand",axis=0,ascending=False).reset_index(drop=True)[:10]
display("Top 10 Countries with High Hospital Beds Per Thousand Rate",hospital_beds)
fig = px.bar(hospital_beds,x="Country",y="Case_fatality_ratio",title="Top 10 Countries with High Hospital Beds Per Thousand Rate",
             hover_data=["hospital_beds_per_thousand","Mortality_ratio"],color="Case_fatality_ratio")
fig.show()

#Impact of Covid19 in Countries with high Cardiovascular Death Rate and Diabeties Privelence
# Top 10 Countries with High Cardiovascular Death Rates
cardiovasc_death_rate = country_total.sort_values("cardiovasc_death_rate",axis=0,ascending=False)[:10]
# Top 10 Countries with High Diabeties Privelence Rates
diabetes_prevalence = country_total.sort_values("diabetes_prevalence",axis=0,ascending=False)[:10]

fig = make_subplots(rows=1, cols=2,subplot_titles=("Case fatality ratio of high cardiovasc death rate countries",
                                                   "Case fatality ratio of high diabetes prevalence countries"))

fig.add_trace(go.Bar(x=cardiovasc_death_rate["Country"],y=cardiovasc_death_rate["Case_fatality_ratio"]),1,1)
fig.add_trace(go.Bar(x=diabetes_prevalence["Country"],y=diabetes_prevalence["Case_fatality_ratio"]),1,2)

fig.update_yaxes(title_text="Case_fatality_ratio", row=1, col=1)
fig.update_yaxes(title_text="Case_fatality_ratio", row=1, col=2)
fig.update_xaxes(title_text="Country with high cardiovasc death rate", row=1, col=1)
fig.update_xaxes(title_text="Country with high diabetes prevalence", row=1, col=2)
fig.show()

#Covid Cases and Deaths in Top 10 Counties with Extreme Poverty
extreme_poverty = country_total.sort_values("extreme_poverty",axis=0,ascending=False).reset_index(drop=True)[:10]
display("Top 10 Countries with  Extreme Poverty Rate",extreme_poverty[["Country","extreme_poverty","total_cases","total_deaths"]])

fig = px.bar(extreme_poverty,x=["total_cases","total_deaths"],y="Country",height=700,barmode="group",
             title="Covid Cases & Deaths in Top 10 Countries with Extreme Poverty Rate",
             hover_data=["human_development_index"],orientation="h",labels={"value":"Count"})
fig.show()

#Covid Cases and Deaths in Counties with high Human Development index
hdi = country_total.sort_values("human_development_index", axis =0 ,ascending=False).reset_index(drop=True)[:10]
display("TOp 10 Countries with High Human Development Index",hdi[["Country","human_development_index","total_cases","total_deaths"]])

# Visualize
fig = px.bar(hdi,x = "Country", y = ["total_cases","total_deaths"],barmode = "group",
             hover_data = ["human_development_index","life_expectancy"],title = "Covid19 & Human Development Index",
             labels = {"Country":"Country (HDI Ranking)"})

fig.show()

#Sort Countries with High Male and Female smokeing Rates
male_smoker = country_total.sort_values("male_smokers", axis=0,ascending = False).reset_index(drop=True)[:10]
female_smoker = country_total.sort_values("female_smokers",axis=0, ascending = False).reset_index(drop=True)[:10]
display("Counties with High Rate of Male Smokers",male_smoker[["Country","male_smokers","total_cases","total_deaths"]])
display("Counties with High Rate of female Smokers",female_smoker[["Country","female_smokers","total_cases","total_deaths"]])

#Visualize Case Fatality Rate in Country with High Male Smoker & Female Smoker
fig, axes = plt.subplots(1, 2,figsize=(17, 7))
male_smoker.plot(kind ="bar",ax=axes[0],x="Country",y="Case_fatality_ratio",ylabel="Case fatality ratio",
                            title="Case Fatality Rate in Countries with High Male Smokers",grid=True)

female_smoker.plot(kind ="bar",ax=axes[1],x="Country",y="Case_fatality_ratio",color="red",ylabel="Case fatality ratio",
                           title="Case Fatality Rate in Countries with High Female Smokers",grid=True)

#Vaccination Data of Countries
#Countries with high people vaccinated per hundred 
country_vaccine = country_total.sort_values("total_vaccinations_per_hundred", axis = 0, ascending = False).reset_index(drop=True)[:10]  
display("COUNTRIES HAVING MOST PEOPLE VACCINATED PER HUNDRED :",country_vaccine)

fig = px.bar(country_vaccine, x="Country", y=["total_vaccinations_per_hundred","population_density"], 
             barmode="group",title="Total Vaccinations Per Hundred",labels={"value":"Count"})
fig.show()

#Top 10 Countries with Most Number of Vaccinations Administered
country_total_vaccination = country_total.sort_values("total_vaccinations", axis = 0, ascending = False)[:10]
top_10_total_vaccination=country_total_vaccination[country_total_vaccination.columns[:10]].reset_index(drop=True)
display("Top 10 Countries with Most Number of Vaccinations",top_10_total_vaccination)

fig = px.bar(country_total_vaccination,x="Country", y=["total_vaccinations","people_vaccinated","people_fully_vaccinated"],
             barmode="group",title="Top 10 Countries with Most Number of Vaccinations Administered",labels={"value":"Count"})
fig.show()

#Top 10 Countries with Most Number of Booster Shot Vaccinations
total_booster = country_total.sort_values("total_boosters", axis = 0, ascending = False)[:10]
top_10_total_booster = total_booster[["Country","population","people_fully_vaccinated","total_boosters"]].reset_index(drop=True)
display("Top 10 Countries With Maximum Booster Shots Administered",top_10_total_booster)

fig = px.bar(top_10_total_booster, x="Country", y=["total_boosters","people_fully_vaccinated"], barmode="group",
             title="Top 10 Countries with Most Number of Booster Shot Vaccinations",labels={"value":"Count"})
fig.show()

#Top 10 Countries with Highest Case Fatality Rate
fig = px.bar(country_total.sort_values("Case_fatality_ratio", ascending=False)[:10], 
             x="Case_fatality_ratio", y="Country",
             title="Top 10 Countries with Highest Case Fatality Rate ",height=600, text="Case_fatality_ratio",
             orientation="h",hover_data=["population"], color="Case_fatality_ratio")
fig.update_layout(yaxis={"categoryorder":"total ascending"})
fig.show()

#Top 10 Countries with Highest Mortality Rate
fig = px.bar(country_cases_sorted.sort_values("Mortality_ratio", ascending=False)[:10], x="Mortality_ratio", 
             y="Country",title="Top 10 Countries with Highest Mortality Rate",height=600, text="Mortality_ratio", 
             orientation="h",hover_data=["population"], color="Mortality_ratio")
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()
#Indian Covid_19 Data
india_data = data.loc[data["location"]=="India"].reset_index(drop=True)

indian_dataset=india_data[["location","date","total_cases","new_cases","total_deaths","new_deaths",
                           "total_tests","new_tests","total_vaccinations","new_vaccinations","people_vaccinated",
                           "people_fully_vaccinated","total_boosters","stringency_index"]]
display("Indian Covid_19 Data",indian_dataset.head(10))

#Total Cases and Deaths in India
fig=px.line(india_data,x="date",y=["total_cases","total_deaths"],title="Total Cases and Deaths in India",
            labels={"value":"Count"})
fig.show()

#Daily Cases and Deaths in India
fig=px.line(india_data,x="date",y=["new_cases","new_deaths"],title="Daily Cases and Deaths in India",
            labels={"value":"Count"})
fig.show()

#Total Testing for Covid_19 in India
fig=px.line(india_data,x="date",y=["total_tests"],title="Total Testing for Covid_19 in India",
            labels={"value":"Count"})
fig.show()

#Number of Covid_19 Testings Daily in India
fig=px.line(india_data,x="date",y=["new_tests"],title="Daily Testing for Covid_19 in India",
            labels={"value":"Count"})
fig.show()

# Grouping Data from Starting Date of Vaccination 
start_date = "2021-01-16"
end_date = "2021-09-23"
mask = (indian_dataset['date'] > start_date) & (indian_dataset['date'] <= end_date)
vaccination_start_india = indian_dataset.loc[mask]
vaccination_start_india.reset_index(drop=True,inplace=True)
display("Vaccination Data in India ",vaccination_start_india.head(10))

#Daily Vaccinations in India
fig=px.line(vaccination_start_india,x="date",y="new_vaccinations",title="Daily Vaccination in India",
            labels={"value":"Count"})
fig.show()

#Vaccination Doses Administered in India
fig=px.line(vaccination_start_india,x="date",y=["total_vaccinations","people_vaccinated","people_fully_vaccinated"],
            title="Vaccination Doses Administered in India",labels={"value":"Count"})
fig.show()

#Impact of Stringency Index on the Rate of Covid Cases and Deaths in India
fig = px.scatter(indian_dataset, x=["new_cases","new_deaths"], y="stringency_index",
                 title="Impact of Stringency Index on the Number of Covid Cases and Deaths in India")
fig.show()

#Comparing Daily Cases in India and United States
usa_data = data.loc[data["location"]=="United States"].reset_index()
usa_data = usa_data[["date","new_cases"]]
usa_df = usa_data.rename(columns={"new_cases": "usa_daily_cases"})
india_cases = indian_dataset[["date","new_cases"]]
india_df = india_cases.rename(columns={"new_cases": "india_daily_cases"})

compare_data = pd.concat([usa_df,india_df])
fig=px.line(compare_data,x="date",y=["usa_daily_cases","india_daily_cases"],
            title="Comparing Daily Cases in India and USA",labels={"value":"Count"})
fig.show()