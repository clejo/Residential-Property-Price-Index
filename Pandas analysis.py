import numpy as np
import pandas as pd

Property_prices_raw = pd.read_csv("Residential Property Price Indices - OECD 2018.csv", delimiter=",")
Macroeco_indicators_raw = pd.read_csv("Economic outlook - OECD Nov 2018.csv", delimiter=",")

Property_prices_whole_countries = Property_prices_raw[Property_prices_raw['Geographical coverage']=='Whole country']
Property_prices_countries = Property_prices_whole_countries[Property_prices_whole_countries['SUBJECT'].isin(['RP0101'])][["LOCATION","Country"]]
Property_prices_unique_countries = Property_prices_countries.drop_duplicates(subset=None, keep='first')

Macroeco_indicators_countries = Macroeco_indicators_raw[["LOCATION","Country"]]
Macroeco_indicators_unique_countries = Macroeco_indicators_countries.drop_duplicates(subset=None, keep='first')

Studied_countries = Macroeco_indicators_unique_countries.loc[Macroeco_indicators_unique_countries['LOCATION'].isin(Property_prices_unique_countries['LOCATION'])]


Macroeco_indicators_unique_indicators = Macroeco_indicators_raw[["VARIABLE","Variable"]].drop_duplicates(subset=None, keep='first')[["VARIABLE","Variable"]]
Studied_indicators = Macroeco_indicators_unique_indicators[Macroeco_indicators_unique_indicators['VARIABLE'].isin(['IRL', 'CPI','CPIH', 'IHV', 'UNR', 'ET_ANNPCT', 'GDPV_CAP', 'GDPV'])]



Filterd_DATA_Property_prices = Property_prices_raw[
                                                Property_prices_raw['LOCATION'].isin(Studied_countries['LOCATION']) 
                                              & Property_prices_raw['Frequency'].isin(['Annual'])
                                              & Property_prices_raw['Unit'].isin(['Index'])
                                              & Property_prices_raw['Geographical coverage'].isin(['Whole country'])
                                              & Property_prices_raw['SUBJECT'].isin(['RP0101'])
                                                    ] [['LOCATION','TIME','Value']]

Filterd_DATA_Macroeco_indicators = Macroeco_indicators_raw[ 
                                                      Macroeco_indicators_raw['LOCATION'].isin(Studied_countries['LOCATION']) \
                                                    & Macroeco_indicators_raw['VARIABLE'].isin(Studied_indicators['VARIABLE'])
                                                    & Macroeco_indicators_raw['Frequency'].isin(['Annual'])
                                                            ] [['LOCATION','VARIABLE','TIME','Value']]                                                    

DATA = {}
for country in Studied_countries['LOCATION']:
    DATA[country]={}

    for variable in Studied_indicators['VARIABLE']:
        DATA[country][variable]={}
        for year in range(2000, 2020):

            #Some countries use CPI, other CPIH, we combine them without distinction
            if variable !='CPIH' and variable !='CPI':
                
                try:
                    DATA[country][variable][year] = Filterd_DATA_Macroeco_indicators[
                                                        Filterd_DATA_Macroeco_indicators['LOCATION'].isin([country,""])
                                                        & Filterd_DATA_Macroeco_indicators['VARIABLE'].isin([variable,""])
                                                        & Filterd_DATA_Macroeco_indicators['TIME'].isin([year,""])
                                                           ].iloc[0]['Value']
                except IndexError:
                    DATA[country][variable][year] = "na"

            else:

                try:
                    DATA[country]['CPI'][year] = Filterd_DATA_Macroeco_indicators[
                                                        Filterd_DATA_Macroeco_indicators['LOCATION'].isin([country,""])
                                                        & Filterd_DATA_Macroeco_indicators['VARIABLE'].isin([variable,""])
                                                        & Filterd_DATA_Macroeco_indicators['TIME'].isin([year,""])
                                                                             ].iloc[0]['Value']
                except IndexError:
                    if year not in DATA[country]['CPI']:
                        DATA[country]['CPI'][year] = "na"
    
    DATA[country]['RP0101']={}
    for year in range(2000, 2020):
        try:
            DATA[country]['RP0101'][year] = Filterd_DATA_Property_prices[
                                                        Filterd_DATA_Property_prices['LOCATION'].isin([country,""])
                                                        & Filterd_DATA_Property_prices['TIME'].isin([str(year),""])
                                                                      ].iloc[0]['Value']                                                       
        except IndexError:
                    DATA[country]['RP0101'][year] = "na"
    
np.save('DATA.npy', DATA)

#removing CPIH since it was combined with CPI
Studied_indicators_wo_CPIH = Studied_indicators[Studied_indicators['VARIABLE'].isin(['IRL', 'CPI', 'IHV', 'UNR', 'ET_ANNPCT', 'GDPV_CAP', 'GDPV'])]
Studied_indicators_wo_CPIH.to_csv('Studied_indicators.csv')
Studied_countries.to_csv('Studied_countries.csv')
