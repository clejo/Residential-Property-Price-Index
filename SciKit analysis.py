import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
DATA = np.load('DATA.npy').item()
Studied_countries = pd.read_csv("Studied_countries.csv", delimiter=",")
Studied_indicators = pd.read_csv("Studied_indicators.csv", delimiter=",")

import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def plot_1_variable(country,firstyear,lastyear,xaxis_var):
    
    reglin_dict = linreg_prediction_1_var(country,firstyear,lastyear,xaxis_var)

    xvalues = reglin_dict["xvalues"]
    yvalues = reglin_dict["yvalues"]
    x_prediction = reglin_dict["x_prediction"]
    y_prediction = reglin_dict["y_prediction"]
    years = reglin_dict["years"]

    #Plot empirical values from OECD dataset
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(xvalues,yvalues,'x')
            #axes labels and title
    x_var_full_name = Studied_indicators[Studied_indicators['VARIABLE'].isin([xaxis_var,''])].iloc[0]['Variable']
    y_var_full_name = 'Residential Property Price Indices'
    country_full_name = Studied_countries[Studied_countries['LOCATION'].isin([country,''])].iloc[0]['Country']
            #labels on dots
    plt.xlabel(x_var_full_name)
    plt.ylabel(y_var_full_name)
    plt.title(country_full_name)
    for i, yr in enumerate(years):
        ax.annotate(yr, (xvalues[i], yvalues[i]))

    #Compute regression model
    model = linear_model.LinearRegression(fit_intercept=True)
    model.fit(np.asarray(xvalues)[:, np.newaxis],yvalues)

    #Plot regression line
    xfit = np.linspace(min(xvalues),max(xvalues), 1000)
    yfit = model.predict(xfit[:, np.newaxis])
    plt.plot(xfit, yfit)

    #Plot prediction in 2019
    plt.plot(x_prediction,y_prediction,'o')
    ax.annotate(2019,(x_prediction,y_prediction))

    #Save the graph to local
    plt.savefig("static/img/plot.png")

    plt.close("all")

    results = reglin_dict["results"]
    return results

def linreg_prediction_1_var(country,firstyear,lastyear,xaxis_var):

    reglin_dict = {} #Will be returned as fuction output
    
    xvalues=[]
    yvalues=[]
    years=[]
    
    for year in range(firstyear,lastyear + 1):
        x = DATA[country][xaxis_var][year]
        y = DATA[country]['RP0101'][year]
        if x != 'na' and y != 'na':
            xvalues.append(x)
            yvalues.append(y)
            years.append(year)

    reglin_dict["xvalues"] = xvalues
    reglin_dict["yvalues"] = yvalues
    reglin_dict["years"] = years
    
    #Compute regression model
    model = linear_model.LinearRegression(fit_intercept=True)
    model.fit(np.asarray(xvalues)[:, np.newaxis],yvalues)

    #Compute prediction for 2019
    x_prediction = DATA[country][xaxis_var][2019]
    x_prediction_2D_array= x_prediction.reshape(-1, 1)
    y_prediction = model.predict(x_prediction_2D_array)
    reglin_dict["x_prediction"] =  x_prediction
    reglin_dict["y_prediction"] =  y_prediction
    
    slope = round(model.coef_[0],3)
    if slope == 0:
        slope = model.coef_[0]
    intercept = round(model.intercept_,3)
    R2 = round(model.score(np.asarray(xvalues)[:, np.newaxis],yvalues),3)
    two_years_average_growth = math.sqrt(y_prediction.item()/DATA[country]['RP0101'][2017].item())-1

    linreg_results={}
    x_var_full_name = Studied_indicators[Studied_indicators['VARIABLE'].isin([xaxis_var,''])].iloc[0]['Variable']
    linreg_results["line0"]= (Studied_countries[Studied_countries['LOCATION'].isin([country,''])].iloc[0]['Country']+'. '+x_var_full_name)
    linreg_results["line1"]= ("Model slope: "+ str(slope))
    linreg_results["line2"]= ("Model intercept: "+ str(intercept))
    linreg_results["line3"]= ("R²:"+ str(R2))
    linreg_results["line4"]= ('Anticipated Residential Price Index 2019: '+ str(round(y_prediction.item(),1)))
    linreg_results["line5"]= ('2017-2019 yearly average growth: '+ str(round(two_years_average_growth*100,1)) +'%')
    linreg_results["line6"]= ""
    linreg_results["line7"]= ""

    reglin_dict["results"] =  linreg_results

    return reglin_dict

        
def linreg_results(country,firstyear,lastyear,xaxis_varS):
        
    linreg_results = {}
    reglin_dict = linreg_prediction(country,firstyear,lastyear,xaxis_varS)
    
    y_prediction = reglin_dict["y_prediction"]
    R2 = reglin_dict["R2"]
    coeff = reglin_dict["coeff"]
    intercept = reglin_dict["intercept"]
    two_years_average_growth = reglin_dict["two_years_average_growth"]
    


    li = ["","",""]
    i=0
    for xaxis_var in xaxis_varS:
        coeff_round = round(coeff[i],3)
        if coeff_round == 0:
            coeff_round = coeff[i]
        var_full_name = Studied_indicators[Studied_indicators['VARIABLE'].isin([xaxis_var,''])].iloc[0]['Variable']
        li[i]=(str(var_full_name)+". Coefficient "+": "+str(coeff_round))
        i=i+1

    linreg_results["line0"]= Studied_countries[Studied_countries['LOCATION'].isin([country,''])].iloc[0]['Country']
    linreg_results["line1"]= li[0]
    linreg_results["line2"]= li[1]
    linreg_results["line3"]= li[2]
    linreg_results["line4"]=("Model intercept: "+ str(round(intercept,3)))
    linreg_results["line5"]=("R²:"+ str(round(R2,3)))
    linreg_results["line6"]=('Anticipated Residential Price Index 2019: '+ str(round(y_prediction.item(),1)))
    linreg_results["line7"]=('2017-2019 yearly average growth: '+ str(round(two_years_average_growth*100,1)) +'%')
    
    return linreg_results

def linreg_prediction(country,firstyear,lastyear,xaxis_varS):

    reglin_dict = {} #Will be returned as fuction output
    
    r_years = relevant_years(country,firstyear,lastyear,xaxis_varS)
    
    xvalues = np.zeros((len(xaxis_varS),len(r_years)))
    yvalues = []

    for year in r_years:
        yvalues.append(DATA[country]['RP0101'][year])
        i = 0
        for xaxis_var in xaxis_varS:
            j=r_years.index(year)
            xvalues[i][j] = DATA[country][xaxis_var][year].item()
            i = i+1
    
    xvalues_tr = list(map(list, zip(*xvalues)))

    reglin_dict["xvalues"] = xvalues_tr
    reglin_dict["yvalues"] = yvalues
    reglin_dict["years"] = r_years
    
    #Compute regression model
    model = linear_model.LinearRegression(fit_intercept=True)
    model.fit(xvalues_tr,yvalues)

    #Compute prediction for 2019
    x_prediction = np.zeros((len(xaxis_varS),1))
    i = 0
    for xaxis_var in xaxis_varS:
        x_prediction[i][0] = DATA[country][xaxis_var][2019].item()
        i = i+1
    x_prediction_tr = list(map(list, zip(*x_prediction)))
    y_prediction = model.predict(x_prediction_tr)
    reglin_dict["x_prediction"] =  x_prediction
    reglin_dict["y_prediction"] =  y_prediction
    
    #Compute regression parameters
    R2 = model.score(xvalues_tr,yvalues)
    reglin_dict["R2"] = R2  
    
    reglin_dict["coeff"]={}
    i=0
    for xaxis_var in xaxis_varS:
        reglin_dict["coeff"][i] =  model.coef_[i]
        i=i+1
        
    reglin_dict["intercept"] =  model.intercept_
    two_years_average_growth = math.sqrt(y_prediction.item()/DATA[country]['RP0101'][2017].item())-1
    reglin_dict["two_years_average_growth"] = two_years_average_growth
    
    return reglin_dict


def relevant_years(country,firstyear,lastyear,xaxis_varS):
    
    r_years = []

    for year in range(firstyear,lastyear+1):
        keep = True
        
        if DATA[country]['RP0101'][year] == 'na':
                keep = False
        
        for xaxis_var in xaxis_varS:
            if DATA[country][xaxis_var][year] == 'na':
                    keep = False
            
        if keep:
            r_years.append(year)
    
    return r_years

from flask import Flask, send_from_directory, render_template, request
app = Flask(__name__, static_url_path="/static")

@app.route('/')
def main_page():
    return render_template("result.html", plot_possible=False, results="")

@app.route('/selection', methods = ['POST', 'GET'])
def dataAnalysis():

    selection=request.form
    country = selection['select country']
    factor1 = selection['select factor1']
    factor2 = selection['select factor2']
    factor3 = selection['select factor3']
    start_year = selection['start year']
    end_year = selection['end year']

    plot_possible = False
    results = ""

    if country !='null' and factor1 != 'null' and start_year != 'null' and end_year !='null':
        
        start_year=int(start_year)
        end_year=int(end_year)
        if factor2 != 'null':
            if factor3 != 'null':
                xaxis_varS=[factor1,factor2,factor3]
                results = linreg_results(country,start_year,end_year,xaxis_varS)
            else:
                xaxis_varS=[factor1,factor2]
                results = linreg_results(country,start_year,end_year,xaxis_varS)
        else:
            results = plot_1_variable(country,start_year,end_year,factor1)
            plot_possible = True
        
    return render_template("result.html", plot_possible=plot_possible, results=results)

   
@app.route("/style.css")
def sendCSS():
    return send_from_directory("static", "style.css")

if __name__ == '__main__':
   app.run()