from django.http.response import HttpResponse
from django.shortcuts import render,redirect

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np
from statistics import mean

# Create your views here.

def portugal(request):
    return render(request,'portugal.html')

def feedback(request):
    return render(request,'feedback.html')

def FootballPrediction(request):
    
    dataS = {}
    dataS['Club'] = "Real Madrid"
    dataS['League_Position'] = "2"
    dataS['Champions_League'] = "Winner"

    return render(request,'Football.html',dataS)

def Football_Landing(request):
    
    # return render(request,'testing.html')
    return render(request,'football_landing.html')

def Season_Based_eredivise(request):

    return render(request,'eredivisie.html')


def Season_Based_la_liga(request):

    return render(request,'la_liga.html')


def Season_Based_premier_league(request):

    return render(request,'premier_league.html')


def Season_Based_serie_a(request):

    return render(request,'serie_a.html')


def Season_Based_Prediction(request):
    
    return render(request,'SeasonBased.html') 


def Team_Based_Prediction(request):
    
    return render(request,'TeamBased.html')     


def Team_Based_atletico_madrid(request):

    df = pd.read_csv('Atletico Madrid.csv')

    labels_league_position = df['League Position']
    labels_champions_league = df['Champions League']
    labels_total_goals = df['Total Goals']
    labels_total_bookings = df['Total Bookings']
    df.drop('League Position',axis=1,inplace=True)
    df.drop('Champions League',axis=1,inplace=True)
    df.drop('Total Goals',axis=1,inplace=True)
    df.drop('Total Bookings',axis=1,inplace=True)

    # champions league predictions
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_champions_league, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionChampionsLeague = list(model.predict(X_test))


    # la liga prediction
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_league_position, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionLeague = model.predict(X_test)

    # total goals
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_goals, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionGoals = model.predict(X_test)

    # total bookings
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_bookings, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionBookings = model.predict(X_test)

    dataS = {}
    dataS['Club'] = "Atletico Madrid"
    dataS['League_Position'] = int(predictionLeague)
    dataS['Champions_League'] = predictionChampionsLeague[0]
    dataS['Goals'] = int(predictionGoals)
    dataS['Bookings'] = int(predictionBookings)

    return render(request,'atletico_madrid.html',dataS)


def Team_Based_real_madrid(request):

    df = pd.read_csv('Atletico Madrid.csv')

    labels_league_position = df['League Position']
    labels_champions_league = df['Champions League']
    labels_total_goals = df['Total Goals']
    labels_total_bookings = df['Total Bookings']
    df.drop('League Position',axis=1,inplace=True)
    df.drop('Champions League',axis=1,inplace=True)
    df.drop('Total Goals',axis=1,inplace=True)
    df.drop('Total Bookings',axis=1,inplace=True)

    # champions league predictions
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_champions_league, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionChampionsLeague = list(model.predict(X_test))


    # la liga prediction
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_league_position, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionLeague = model.predict(X_test)

    # total goals
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_goals, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionGoals = model.predict(X_test)

    # total bookings
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_bookings, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionBookings = model.predict(X_test)

    dataS = {}
    dataS['Club'] = "Real Madrid"
    dataS['League_Position'] = int(predictionLeague)
    dataS['Champions_League'] = predictionChampionsLeague[0]
    dataS['Goals'] = int(predictionGoals)
    dataS['Bookings'] = int(predictionBookings)

    return render(request,'real_madrid.html',dataS)


def Team_Based_ac_milan(request):

    df = pd.read_csv('Atletico Madrid.csv')

    labels_league_position = df['League Position']
    labels_champions_league = df['Champions League']
    labels_total_goals = df['Total Goals']
    labels_total_bookings = df['Total Bookings']
    df.drop('League Position',axis=1,inplace=True)
    df.drop('Champions League',axis=1,inplace=True)
    df.drop('Total Goals',axis=1,inplace=True)
    df.drop('Total Bookings',axis=1,inplace=True)

    # champions league predictions
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_champions_league, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionChampionsLeague = list(model.predict(X_test))


    # la liga prediction
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_league_position, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionLeague = model.predict(X_test)

    # total goals
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_goals, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionGoals = model.predict(X_test)

    # total bookings
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_bookings, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionBookings = model.predict(X_test)

    dataS = {}
    dataS['Club'] = "AC Milan"
    dataS['League_Position'] = int(predictionLeague)
    dataS['Champions_League'] = predictionChampionsLeague[0]
    dataS['Goals'] = int(predictionGoals)
    dataS['Bookings'] = int(predictionBookings)

    return render(request,'ac_milan.html',dataS)


def Team_Based_inter_milan(request):

    df = pd.read_csv('Atletico Madrid.csv')

    labels_league_position = df['League Position']
    labels_champions_league = df['Champions League']
    labels_total_goals = df['Total Goals']
    labels_total_bookings = df['Total Bookings']
    df.drop('League Position',axis=1,inplace=True)
    df.drop('Champions League',axis=1,inplace=True)
    df.drop('Total Goals',axis=1,inplace=True)
    df.drop('Total Bookings',axis=1,inplace=True)

    # champions league predictions
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_champions_league, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionChampionsLeague = list(model.predict(X_test))


    # la liga prediction
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_league_position, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionLeague = model.predict(X_test)

    # total goals
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_goals, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionGoals = model.predict(X_test)

    # total bookings
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_bookings, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionBookings = model.predict(X_test)

    dataS = {}
    dataS['Club'] = "Inter Milan"
    dataS['League_Position'] = int(predictionLeague)
    dataS['Champions_League'] = predictionChampionsLeague[0]
    dataS['Goals'] = int(predictionGoals)
    dataS['Bookings'] = int(predictionBookings)

    return render(request,'inter_milan.html',dataS)

def Team_Based_juventus(request):

    df = pd.read_csv('Atletico Madrid.csv')

    labels_league_position = df['League Position']
    labels_champions_league = df['Champions League']
    labels_total_goals = df['Total Goals']
    labels_total_bookings = df['Total Bookings']
    df.drop('League Position',axis=1,inplace=True)
    df.drop('Champions League',axis=1,inplace=True)
    df.drop('Total Goals',axis=1,inplace=True)
    df.drop('Total Bookings',axis=1,inplace=True)

    # champions league predictions
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_champions_league, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionChampionsLeague = list(model.predict(X_test))


    # la liga prediction
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_league_position, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionLeague = model.predict(X_test)

    # total goals
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_goals, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionGoals = model.predict(X_test)

    # total bookings
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_bookings, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionBookings = model.predict(X_test)

    dataS = {}
    dataS['Club'] = "Juventus"
    dataS['League_Position'] = int(predictionLeague)
    dataS['Champions_League'] = predictionChampionsLeague[0]
    dataS['Goals'] = int(predictionGoals)
    dataS['Bookings'] = int(predictionBookings)

    return render(request,'juventus.html',dataS)


def Team_Based_ajax(request):

    df = pd.read_csv('Atletico Madrid.csv')

    labels_league_position = df['League Position']
    labels_champions_league = df['Champions League']
    labels_total_goals = df['Total Goals']
    labels_total_bookings = df['Total Bookings']
    df.drop('League Position',axis=1,inplace=True)
    df.drop('Champions League',axis=1,inplace=True)
    df.drop('Total Goals',axis=1,inplace=True)
    df.drop('Total Bookings',axis=1,inplace=True)

    # champions league predictions
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_champions_league, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionChampionsLeague = list(model.predict(X_test))


    # la liga prediction
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_league_position, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionLeague = model.predict(X_test)

    # total goals
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_goals, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionGoals = model.predict(X_test)

    # total bookings
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_bookings, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionBookings = model.predict(X_test)

    dataS = {}
    dataS['Club'] = "Ajax"
    dataS['League_Position'] = int(predictionLeague)
    dataS['Champions_League'] = predictionChampionsLeague[0]
    dataS['Goals'] = int(predictionGoals)
    dataS['Bookings'] = int(predictionBookings)

    return render(request,'ajax.html',dataS)    


def Team_Based_feyenoord(request):

    df = pd.read_csv('Atletico Madrid.csv')

    labels_league_position = df['League Position']
    labels_champions_league = df['Champions League']
    labels_total_goals = df['Total Goals']
    labels_total_bookings = df['Total Bookings']
    df.drop('League Position',axis=1,inplace=True)
    df.drop('Champions League',axis=1,inplace=True)
    df.drop('Total Goals',axis=1,inplace=True)
    df.drop('Total Bookings',axis=1,inplace=True)

    # champions league predictions
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_champions_league, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionChampionsLeague = list(model.predict(X_test))


    # la liga prediction
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_league_position, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionLeague = model.predict(X_test)

    # total goals
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_goals, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionGoals = model.predict(X_test)

    # total bookings
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_bookings, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionBookings = model.predict(X_test)

    dataS = {}
    dataS['Club'] = "Feyenoord"
    dataS['League_Position'] = int(predictionLeague)
    dataS['Champions_League'] = predictionChampionsLeague[0]
    dataS['Goals'] = int(predictionGoals)
    dataS['Bookings'] = int(predictionBookings)

    return render(request,'feyenoord.html',dataS)


def Team_Based_psv(request):

    df = pd.read_csv('Atletico Madrid.csv')

    labels_league_position = df['League Position']
    labels_champions_league = df['Champions League']
    labels_total_goals = df['Total Goals']
    labels_total_bookings = df['Total Bookings']
    df.drop('League Position',axis=1,inplace=True)
    df.drop('Champions League',axis=1,inplace=True)
    df.drop('Total Goals',axis=1,inplace=True)
    df.drop('Total Bookings',axis=1,inplace=True)

    # champions league predictions
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_champions_league, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionChampionsLeague = list(model.predict(X_test))


    # la liga prediction
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_league_position, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionLeague = model.predict(X_test)

    # total goals
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_goals, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionGoals = model.predict(X_test)

    # total bookings
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_bookings, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionBookings = model.predict(X_test)

    dataS = {}
    dataS['Club'] = "PSV"
    dataS['League_Position'] = int(predictionLeague)
    dataS['Champions_League'] = predictionChampionsLeague[0]
    dataS['Goals'] = int(predictionGoals)
    dataS['Bookings'] = int(predictionBookings)

    return render(request,'psv.html',dataS)

def what_is_deepshot(request):

    return render(request,'what_is_deepshot.html')

def netherlands(request):
     return render(request,'netherlands.html')

def argentina(request):
     return render(request,'argentina.html')

def cameroon(request):
     return render(request,'cameroon.html')

def croatia(request):
     return render(request,'croatia.html')

def japan(request):
     return render(request,'japan.html')

def man_u(request):
     return render(request,'man_u.html')

def morocco(request):
     return render(request,'morocco.html')

def tot(request):
     return render(request,'tot.html')

def wales(request):
     return render(request,'wales.html')




def maps(request):

    df = pd.read_csv('Atletico Madrid.csv')

    labels_league_position = df['League Position']
    labels_champions_league = df['Champions League']
    labels_total_goals = df['Total Goals']
    labels_total_bookings = df['Total Bookings']
    df.drop('League Position',axis=1,inplace=True)
    df.drop('Champions League',axis=1,inplace=True)
    df.drop('Total Goals',axis=1,inplace=True)
    df.drop('Total Bookings',axis=1,inplace=True)

    # champions league predictions
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_champions_league, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionChampionsLeague = list(model.predict(X_test))


    # la liga prediction
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_league_position, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionLeague = model.predict(X_test)

    # total goals
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_goals, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionGoals = model.predict(X_test)

    # total bookings
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_bookings, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionBookings = model.predict(X_test)

    dataS = {}
    dataS['Club'] = "Manchester City"
    dataS['League_Position'] = int(predictionLeague)
    dataS['Champions_League'] = predictionChampionsLeague[0]
    dataS['Goals'] = int(predictionGoals)
    dataS['Bookings'] = int(predictionBookings)

    return render(request,'maps.html',dataS)


def Team_Based_chelsea(request):

    df = pd.read_csv('Atletico Madrid.csv')

    labels_league_position = df['League Position']
    labels_champions_league = df['Champions League']
    labels_total_goals = df['Total Goals']
    labels_total_bookings = df['Total Bookings']
    df.drop('League Position',axis=1,inplace=True)
    df.drop('Champions League',axis=1,inplace=True)
    df.drop('Total Goals',axis=1,inplace=True)
    df.drop('Total Bookings',axis=1,inplace=True)

    # champions league predictions
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_champions_league, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionChampionsLeague = list(model.predict(X_test))


    # la liga prediction
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_league_position, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionLeague = model.predict(X_test)

    # total goals
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_goals, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionGoals = model.predict(X_test)

    # total bookings
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_bookings, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionBookings = model.predict(X_test)

    dataS = {}
    dataS['Club'] = "Chelsea"
    dataS['League_Position'] = int(predictionLeague)
    dataS['Champions_League'] = predictionChampionsLeague[0]
    dataS['Goals'] = int(predictionGoals)
    dataS['Bookings'] = int(predictionBookings)

    return render(request,'chelsea.html',dataS)


def Team_Based_liverpool(request):

    df = pd.read_csv('Atletico Madrid.csv')

    labels_league_position = df['League Position']
    labels_champions_league = df['Champions League']
    labels_total_goals = df['Total Goals']
    labels_total_bookings = df['Total Bookings']
    df.drop('League Position',axis=1,inplace=True)
    df.drop('Champions League',axis=1,inplace=True)
    df.drop('Total Goals',axis=1,inplace=True)
    df.drop('Total Bookings',axis=1,inplace=True)

    # champions league predictions
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_champions_league, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionChampionsLeague = list(model.predict(X_test))


    # la liga prediction
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_league_position, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionLeague = model.predict(X_test)

    # total goals
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_goals, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionGoals = model.predict(X_test)

    # total bookings
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_bookings, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionBookings = model.predict(X_test)

    dataS = {}
    dataS['Club'] = "Liverpool"
    dataS['League_Position'] = int(predictionLeague)
    dataS['Champions_League'] = predictionChampionsLeague[0]
    dataS['Goals'] = int(predictionGoals)
    dataS['Bookings'] = int(predictionBookings)

    return render(request,'liverpool.html',dataS)    


def Team_Based_barcelona(request):

    df = pd.read_csv('Barcelona.csv')

    labels_league_position = df['League Position']
    labels_champions_league = df['Champions League']
    labels_total_goals = df['Total Goals']
    labels_total_bookings = df['Total Bookings']
    df.drop('League Position',axis=1,inplace=True)
    df.drop('Champions League',axis=1,inplace=True)
    df.drop('Total Goals',axis=1,inplace=True)
    df.drop('Total Bookings',axis=1,inplace=True)

    # champions league predictions
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_champions_league, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionChampionsLeague = list(model.predict(X_test))


    # la liga prediction
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_league_position, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionLeague = model.predict(X_test)

    # total goals
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_goals, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionGoals = model.predict(X_test)

    # total bookings
    X_train, X_test, Y_train, Y_test = train_test_split(df, labels_total_bookings, test_size=0.05, random_state=10)
    model = RandomForestClassifier(max_depth=1000,random_state=1)
    model.fit(X_train,Y_train)
    predictionBookings = model.predict(X_test)

    dataS = {}
    dataS['Club'] = "Barcelona"
    dataS['League_Position'] = int(predictionLeague)
    dataS['Champions_League'] = predictionChampionsLeague[0]
    dataS['Goals'] = int(predictionGoals)
    dataS['Bookings'] = int(predictionBookings)

    return render(request,'barcelona.html',dataS)
