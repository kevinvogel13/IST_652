# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:56:21 2020

@author: kevin
"""

#########################################
#
# load packages
#
#########################################

import os
import csv
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import preprocessor as p
import re #regular expression
from textblob import TextBlob
import GetOldTweets3 as got
from nltk.tokenize import TweetTokenizer


#########################################
#
# change some sweet, sweet settings
#
#########################################

# make pandas show us the gooooooods
# pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)

#########################################
#
# load data
#
#########################################

# kevin's directory
os.chdir("C:\\Users\\kevin\\Syracuse University\\IST 652 - Scripting - General\\Project stuff")

# nina's directory
# os.chdir("C:\\Users\\16196\\OneDrive\Documents\\SYRACUSE\\IST 652\\Final_Project")

# read the csv into pandas data frame
df = pd.read_csv("NFL.csv") 

# make a list of team names
TEAMS = ('ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET',
         'GB', 'HOU', 'IND', 'JAC', 'KC', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO',
         'NYG', 'NYJ', 'OAK', 'PHI', 'PIT', 'SD', 'SEA', 'SF', 'STL', 'TB', 'TEN', 'WAS')

# make a list of whether or not the played their home games in a dome
DOME = ('Yes', 'Yes', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 
            'Yes', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'No', 
            'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No')

# put those lists into a df
dfStad = pd.DataFrame({'Team': TEAMS, 'Dome': DOME})

#########################################
#
# data munging and merging
#
#########################################

# Replace JAX to JAC to remain consistent with the rest of the dataset
df = df.replace(to_replace = "JAX", value = "JAC")

# Confirm that there are no more "JAX" entries in the data
sum(df.isin(['JAX']).any())

# Joins the two datasets together so that the dome info corresponds w/the home team
df = pd.merge(df, dfStad, left_on ='home_team', right_on = 'Team', how ='left') 

# Filter for everything past 2010-04-0
df=df[(df['game_date'] > '2010-04-01')]

# make a list of all of the columns
cols = df.columns.tolist()
for col in cols:
    print(col)
    
#########################################
#
# should the seahawks have run the ball or passed?
#
#########################################   

# make a smaller data set to show yards gained by play type
df_toteff = df[['play_type', 'yards_gained', 'yardline_100']]
df_toteff = df_toteff[(
    (df_toteff['play_type']=='pass') | 
    (df_toteff['play_type']=='run')
    )] 
df_toteff = df_toteff.sort_values(by='play_type', ascending=False)

# make a boxplot for total efficiency by run or pass
sns.boxplot(x = 'play_type', y = 'yards_gained', data = df_toteff).set_title(
    'Yards gained per play by play type (not adjusted for field position)')
plt.show()

# passing is more volatile than running
# running has a reduced chance of a negative play
# this is an old school way of thinking but darrell royal once said that, "3 things can 
# happen on any passing play and two of them are bad"

# here is the passing volatility shown on a histogram
df_toteffpass = df_toteff[df_toteff['play_type']=='pass']
sns.distplot(df_toteffpass['yards_gained']).set_title(
    'Yards gained by passing (not adjusted for field position)')
plt.show()

df_toteffrun = df_toteff[df_toteff['play_type']=='run']
sns.distplot(df_toteffrun['yards_gained']).set_title(
    'Yards gained by rushing (not adjusted for field position)')
plt.show()


# but what does all of this look like at the goal line?
# what should they do there? how is it different?

# Create df_goal data frame with all plays from the 1 yard line
df_goal = df[['play_type', 'run_location', 'pass_location', 'yards_gained', 'yardline_100', 'epa']]
df_goal = df_goal[df_goal['yardline_100']==1] 
df_goal = df_goal[(
    (df_goal['play_type']=='pass') | 
    (df_goal['play_type']=='run')
    )] 

# show the difference in expected points 
sns.boxplot(x = 'play_type', y = 'epa', data = df_goal).set_title(
    'Expected Points Added (EPA) by play type from the 1')
plt.show()

# again, it looks like passing the ball is more volatile
# passing, because it requires the QB to be behind the line of scrimmage takes on the inherent
# risk of a negative play in the form of a sack if pass coverage breaks down
# passing does add more expected points per attempt but only marginally

# given that the seahawks had at least 2 plays to get the touchdown, it seems more likely that 
# a run play followed by a timeout if unsuccessful would have been a better decision here

# which way should they have run it though?

# create a df for running plays from the 1 with their outcome
df_goal_run = df_goal[df_goal['play_type']=='run'] 
df_goal_run = df_goal_run[['run_location', 'yards_gained', 'epa']]
df_goal_run = df_goal_run.sort_values(by='run_location', ascending=True)
# show the difference in expected points 
sns.boxplot(x = 'run_location', y = 'epa', data = df_goal_run).set_title(
    'Expected Points Added (EPA) by run location from the 1')
plt.show()

# there is no globally optimal direction to run the ball from the 1
# we can take a look at how the seahawks ran the ball in 2015 on the whole
# but they only had a few attempts from the 1 all year so small sample size would
# preclude us from getting anything out of that
# just for fun, let's see if they were better rushing to the left, right, or middle that year

# create a df for seattle running plays that season
df_sea_run = df[['play_type', 'run_location', 'yards_gained', 'posteam', 'game_date', 'yardline_100', 'epa']]
df_sea_run = df_sea_run[df_sea_run['play_type']=='run'] 
df_sea_run = df_sea_run[df_sea_run['posteam']=='SEA'] 
df_sea_run = df_sea_run[(df_sea_run['game_date'] > '2014-04-01')]
df_sea_run = df_sea_run[(df_sea_run['game_date'] < '2015-04-01')]
df_sea_run = df_sea_run.sort_values(by='run_location', ascending=True)
# show how they did on the year 
sns.boxplot(x = 'run_location', y = 'yards_gained', data = df_sea_run).set_title(
    'Yards gained by run location (not adjusted for field position)')
plt.show()
# the seahawks were better at running the ball to the right, only slightly
# let's see how they did inside the 10

# create a df for plays inside the 10 that year from the previous data set
df_sea_run_10 = df_sea_run[df_sea_run['yardline_100'] <= 10] 
# there were only 41 plays so we may wind up with a bad distribution
# show how they did on the year from inside the 10
sns.boxplot(x = 'run_location', y = 'yards_gained', data = df_sea_run_10).set_title(
    'Yards gained by run location on plays inside the 10')
plt.show()
# again, they were better inside the 10 going to the right
# given that we know that they passed it, did they at least pass it the correct direction?

# create a df for passing plays from the 1 with their outcome
df_goal_pass = df_goal[df_goal['play_type']=='pass'] 
df_goal_pass = df_goal_pass[['pass_location', 'yards_gained', 'epa']]
# show the difference in expected points 
sns.boxplot(x = 'pass_location', y = 'epa', data = df_goal_pass).set_title(
    'Expected Points Added (EPA) by pass location from the 1')
plt.show()

# passing to the right or middle is optimal if you were to pass it from the 1
# this is likely due to the higher density of players and tighter windows to have to 
# fit the ball into. most NFL QB's are right handed so throwing to the right means that 
# the football can take a better angle on outside throws to WR's
# they threw the ball to the middle/right on a quick slant -- likely the most optimal throw
# next to a fade into the right corner

# they should have run the ball from the 1

# but how long would that have taken? would they have had to call a timeout immediately or could they run another play?

# put those lists into a df
dfStad = pd.DataFrame({'Team': TEAMS, 'Dome': DOME})
df_diff = pd.DataFrame({'half_seconds_remaining': df['half_seconds_remaining'], 'play_type': df['play_type'], 'time_diff': df['game_seconds_remaining'].diff(periods=-1)})
# filter for plays in the final 2 minutes 
# this scenario closely matches the seahawks scenario
df_diff = df_diff[df_diff['half_seconds_remaining'] <= 120]
df_diff = df_diff[df_diff['time_diff'] > 0]
df_diff = df_diff[df_diff['time_diff'] <= 40]
df_diff = df_diff[(
    (df_diff['play_type']=='pass') | 
    (df_diff['play_type']=='run')
    )] 
sns.boxplot(x = 'play_type', y = 'time_diff', data = df_diff).set_title(
    'Time per play in the final two minutes of a half')
plt.show()

# running the ball does take more time than passing because an incomplete (failed) pass stops the clock
# immediately, however the median time for a run play in the final two minutes is only 15 seconds. they could have
# QB sneaked the ball with Russell Wilson and had time for two more plays as the average QB sneak takes less than an 
# average run to the RB and is lower risk as you don't take the ball backward before handing it off and going forward

# with 66 seconds left in the game the clock was stopped and the seahawks had the ball on the 5 with one timeout remaining
# they ran the ball to the left and got down the the 1
# situation: 2nd and goal from the 1 with 60 seconds left and the 40 second clock (play clock) ticking
# they run the clock all the way down to 26 seconds when they snap it
# at this point there were two options
    # run
    # pass
# the play call was pass and should have been run (prefereably QB sneak)
# if they had run it and not gotten the ball in the end zone they then could have called their final timeout
# leaving them with 15 seconds and 2 downs (at most) to score
# with 15 seconds and 0 timeouts, there are two options
    # pass the ball in a 
    # run the ball likely against their pass defense 
# the optimal decision is NOW to pass the ball being that an incomplete pass stops the clock
# it should have been a fade or other low percentage interception play instead of a slant however
# then, if you still have not scored and it is 4th and 1, you give it to the best power back in football
# and let him work...but Pete Carroll is a clown
# in 2015 sports illustrated wrote an article referencing how NFL teams were treating analytics
# https://www.espn.com/espn/feature/story/_/id/12331388/the-great-analytics-rankings
# the seahawks were referenced as being "one foot in" depsite the teams owner being no other than 
# Paul Allen, the co-founder of Microsoft

# so, let's see how the fans responded on twitter to this boneheaded call 


#######################################################
#
#
# set the working directory and get data
#
#
#######################################################

# set wd
os. getcwd()
path_kevin = ('C:\\Users\\kevin\\Syracuse University\\IST 652 - Scripting - General\\Project stuff')
# path_nina = ('C:\\Users\\16196\\Documents\\SYRACUSE\\IST 652\\Final_Project')
os.chdir(path_kevin)
# os.chdir(path_nina)

# parameters for the tweeter search
text_query = 'Marshawn Lynch'
since_date = '2015-02-02'
until_date = '2015-02-03'
count = 15000

# Creation of query object
tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query).setSince(since_date).setUntil(until_date).setMaxTweets(count)

# Creation of list that contains all tweets
tweets = got.manager.TweetManager.getTweets(tweetCriteria)


#######################################################
#
#
# write the unprocessed tweets before processing
#
#
#######################################################

tweetlist = [] 
for tweet in tweets:
    temp = tweet.text
    tweetlist.append(temp)
    
# create the df
df = pd.DataFrame(tweetlist, columns = ['tweet']) 

# write the tweets to csv before manipulating them
df.to_csv('tweets.csv', encoding="utf8", index = False)

#######################################################
#
#
# process the tweets
#
#
#######################################################

tweetlist = [] 
for tweet in tweets:
    temp = tweet.text
    temp = p.clean(temp)
    # temp = clean_tweets(temp)
    blob = TextBlob(temp)
    temp = TweetTokenizer().tokenize(temp)
    temp = [w.lower() for w in temp]
    Sentiment = blob.sentiment     
    polarity = Sentiment.polarity
    subjectivity = Sentiment.subjectivity
    tweetlist.append([temp, polarity, subjectivity])
    
# create the df
df = pd.DataFrame(tweetlist, columns =['tweet', 'polarity', 'subjectivity']) 

# describe the df
df[['polarity', 'subjectivity']].describe()

# write the data to csv2
df.to_csv('tweets2.csv', encoding="utf8", index = False)

# We are surprised that the sentiment isn't worse than it is 
# actually, it skews just positive
# this may be an issue with the sentiment analysis, specifically negation
# or the fact that people use peculiar language on twitter
# or it could be that people just weren't that upset with one of the 
# stupidest play calls EVER as we are :-)
 

# # read the new data back in 
# df = pd.read_csv('C:\\Users\\16196\\Documents\\SYRACUSE\\IST 652\\Final_Project')

#######################################################
#
#
# below is the data from the first homework that we used
# to familiarize ourselves with the data set
#
#
#######################################################

# QUESTION #1    
 
# Create df_run data frame 
df_run = df[['play_type', 'run_location', yards_gained']]

# Filter df for only running plays    
df_run = df_run[df_run['play_type']=='run'] 

# make a pivot
df_run_pvt = df_run.pivot_table(index='run_location', values='yards_gained', aggfunc='mean')
df_run_pvt

########################################################    

# QUESTION #2    
 
# Create df_kick data frame 
df_kick = df[['play_type', 'field_goal_result','kick_distance',]]
# Filter df for only field goal plays    
df_kick = df_kick[df_kick['play_type']=='field_goal'] 
df_kick = df_kick.replace(to_replace = "made", value = 1)
df_kick = df_kick.replace(to_replace = "missed", value = 0)
df_kick = df_kick.replace(to_replace = "made", value = 0)
df_kick = df_kick.replace(to_replace = "blocked", value = 0)
df_kick.astype({'field_goal_result': 'float64'}).dtypes
df_kick_pvt = df_kick.pivot_table(index='kick_distance', values='field_goal_result', aggfunc='mean')

df_kick_pvt['kick_dist'] = df_kick_pvt.index
df_kick_pvt.plot(kind='scatter', x='kick_dist', y='field_goal_result')

# # kick 20 yards or less
# df_kick20 = df_kick[(df_kick['kick_distance']<21)]
# df_kick20 = df_kick20['field_goal_result'].value_counts(normalize=True)
# 100*df_kick20    

# # kick between 21 & 40 yard
# df_kick40 = df_kick[(df_kick['kick_distance']>20) & (df_kick['kick_distance']<41)]
# df_kick40 = df_kick40['field_goal_result'].value_counts(normalize=True)
# 100*df_kick40    

# # kick between 41 & 60 yard
# df_kick60 = df_kick[(df_kick['kick_distance']>40) & (df_kick['kick_distance']<61)]
# df_kick60 = df_kick60['field_goal_result'].value_counts(normalize=True)
# 100*df_kick60  

# # kick between 61 & 80 yard
# df_kick80 = df_kick[(df_kick['kick_distance']>60) & (df_kick['kick_distance']<81)]
# df_kick80 = df_kick80['field_goal_result'].value_counts(normalize=True)
# 100*df_kick80 

# The above resulting tables provide bases for analysis in paper

########################################################    

# QUESTION #3    
 
# Create df_extra data frame   
df_extra = df[['game_date','play_type','extra_point_result']]

# Filter for new rule
df_extra_new = df_extra[(df_extra['game_date'] > '2015-04-01')]
df_extra_new=df_extra_new['extra_point_result'].value_counts(normalize=True)
df_extra_new

# Filter for old rule
df_extra_old = df_extra[(df_extra['game_date'] < '2015-04-01')]
df_extra_old=df_extra_old['extra_point_result'].value_counts(normalize=True)
df_extra_old

# What is the variance? 5.2% less likely to make the extra point with new rules
100*(df_extra_new - df_extra_old)


########################################################    

# QUESTION #4    
 
# Create df_dome data frame
df_dome = df[['play_type', 'field_goal_result','Dome']]    

# Filter df for only field goal plays    
df_dome = df_dome[df_dome['play_type']=='field_goal']

# Aggregate
df_dome.groupby(["Dome","field_goal_result"]).count()

# Filter df for only field goal plays in a Dome  
df_domeY = df_dome[df_dome['Dome']=='Yes']
df_domeY=df_domeY['field_goal_result'].value_counts(normalize=True)
df_domeY

# Filter df for only field goal plays not in a Dome   
df_domeN = df_dome[df_dome['Dome']=='No']
df_domeN=df_domeN['field_goal_result'].value_counts(normalize=True)
df_domeN

# What is the variance? .7% more likely to make it in a dome
100*(df_domeY - df_domeN)
###########
#
# END
#
##########


