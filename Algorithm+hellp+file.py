
# coding: utf-8

# # 1 )Liblaries
# 

# In[148]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data
from pandas.tseries.offsets import *
from yahoo_finance import Share
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
get_ipython().magic('matplotlib inline')


# # 2) Loading data

# In[149]:

# pd.read_csv(xyz.csv)#loading data from your computer
google = data.DataReader(name="GOOG",data_source="google",
                start = dt.date(2000,2,2), end = dt.datetime.now()) # https://pandas-datareader.readthedocs.io/en/latest/
companies = ["MSFT", "GOOG","AAPL","YHOO","AMZN"]#Company name
start = "2010-01-01"
end = "2016-12-31"
p= data.DataReader(name = companies, data_source = "google",
               start= start , end = end ) # Dowland data from Google for 5 companies


# In[150]:

# Downloading data from webside and import do excel
URL = "https://data.cityofnewyork.us/api/views/25th-nujf/rows.csv"
baby_names = pd.read_csv(URL)
baby_names.head(3)


# # 3) Function

# In[151]:

google.head()#print 5 first items
google.tail()# print 5 last items
google.info()# show simple information about data
len(google)# show number of data
google.index
google.ndim
google.shape# numer row ale columns


# In[152]:

google.dropna()#Return object with labels on given axis omitted where alternately any or all of the data are missing


# In[153]:

google["Close"].mul(0.3)# multiplication
google["Close"].add(52)#add
google["Close"].sub(5000000)# submit
google["Close"].div(500)#divade 
google["Close"].value_counts()# how much data a particular type
google.fillna(0)# in place of missing data inserts 0
google["Open"].nunique()#given how many kinds of variables
google.sort_values("Close", ascending = True)#sorts the data 
google["close rank"]=google["Close"].rank(ascending=True).astype("int")# give ranking number 


# In[154]:

#Several variables satisfies the specified condition
mask1=google["Open"]< 123456
mask2=google["Close"]>123
google[mask1&mask2]


# In[155]:

#Or
mask1=google["Open"]< 123456
mask2=google["Close"]>123
google[mask1|mask2]


# In[156]:

#You check whether the column is a variable returns True / False
google["Close"].isin([0])
#If there is a lack of returns True False
google["Close"].isnull()
#Values between
google[google["Close"].between(100,3000)]
#Show duplication
google[google["Close"].duplicated()]
#It shows the unique variables
google["Close"].unique()
len(google["Close"].unique())


# In[157]:

google.nlargest(3, columns = "Close")# the largest close price
google.nsmallest(3, columns = "Close") # th smallest close price


# In[158]:

someday = dt.date(2017,2,15)#date
str(someday)
sometime= dt.datetime(2017,2,12,8,25,23)#date with time
str(sometime)


# In[159]:

pd.Timestamp("2017-03-31")
pd.Timestamp("2017/03/31")
pd.Timestamp("2017, 03, 31")


# In[160]:

dates = ["2016-01-02","2017-03-15","2012-05-28"]
dtIndex= pd.DatetimeIndex(dates)

values=[100,200,300]
pd.Series(values, index = dtIndex)
pd.to_datetime([84509],unit = "m")


# In[161]:

animal_list = ['cats', 'dogs', 'dogs', 'dogs', 'lizards', 'sponges', 'cows', 'bats', 'sponges']
animal_set = set(animal_list)
animal_set # Removes all extra instances from the list


# In[162]:

'cats' in animal_set # Here we check for membership using the `in` keyword.


# In[163]:

my_dict = {"High Fantasy": ["Wheel of Time", "Lord of the Rings"], 
           "Sci-fi": ["Book of the New Sun", "Neuromancer", "Snow Crash"],
           "Weird Fiction": ["At the Mountains of Madness", "The House on the Borderland"]}


# In[164]:

my_dict["Sci-fi"]


# In[165]:

'insert %s here' % 5


# In[166]:

if "Condition": 
    # This block of code will execute because the string is non-empty
    # Everything on these indented lines
     True
else:
    # So if the condition that we examined with if is in fact False
    # This block of code will execute INSTEAD of the first block of code
    # Everything on these indented lines
     False


# # If-statements (zmień)

# An else statement can be combined with an if statement. An else statement contains the block of code that executes if the conditional expression in the if statement resolves to 0 or a FALSE value.
# The else statement is an optional statement and there could be at most only one else statement following if 

# 

# In[167]:

i = 10
if i % 2 == 0:
    if i % 3 == 0:
        print ('i is divisible by both 2 and 3! Wow!')
    elif i % 5 == 0:
        print ('i is divisible by both 2 and 5! Wow!')
    else:
        print ('i is divisible by 2, but not 3 or 5. Meh.')
else:
    print ('I guess that i is an odd number. Boring.')


# In[168]:

i = 5
j = 12
if i < 10 and j > 11:
    print ('{0} is less than 10 and {1} is greater than 11! How novel and interesting!'.format(i, j))


# In[ ]:




# # Loop Structures
# 

# In general, statements are executed sequentially: The first statement in a function is executed first, followed by the second, and so on. There may be a situation when you need to execute a block of code several number of times.
# Programming languages provide various control structures that allow for more complicated execution paths.
# A loop statement allows us to execute a statement or group of statements multiple times. The following diagram illustrates a loop statement 

# In[169]:

i = 5
while i > 0: # We can write this as 'while i:' because 0 is False!
    i -= 1
    print ('I am looping! {0} more to go!'.format(i))


# In[170]:

my_list = {'cats', 'dogs', 'lizards', 'cows', 'bats', 'sponges', 'humans'} # Lists all the animals in the world
mammal_list = {'cats', 'dogs', 'cows', 'bats', 'humans'} # Lists all the mammals in the world
my_new_list = set()
for animal in my_list:
    if animal in mammal_list:
        # This adds any animal that is both in my_list and mammal_list to my_new_list
        my_new_list.add(animal)
        
print (my_new_list)


# In[171]:

def multiply_by_five(x):
    """ Multiplies an input number by 5 """
    return x * 5

n = 4
print (n)
print (multiply_by_five(n))


# In[172]:

def calculate_area(length, width):
    """ Calculates the area of a rectangle """
    return length * width


# In[173]:

l = 5
w = 10
print ('Area: ', calculate_area(l, w))
print ('Length: ', l)
print ('Width: ', w)


# In[174]:

def sum_values(*args):
    sum_val = 0
    for i in args:
        sum_val += i
    return sum_val


# In[175]:

print (sum_values(1, 2, 3))
print (sum_values(10, 20, 30, 40, 50))
print (sum_values(4, 2, 5, 1, 10, 249, 25, 24, 13, 6, 4))


# In[ ]:




# # 4) Panels with p (companies)

# In[ ]:




# In[176]:

p
p.items
p.major_axis
p.minor_axis
p.axes
p.ndim
p.dtypes
p.shape
p.size
p.values


# In[177]:

p["Close"]# Print data like: Open High, Low, Close, Volume


# In[178]:

p.loc["Open","2010-01-01" : , "GOOG"]#Print Open price start from 2010-01-01 Google     


# In[179]:

p.iloc[0,1,1]# First argumate it is number of "Colum" secend "Name index" and fird is number of " Row" 


# In[180]:

df = p.to_frame()#Data Frame
df.to_panel()# Information


# In[181]:

df


# In[182]:

p.major_xs("2016-09-06")# Print data from 2016-09-06
p.minor_xs("MSFT")# Print data for MSFT


# In[183]:

p2= p.transpose(2,1,0)
p2["GOOG"]
p2.major_xs("2010-01-04")


# In[184]:

p2


# In[185]:

p2["GOOG"]


# #  5)Google stock price

# In[186]:

google.index + pd.tseries.offsets.MonthEnd()# przed importem bibliotekipandas.tseries.offsets
google.index + MonthEnd()# po imporcie biblioteki
google.index - BMonthEnd()# dzien powszedni 
google.index +QuarterEnd()


# In[187]:

google.index + MonthEnd()


# In[188]:

google.index - BMonthEnd()


# In[189]:

google.values
google.columns
google.index[0]
google.axes
google.loc["2014-03-04"]
google.iloc[300]
google.ix["2014-03-04"]
google.loc["2014-03-04":"2015-03-04"]
google.ix["2014-03-04":"2015-03-04"]
birthday = pd.date_range(start="1993-03-27", end = "2017-03-27", freq = pd.DateOffset(years =1))
mask = google.index.isin(birthday)
google[mask]


# In[190]:

someday = google.index[1]#number of day from data frame
someday #primt date and time
someday.day # print number of day
someday.month# print number ofmonth
someday.year# print number of year
someday.weekday()# print number of weekday
someday.is_month_end# print number of month_end
someday.is_month_start# print number of month_start


# In[191]:

google.insert(0,"Day of Week", google.index.weekday)#creates a new column
google.insert(0,"Day of start month", google.index.is_month_start)#creates a new column
google.head(3)# print 3 first items
google[google["Day of start month"]]
google.truncate(before= "2011-02-05",after ="2012-02-05")


# In[ ]:




# In[ ]:




# # 6) Visualization Google price stock

# In[192]:

google2= data.DataReader(name= "GOOG", data_source = "google", start= "2007-01-01", end = "2012-12-31")
google2.head(3)


# In[193]:

google2.plot()


# In[194]:

plt.style.available# chceck available chart style
plt.style.use("classic")
google2.plot(y="Close")


# In[195]:

dailyret = google2["Close"].pct_change().dropna()

plt.figure( figsize=(10, 4),)
plt.plot(dailyret)


# In[196]:

plt.plot(google2["Open"], 'r')
plt.plot(google2["High"], 'w.')
plt.plot(google2["Low"], 'm.')
plt.plot(google2["Close"], 'g.')


# In[ ]:




# In[197]:

'''#def rollingmean(price, n, j):
    y = price[j-n:j]
    output = np.mean(y)
    return output

MAG1 = 50
MAG2 = 100
rol1 = []
rol2 = []
for i in range(MAG1, range(1510)):
    rol1.append(rollingmean(google2, MAG1, i))
for i in range(MAG2, range(1510)):
    rol2.append(rollingmean(google2, MAG2, i))
    
plt.plot(google2, 'r')
plt.plot([MAG1], rol1, 'b')
plt.plot([MAG2], rol2, 'g')
'''


# In[198]:

len(google2)


# In[199]:

def rank_performance(stock_price):
    if stock_price<150:
        return "Poor"
    elif stock_price>150 and stock_price<=200:
        return"Satisfactory"
    else:
        return "Stellar"


# In[200]:

plt.style.use("ggplot")
google2["Close"].apply(rank_performance).value_counts().plot(kind = "bar")


# In[201]:

plt.style.use("ggplot")
google2["Close"].apply(rank_performance).value_counts().plot(kind = "pie", legend = True)


# In[202]:

MAVG = pd.rolling_mean(google2, window=60)
plt.plot(google2.index, google2.values)
plt.plot(MAVG.index, MAVG.values)
plt.ylabel('Price')
plt.legend(['google2', '60-day MAVG']);


# In[203]:

add_returns = google2.diff()[1:]
mult_returns = google2.pct_change()[1:]


# In[204]:

mult_returns.plot();


# In[205]:

plt.hist(google2["Close"], bins=20)


# In[206]:

google2.describe()


# # Returns Histogram
# In finance rarely will we look at the distribution of prices. The reason for this is that prices are non-stationary and move around a lot. For more info on non-stationarity please see this lecture. Instead we will use daily returns. Let's try that now.
# In [6]:
# 

# In[207]:

# Remove the first element because percent change from nothing to something is NaN
R = google2["Close"].pct_change()[1:]

# Plot a histogram using 20 bins
plt.hist(R, bins=50)
plt.xlabel('Return')
plt.ylabel('Number of Days Observed')
plt.title('Frequency Distribution of google Returns, 2014');


# # Cumulative Histogram (Discrete Estimated CDF)
# An alternative way to display the data would be using a cumulative distribution function, in which the height of a bar represents the number of observations that lie in that bin or in one of the previous ones. This graph is always nondecreasing since you cannot have a negative number of observations. The choice of graph depends on the information you are interested in

# In[208]:

# Remove the first element because percent change from nothing to something is NaN
R = google2["Close"].pct_change()[1:]

# Plot a histogram using 20 bins
plt.hist(R, bins=50, cumulative=True)
plt.xlabel('Return')
plt.ylabel('Number of Days Observed')
plt.title('Cumulative Distribution of Google Returns');


# # Scatter plot
# A scatter plot is useful for visualizing the relationship between two data sets. We use two data sets which have some sort of correspondence, such as the date on which the measurement was taken. Each point represents two corresponding values from the two data sets. However, we don't plot the date that the measurements were taken on.

# In[209]:

plt.scatter(p.loc["Close","2010-01-01" : , "AAPL"], p.loc["Close","2010-01-01" : , "GOOG"],c=["red","blue"])
plt.xlabel('AAPL')
plt.ylabel('GOOG')
plt.title('Daily Prices AAPL, GOOG');


# In[210]:

R_aapl = p.loc["Close","2010-01-01" : , "AAPL"].pct_change()[1:]
R_goog = p.loc["Close","2010-01-01" : , "GOOG"].pct_change()[1:]

plt.scatter(R_aapl, R_goog, c=["red","blue"])
plt.xlabel('AAPL')
plt.ylabel('GOOG')
plt.title('Daily Returns in 2014');


# # 7) Data from website

# In[211]:

girls = baby_names[baby_names["GNDR"]=="FEAMALE"]
boys = baby_names[baby_names["GNDR"]=="MALE"]


# In[212]:

excel_file = pd.ExcelWriter("Baby Names.xlsx")
girls.to_excel(excel_file, sheet_name = "Girls", index = False)
boys.to_excel(excel_file, sheet_name = "Boys", index = False,columns = ["GNDR","NM","CNT"])
excel_file.save()


# # 7) Simple statistics

# # Mode
# The mode is the most frequently occuring value in a data set. It can be applied to non-numerical data, unlike the mean and the median. One situation in which it is useful is for data whose possible values are independent. For example, in the outcomes of a weighted die, coming up 6 often does not mean it is likely to come up 5; so knowing that the data set has a mode of 6 is more useful than knowing it has a mean of 4.5.

# In[225]:

# We'll use these two data sets as examples
x1 = [1, 2, 2, 3, 4, 5, 5, 7]
x2 = x1 + [100]


# In[221]:

# We'll use these two data sets as examples
x1 = [1, 2, 2, 3, 4, 5, 5, 7]
x2 = x1 + [100]

# Scipy has a built-in mode function, but it will return exactly one value
# even if two values occur the same number of times, or if no value appears more than once
print ('One mode of x1:', stats.mode(x1)[0][0])

# So we will write our own
def mode(l):
    # Count the number of times each element appears in the list
    counts = {}
    for e in l:
        if e in counts:
            counts[e] += 1
        else:
            counts[e] = 1
            
    # Return the elements that appear the most times
    maxcount = 0
    modes = {}
    for (key, value) in counts():
        if value > maxcount:
            maxcount = value
            modes = {key}
        elif value == maxcount:
            modes.add(key)
            
    if maxcount > 1 or len(l) == 1:
        return list(modes)
    return 'No mode'
    
print ('All of the modes of x1:', mode(x1))


# # Geometric mean
# While the arithmetic mean averages using addition, the geometric mean uses multiplication:
# $$ G = \sqrt[n]{X_1X_1\ldots X_n} $$
# 
# for observations $X_i \geq 0$. We can also rewrite it as an arithmetic mean using logarithms:
# $$ \ln G = \frac{\sum_{i=1}^n \ln X_i}{n} $$
# 
# The geometric mean is always less than or equal to the arithmetic mean (when working with nonnegative observations), with equality only when all of the observations are the same.

# In[228]:

# Use scipy's gmean function to compute the geometric mean
print ('Geometric mean of x1:', stats.gmean(x1))
print ('Geometric mean of x2:', stats.gmean(x2))


# What if we want to compute the geometric mean when we have negative observations? This problem is easy to solve in the case of asset returns, where our values are always at least $-1$. We can add 1 to a return $R_t$ to get $1 + R_t$, which is the ratio of the price of the asset for two consecutive periods (as opposed to the percent change between the prices, $R_t$). This quantity will always be nonnegative. So we can compute the geometric mean return,
# $$ R_G = \sqrt[T]{(1 + R_1)\ldots (1 + R_T)} - 1$$

# In[230]:

# Add 1 to every value in the returns array and then compute R_G
ratios = returns + np.ones(len(returns))
R_G = stats.gmean(ratios) - 1
print ('Geometric mean of returns:', R_G)


# # Harmonic mean
# 
# The harmonic mean is less commonly used than the other types of means. It is defined as
# $$ H = \frac{n}{\sum_{i=1}^n \frac{1}{X_i}} $$
# 
# As with the geometric mean, we can rewrite the harmonic mean to look like an arithmetic mean. The reciprocal of the harmonic mean is the arithmetic mean of the reciprocals of the observations:
# $$ \frac{1}{H} = \frac{\sum_{i=1}^n \frac{1}{X_i}}{n} $$
# 
# The harmonic mean for nonnegative numbers $X_i$ is always at most the geometric mean (which is at most the arithmetic mean), and they are equal only when all of the observations are equal.

# In[231]:

print ('Harmonic mean of x1:', stats.hmean(x1))
print ('Harmonic mean of x2:', stats.hmean(x2))


# The harmonic mean can be used when the data can be naturally phrased in terms of ratios. For instance, in the dollar-cost averaging strategy, a fixed amount is spent on shares of a stock at regular intervals. The higher the price of the stock, then, the fewer shares an investor following this strategy buys. The average (arithmetic mean) amount they pay for the stock is the harmonic mean of the prices

# # Point Estimates Can Be Deceiving
# 
# Means by nature hide a lot of information, as they collapse entire distributions into one number. As a result often 'point estimates' or metrics that use one number, can disguise large programs in your data. You should be careful to ensure that you are not losing key information by summarizing your data, and you should rarely, if ever, use a mean without also referring to a measure of spread.
# 
# ## Underlying Distribution Can be Wrong
# 
# Even when you are using the right metrics for mean and spread, they can make no sense if your underlying distribution is not what you think it is. For instance, using standard deviation to measure frequency of an event will usually assume normality. Try not to assume distributions unless you have to, in which case you should rigourously check that the data do fit the distribution you are assuming.

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



