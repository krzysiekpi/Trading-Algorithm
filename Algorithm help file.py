
# coding: utf-8

# # 1 )Liblaries
# 

# In[13]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data
from pandas.tseries.offsets import *
from yahoo_finance import Share
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# # 2) Loading data

# In[14]:

# pd.read_csv(xyz.csv)#loading data from your computer
google = data.DataReader(name="GOOG",data_source="google",
                start = dt.date(2000,2,2), end = dt.datetime.now()) # dowland data from Google Fince

companies = ["MSFT", "GOOG","AAPL","YHOO","AMZN"]#Company name
start = "2010-01-01"
end = "2016-12-31"
p= data.DataReader(name = companies, data_source = "google",
               start= start , end = end ) # Dowland data from Google for 5 companies


# In[15]:

# Downloading data from webside and import do excel
URL = "https://data.cityofnewyork.us/api/views/25th-nujf/rows.csv"
baby_names = pd.read_csv(URL)
baby_names.head(3)


# # 3) Function

# In[16]:

google.head()#print 5 first items
google.tail()# print 5 last items
google.info()# show simple information about data
len(google)# show number of data
google.index
google.ndim
google.shape# numer row ale columns


# In[ ]:




# In[17]:

google["Close"].mul(0.3)# multiplication
google["Close"].add(52)#add
google["Close"].sub(5000000)# submit
google["Close"].div(500)#divade 
google["Close"].value_counts()# how much data a particular type
google.fillna(0)# in place of missing data inserts 0
google["Open"].nunique()#given how many kinds of variables
google.sort_values("Close", ascending = True)#sorts the data 
google["close rank"]=google["Close"].rank(ascending=True).astype("int")# give ranking number 


# In[18]:

#Several variables satisfies the specified condition
mask1=google["Open"]< 123456
mask2=google["Close"]>123
google[mask1&mask2]


# In[19]:

#Or
mask1=google["Open"]< 123456
mask2=google["Close"]>123
google[mask1|mask2]


# In[20]:

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


# In[21]:

google.nlargest(3, columns = "Close")# the largest close price
google.nsmallest(3, columns = "Close") # th smallest close price


# In[22]:

someday = dt.date(2017,2,15)#date
str(someday)
sometime= dt.datetime(2017,2,12,8,25,23)#date with time
str(sometime)


# In[23]:

pd.Timestamp("2017-03-31")
pd.Timestamp("2017/03/31")
pd.Timestamp("2017, 03, 31")


# In[24]:

dates = ["2016-01-02","2017-03-15","2012-05-28"]
dtIndex= pd.DatetimeIndex(dates)

values=[100,200,300]
pd.Series(values, index = dtIndex)
pd.to_datetime([84509],unit = "m")


# In[ ]:




# # 4) Panels with p (companies)

# In[25]:

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


# In[26]:

p["Close"]# Print data like: Open High, Low, Close, Volume


# In[27]:

p.loc["Open","2010-01-01" : , "GOOG"]#Print Open price start from 2010-01-01 Google     


# In[28]:

p.iloc[0,1,1]# First argumate it is number of "Colum" secend "Name index" and fird is number of " Row" 


# In[29]:

df = p.to_frame()#Data Frame
df.to_panel()# Information


# In[30]:

df


# In[31]:

p.major_xs("2016-09-06")# Print data from 2016-09-06
p.minor_xs("MSFT")# Print data for MSFT


# In[ ]:




# In[32]:

p2= p.transpose(2,1,0)
p2["GOOG"]
p2.major_xs("2010-01-04")


# In[33]:

p2


# In[34]:

p2["GOOG"]


# #  5)Google stock price

# In[35]:

google.index + pd.tseries.offsets.MonthEnd()# przed importem bibliotekipandas.tseries.offsets
google.index + MonthEnd()# po imporcie biblioteki
google.index - BMonthEnd()# dzien powszedni 
google.index +QuarterEnd()


# In[36]:

google.index + MonthEnd()


# In[37]:

google.index - BMonthEnd()


# In[38]:

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


# In[39]:

someday = google.index[1]#number of day from data frame
someday #primt date and time
someday.day # print number of day
someday.month# print number ofmonth
someday.year# print number of year
someday.weekday()# print number of weekday
someday.is_month_end# print number of month_end
someday.is_month_start# print number of month_start


# In[40]:

google.insert(0,"Day of Week", google.index.weekday)#creates a new column
google.insert(0,"Day of start month", google.index.is_month_start)#creates a new column
google.head(3)# print 3 first items
google[google["Day of start month"]]
google.truncate(before= "2011-02-05",after ="2012-02-05")


# In[ ]:




# # 6) Visualization Google price stock

# In[41]:

google2= data.DataReader(name= "GOOG", data_source = "google", start= "2007-01-01", end = "2012-12-31")
google2.head(3)


# In[42]:

google2.plot()


# In[67]:

plt.style.available# chceck available chart style
plt.style.use("classic")
google2.plot(y="Close")


# In[69]:

dailyret = google2["Close"].pct_change().dropna()

plt.figure( figsize=(10, 4),)
plt.plot(dailyret)


# In[44]:




# In[72]:

plt.plot(google2["Open"], 'r')
plt.plot(google2["High"], 'w.')
plt.plot(google2["Low"], 'm.')
plt.plot(google2["Close"], 'g.')


# In[ ]:




# In[94]:

def rollingmean(price, n, j):
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


# In[88]:

len(google2)


# In[ ]:

def rank_performance(stock_price):
    if stock_price<150:
        return "Poor"
    elif stock_price>150 and stock_price<=200:
        return"Satisfactory"
    else:
        return "Stellar"


# In[ ]:

plt.style.use("ggplot")
google2["Close"].apply(rank_performance).value_counts().plot(kind = "bar")


# In[ ]:

plt.style.use("ggplot")
google2["Close"].apply(rank_performance).value_counts().plot(kind = "pie", legend = True)


# In[86]:




# # 7) Data from website

# In[ ]:

girls = baby_names[baby_names["GNDR"]=="FEAMALE"]
boys = baby_names[baby_names["GNDR"]=="MALE"]


# In[ ]:

excel_file = pd.ExcelWriter("Baby Names.xlsx")
girls.to_excel(excel_file, sheet_name = "Girls", index = False)
boys.to_excel(excel_file, sheet_name = "Boys", index = False,columns = ["GNDR","NM","CNT"])
excel_file.save()


# In[ ]:



