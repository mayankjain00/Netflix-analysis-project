#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # data processing, CSV file I/O
import numpy as np # linear algebra
import seaborn as sns #drawing attractive and informative statistical graphics
import matplotlib.pyplot as plt # cross-platform, data visualization and graphical plotting library 


# In[2]:


#add dataset
data=pd.read_csv("netflix_titles.csv")
data


# In[3]:


import warnings
warnings.filterwarnings('ignore')
data.fillna(method="ffill",inplace=True)


# In[4]:


#top 5 record to fatch
data.head()


# In[5]:


#all data columns shlowing
data.columns


# In[6]:


#all infromation in data
data.info()


# In[6]:


#qualitative statistics of object data 
data.describe()


# In[7]:


#qualitative statistics of object data with transpose
data.describe().T


# In[7]:


#data shape showing rows and columns
data.shape


# In[8]:


## Check for Null values using isnull() function
data.isnull().sum()


# In[9]:


#to findout whether column is contaning any missing values 
data.isna().any()


# In[10]:


#Droping all null values
data = data.dropna(how='any' ,axis=0)
data.shape


# In[11]:


data


# In[12]:


#to check whether missing values are present or not 
data.isna()


# In[13]:


data.isna().all()


# In[14]:


#check dimension of this dataset
data.ndim


# In[15]:


#Visualizing Missing Data with Seaborn Heatmap 
#but right now missung value
#this is another method
sns.heatmap(data.isnull())


# In[16]:



#replacing the missing value with mode
data['director'] = data['director'].fillna(data['director'].mode()[0])
data['cast'] = data['cast'].fillna(data['cast'].mode()[0])
data['country'] = data['country'].fillna(data['country'].mode()[0])
data['date_added'] = data['date_added'].fillna(data['date_added'].mode()[0])
data['director'] = data['director'].fillna(data['director'].mode()[0])
data['rating'] = data['rating'].fillna(data['rating'].mode()[0])
data['duration'] = data['duration'].fillna(data['duration'].mode()[0])


# In[17]:


#no missing value find
sns.heatmap(data.isnull())


# In[18]:


data['date_added']


# In[19]:


data


# In[20]:


#separating the date_added to days,year and months columns
data["date_added"] = pd.to_datetime(data['date_added'])
data['day_added'] = data['date_added'].dt.day
print("Date:\n",data['day_added'])
data['year_added'] = data['date_added'].dt.year
print("Year:\n",data['year_added'])
data['month_added']=data['date_added'].dt.month
print("Month:\n",data['month_added'])
data['year_added'].astype(int)
data['day_added'].astype(int)
data['weekday'] = data['date_added'].dt.weekday #returns 0 to 6
data['weekday']


# In[21]:


data1=data.drop(['show_id','date_added','description','director','cast'],axis=1)


# In[22]:


data1


# In[23]:


data1.rename(columns = {'listed_in':'Genres','day_added':'date_added'}, inplace = True)
data1


# In[24]:


# Lets retrieve just the first country
data1['country'] = data1['country'].apply(lambda x: x.split(",")[0])
data1['Genres'] = data1['Genres'].apply(lambda x: x.split(",")[0])
data1.to_csv("updated_data_netflix.csv")


# In[25]:


# Setting labels for items in Chart
labels = ['Movie', 'TV show']
size = data1['type'].value_counts()
  
# colors
colors = ['Yellow','Pink']
  
# Pie Chart
plt.pie(size, colors=colors, labels=labels,
        autopct='%1.1f%%', pctdistance=0.85)
  
# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
  
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)
  
# Adding Title of chart
plt.title('Netflix Distribution on Type')
  
# Displaying Chart
plt.show()


# In[26]:


data1['rating'].replace({'PG-13':'Above 13 years', 
                      'TV-MA':'Adult','PG': 
                      'Parental Guidance:Kids',
                      'TV-14':'Above 13 years',
                      'TV-PG':'Parental Guidance:Kids',
                      'TV-Y': 'Kids',
                      'TV-Y7': 'Kids above 7 years',
                      'R': 'Adult',
                      'TV-G':'Kids',
                      'G':'Kids',
                      'NC-17':'Adult',
                      '74 min':'Adult',
                      '84 min':'Adult',
                      '66 min':'Adult',
                      'UR':'Unrated',
                      'NR':'Unrated',
                      'TV-Y7-FV':'Kids above 7 years'},inplace=True)


# In[27]:


count=pd.DataFrame(data1['rating'].value_counts())
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,6),dpi=100)
mycolors=['#52D726','#FFEC00','#FF7300','#007ED6','#FF0000','#7CDDDD']
x=data1['rating'][data1.type=='Movie'].value_counts()
x1=data1['rating'][data1.type=='TV Show'].value_counts()
y=count.index
axes[0].pie(x,labels=y,colors=mycolors,autopct='%1.2f%%' )
axes[0].set_title('Movie',fontsize = 20)
axes[1].pie(x1,labels=y,colors=mycolors,autopct='%1.2f%%' )
axes[1].set_title('TV Show',fontsize = 20)
plt.tight_layout()
plt.show()


# In[28]:


#top 5 country with number of shows
topfive=data1['country'].value_counts()[:5]
x1= topfive.index
y1= topfive.values
plt.figure(figsize=(12,6))
plt.title("Top 5 Countries Producing Netflix Shows")
plt.xlabel("Countries")
plt.ylabel("Number of shows")
sns.barplot(x=x1,y=y1)
plt.show()


# In[29]:


#let's see side by side comparision
x2=data1[data1['country'].isin(x1)]
plt.figure(figsize=(17,6))
plot = sns.countplot(x='type',data=x2,hue='country')
plot.set_ylabel("Count of Content",fontsize=10, weight='bold')
plot.set_xlabel("Countries",fontsize=10,weight='bold')
plt.legend(title='country', loc='upper right')
plt.title(' Comparision of TV Show/ Movies of top five content producing countries');


# In[30]:


plt.figure(figsize = [12,6])
ax=sns.countplot(x = data1['rating'], order=data1['rating'].value_counts().index)
ax.set(xlabel='Rating', ylabel='Number of Shows')
plt.title("Netflix Show Based on Rating")
plt.show()


# In[31]:


#displaying the number of shows released yearwise
x1=data1[data1['year_added']>2015]
x=x1['year_added'].unique()
#x=df1['year_added'].unique()
y=x1['year_added'].value_counts()
plt.figure(figsize = [10,5])
ax=sns.barplot(x,y)
ax.set(xlabel='Year', ylabel='Number of Shows')
plt.title("Netflix Show Based on Year")
plt.show()


# In[32]:


data1['month_added'].replace({1:'January', 
                            2:'February',
                            3:'March',
                            4:'April',
                            5:'May',
                            6:'June',
                            7:'July',
                            8:'August',
                            9:'September',
                            10:'October',
                            11:'November',
                            12:'December'},inplace=True)


# In[33]:


data1


# In[34]:


#displaying the number of shows released monthwise
x=data1['month_added'].unique()
y=data1['month_added'].value_counts().sort_values(ascending=True)
plt.figure(figsize = [10,5])
ax=sns.barplot(x,y)
ax.set(xlabel='Month', ylabel='Number of Shows')
plt.xticks(rotation =45)
plt.title("Netflix Shows Based on Months")
plt.show()


# In[35]:


#displaying the number of shows released daywise
x=data1['date_added'].unique()
y=data1['date_added'].value_counts()
plt.figure(figsize = [10,6])
ax=sns.barplot(x,y)
ax.set(xlabel='Date', ylabel='Number of Shows')
plt.xticks(rotation =45)
plt.title("Netflix Shows Based on Dates")
plt.show()


# In[36]:


#highest weekwise shows
week1=data1[(data1['date_added']>=1) & (data1['date_added']<8)]
week2=data1[(data1['date_added']>=8) & (data1['date_added']<15)]
week3=data1[(data1['date_added']>=15) & (data1['date_added']<22)]
week4=data1[(data1['date_added']>=22) & (data1['date_added']<31)]

xweek = ['week 1','week 2','week 3','week 4']
yweek = [len(week1), len(week2), len(week3), len(week4)]
sns.color_palette("mako", as_cmap=True)
sns.barplot(x = xweek, y = yweek,palette='rainbow')
plt.ylabel('Number of Shows')
plt.title("Netflix Show Weekwise")
plt.show()


# In[37]:


#all data columns shlowing
data.columns


# In[38]:


filtered_genres = data1.set_index('title').Genres.str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
plt.figure(figsize=(10,10))
g = sns.countplot(y = filtered_genres, order=filtered_genres.value_counts().index[:10],palette="Set3")
plt.title('Top 10 Genres on Netflix')
plt.show()


# In[39]:


#Top 5 Netflix Shows based on Duration
fill_duration = data1.set_index('title').duration
plt.figure(figsize=(10,6))
g = sns.countplot(x = fill_duration, order=fill_duration.value_counts().index[:5],palette="Set3")
plt.title('Top 5 Netflix Shows based on Duration')
plt.xlabel('Duration')
plt.ylabel('No of Shows')
plt.show()


# In[41]:


data['weekday'] = data['date_added'].dt.weekday #returns 0 to 6
data['weekday']


# In[42]:


data


# In[43]:


def days(weekday):
    if weekday<2:
        return 'Monday'
    elif weekday<3:
        return 'Tuesday'
    elif weekday<4:
        return 'Wednesday'
    elif weekday<5:
        return 'Thursday'
    elif weekday<6:
        return 'Friday'
    else:
        return 'Saturday'


# In[44]:


data['weekday']=data['weekday'].apply(days)


# In[45]:


p=data['weekday'].value_counts().sort_values(ascending=True)
x=p.keys()
y=p.values
sns.barplot(x,y,palette='rocket')
plt.title("Netflix Show Production Daywise")


# In[46]:


from sklearn.preprocessing import LabelEncoder
lbl_encode=LabelEncoder()


# In[47]:


data1['title']=lbl_encode.fit_transform(data1['title'])
data1['country']=lbl_encode.fit_transform(data1['country'])
data1['date_added']=lbl_encode.fit_transform(data1['date_added'])
data1['release_year']=lbl_encode.fit_transform(data1['release_year'])
data1['month_added']=lbl_encode.fit_transform(data1['month_added'])
data1['rating']=lbl_encode.fit_transform(data1['rating'])
data1['duration']=lbl_encode.fit_transform(data1['duration'])
data1['Genres']=lbl_encode.fit_transform(data1['Genres'])


# In[48]:


data1.to_csv("machine.csv")


# In[49]:


x=data1.drop(['type'],axis=1)
y=data1['type']


# In[50]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[51]:


xtrain


# In[52]:


from sklearn.feature_selection import mutual_info_classif
mutual_info=mutual_info_classif(xtrain,ytrain)
mutual_info=pd.Series(mutual_info)
mutual_info.index=xtrain.columns
s=mutual_info.sort_values(ascending=False)
s.plot.bar(figsize=(10,6))


# In[53]:


from sklearn.feature_selection import SelectKBest
top_five_cols=SelectKBest(mutual_info_classif,k=5)
top_five_cols.fit(xtrain.fillna(0),ytrain)
xtrain.columns[top_five_cols.get_support()]


# In[54]:


x1=data1.drop(['type','title','month_added','date_added','release_year'],axis=1)
y1=data1['type']


# In[55]:


x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.2)


# In[56]:


x_train


# In[57]:


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(x_train, y_train)


# In[58]:


result=logr.predict(x_test)
result


# In[59]:


score = logr.score(x_test,y_test)*100
print(score)


# In[60]:


from sklearn import metrics
cm = metrics.confusion_matrix(y_test,result)


# In[61]:


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()


# In[62]:


from sklearn import tree
train_features, test_features, train_targets, test_targets = train_test_split(x1,y1, test_size=0.2, random_state=123)
decision_tree = tree.DecisionTreeClassifier(random_state=456)
decision_tree = decision_tree.fit(train_features, train_targets)
class_names = ['TV Show','Movie']
feature_names=['country','rating','duration','Genres','year_added','weekday']


# In[63]:


plt.subplots(figsize=(17, 12))
tree.plot_tree(decision_tree, feature_names=feature_names, filled=True, rounded=True, class_names=class_names)
plt.savefig("decision_tree.png")


# In[64]:


from sklearn.metrics import accuracy_score
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x1,y1)
labels_test = clf.predict(test_features)
acc = accuracy_score(labels_test,test_targets)*100
print("Accuracy of Decision Tree Classifier is:",acc)


# In[65]:


from sklearn import metrics
cm1 = metrics.confusion_matrix(test_targets,labels_test)


# In[66]:


plt.figure(figsize=(9,9))
sns.heatmap(cm1, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(acc)
plt.title(all_sample_title, size = 15)
plt.show()


# In[67]:


from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(max_depth=4, random_state = 10) 
model.fit(x_train, y_train)


# In[68]:


from sklearn.metrics import accuracy_score
pred = model.predict(x_test)
acc1=round(accuracy_score(y_test,pred)*100,2)
print("Accuracy of Random Forest Classifier is: ",acc1)


# In[69]:


from sklearn import metrics
cm2 = metrics.confusion_matrix(y_test,pred)


# In[70]:


plt.figure(figsize=(9,9))
sns.heatmap(cm2, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(acc1)
plt.title(all_sample_title, size = 15)
plt.show()


# In[71]:


from sklearn.feature_selection import mutual_info_classif
mutual_info=mutual_info_classif(xtrain,ytrain)
mutual_info=pd.Series(mutual_info)
mutual_info.index=xtrain.columns
s=mutual_info.sort_values(ascending=False)
s.plot.bar(figsize=(10,6))


# In[ ]:




