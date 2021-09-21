#import all library that needed
import sqlite3
import tweepy
from nltk.corpus import stopwords
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import json
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
import numpy as np
from PIL import Image

# create class sentiment

class sentiment:
    def __init__(self, topik, database):
        self.topik = topik
        self.database = database
        
#function for crawling data from twitter
        
    def crawling(self, angka):
        consumer_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxx'
        consumer_secret = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        access_token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        access_token_secret = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        query = self.topik
        new_query = query+'-filter:retweets'
        get_data = api.search(q=new_query, count=angka, lang='id', result_type='mixed', tweet_mode='extended')
        return get_data
    
#function for create json file   
        
    def ke_json (self, nama_file):
        with open("coba.json", "w") as write_file:
            return json.dump(nama_file, write_file)
        
#function for create DataFrame        

    def ke_df(self, nama_file):
        df = pd.DataFrame(nama_file)
        return df
    
#function for open connection with Database
        
    def open_connection(self):
        self.connection = sqlite3.connect(self.database)
        
#function for updating scrapping status in Database  
        
    def check_scrappingid(self):
        query = '''UPDATE Lastscrapping SET status = 0 WHERE status = 1;'''
        conn = self.connection
        cursor = conn.cursor()
        cursor.execute(query)
        hasil = cursor.fetchall()
        conn.commit()
        cursor.close()
        return hasil
    
#function for input scrapping data into Database 
        
    def input_scrappingid(self, date, stat):
        query = '''INSERT INTO Lastscrapping (last_get, status)
        VALUES (?,?);'''
        conn = self.connection
        cursor = conn.cursor()
        cursor.execute(query, (date, stat))
        hasil = cursor.fetchall()
        conn.commit()
        cursor.close()
        return hasil
    
#function for input user data into Database    
        
    def input_databaseuser(self, usid, nm, snme, loc, acc_cr, fwr, frd, vfd):
        query = '''INSERT OR IGNORE INTO User VALUES (?,?,?,?,?,?,?,?);'''
        cursor = self.connection.cursor()
        conn = self.connection
        for j in range(len(nm)):
            cursor.execute(query, (usid[j], nm[j], snme[j], loc[j], acc_cr[j], fwr[j], frd[j], vfd[j]))
        hasil = cursor.fetchall()
        conn.commit()
        cursor.close()
        return hasil
    
#fuction for input twet data into the Database
        
    def input_tweet(self, twid, usid, ctd, twt, scid):
        query = '''INSERT OR IGNORE INTO Tweet (tweetid, userid, createddate, tweet, scrapping_id)
        VALUES (?,?,?,?,?);'''
        cursor = self.connection.cursor()
        conn = self.connection
        for i in range(len(twid)):
            cursor.execute(query, (twid[i], usid[i], ctd[i], twt[i], scid))
        hasil = cursor.fetchall()
        conn.commit()
        cursor.close()
        return hasil
    
#function for select some spesific table    
        
    def select_table(self, code):
        query = code
        cursor = self.connection.cursor()
        conn = self.connection
        cursor.execute(query)
        hasil = cursor.fetchall()
        conn.commit()
        cursor.close()
        return hasil
    
#function to input sentiment
    
    def input_sentiment(self, twtid, stmnt):
        query = '''INSERT OR IGNORE INTO Sentiment
        VALUES (?,?);'''
        cursor = self.connection.cursor()
        conn = self.connection
        for i in range(len(twtid)):
            cursor.execute(query, (twtid[i], stmnt[i]))
        hasil = cursor.fetchall()
        conn.commit()
        cursor.close()
        return hasil
    
#function for input cleaned data    
        
    def input_clean(self, clean, twtid):
        query = '''UPDATE Tweet SET cleantweet = ? WHERE tweetid = ?;'''
        cursor = self.connection.cursor()
        conn = self.connection
        for k in range(len(clean)):
            cursor.execute(query, (clean[k], twtid[k]))
        hasil = cursor.fetchall()
        conn.commit()
        cursor.close()
        return hasil
    
#function for close the connection    
        
    def close_connection(self):
        self.connection.commit()
        self.connection.close()
        
 #fuction for cleaning tweet words  
        
    def cleaning(self, kalimat):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stopword = stopwords.words('indonesian')
        lower = kalimat.lower()
        del_num = re.sub(r"\d+", "", lower)
        del_link = del_num.split('http')[0]
        del_punc = del_link.translate(str.maketrans('','',string.punctuation))
        del_space = del_punc.strip()
        words = del_space.split()
        resultwords  = [word for word in words if word not in stopword]
        result = ' '.join(resultwords)
        return stemmer.stem(result)

#run the class
analisa = sentiment("Habib Rizieq Shihab", 'firdaus.salim24_final.db')

#start to crawl data
x = analisa.crawling(200)

#gather data from crawling result into spesific tweet data
user = [a.user for a in x]
usid = [a.id for a in x]
tweet = [a.full_text for a in x]
date = [a.created_at for a in x]
twtid = [a.id for a in x]

#gather data from crawling result into spesific user data
userid = [a.user.id for a in x]
name = [a.user.name for a in x]
screenname = [a.user.screen_name for a in x]
location = [a.user.location for a in x]
account_created = [a.user.created_at for a in x]
follower = [a.user.followers_count for a in x]
friend = [a.user.friends_count for a in x]
verified = [a.user.verified for a in x]
acc_date = [a.user.created_at for a in x]

#open connection to sqlite database
analisa.open_connection()

#check batch scrapping
analisa.check_scrappingid()

#input tweet table
analisa.input_tweet(twtid, userid, date, tweet, 17)

#update scrapping batch 
analisa.input_scrappingid('2020-11-26', 1)

#input table user
analisa.input_databaseuser(userid, name, screenname, location, acc_date, follower, friend, verified)

#cleaning data
data_clean = []
for kata in tweet:
    v = analisa.cleaning(kata)
    data_clean.append(v)
    
#input cleaned tweet
analisa.input_clean(data_clean, twtid)

#sqlite select table
x = analisa.select_table('''SELECT * FROM Tweet;''')

user = analisa.select_table('''SELECT * FROM User;''')

sentimen = analisa.select_table('''SELECT * FROM Sentiment;''')

sentimen_id = analisa.select_table('''SELECT a.name, c.sentiment
FROM User a
INNER JOIN Tweet b ON a.userid = b.userid
INNER JOIN Sentiment c ON b.tweetid = c.tweetid
;''')


#Top Words

df = pd.DataFrame(x)
tweetbersih = df[4].tolist()
print(tweetbersih)
data = [word_tokenize(paragraf) for paragraf in tweetbersih]
data = list(itertools.chain(*data))
fqdist = FreqDist(data)

#Plot Top Words

plt.figure(figsize=(15,10))
fqdist.plot(10,cumulative=False, marker='o')
plt.show()

#Wordcloud
data_1 = ' '.join(data)

job_mask = np.array(Image.open("twitter-2012-positive.png"))

wordcloud = WordCloud(background_color="white", max_words=2000, mask=job_mask, width=1600, height=800, max_font_size=200).generate(data_1)
wordcloud.to_file("wordcloud twitter.png")

plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.figure()
plt.axis("off")
plt.show()

#sentiment Analysis

pos_list= open("./kata_positif.txt","r")
pos_kata = pos_list.readlines()
neg_list= open("./kata_negatif.txt","r")
neg_kata = neg_list.readlines()

for item in tweetbersih:
    count_p = 0
    count_n = 0
    for kata_pos in pos_kata:
        if kata_pos.strip() in item:
            count_p +=1
    for kata_neg in neg_kata:
        if kata_neg.strip() in item:
            count_n +=1
    print ("positif: "+str(count_p))
    print ("negatif: "+str(count_n))

hasil = []
for item in tweetbersih:
    count_p1 = 0
    count_n1 = 0
    for kata_pos in pos_kata:
        if kata_pos.strip() in item:
            count_p1 +=1
    for kata_neg in neg_kata:
        if kata_neg.strip() in item:
            count_n1 +=1
    hasil.append(count_p1 - count_n1)
    
print ("Nilai rata-rata: "+str(np.mean(hasil)))
print ("Standar deviasi: "+str(np.std(hasil)))

labels, counts = np.unique(hasil, return_counts=True)
plt.figure(figsize=(14,7))
plt.bar(labels, counts, align='center', edgecolor='black')
plt.gca().set_xticks(labels)
plt.xlabel('Sentiment Score')
plt.ylabel('Count')
plt.title('Sentiment Analysis')
plt.savefig('sentiment_score', dpi=100)
plt.show()

sentiment = []
for nilai in hasil:
    if nilai > 0 :
        sentiment.append('Positive')
    elif nilai == 0:
        sentiment.append('Neutral')
    else:
        sentiment.append('Negative')

df_sentiment = pd.DataFrame({'tweetbersih':tweetbersih, 'sentimen_angka':hasil, 'sentiment':sentiment})

labels, counts = np.unique(df_sentiment['sentiment'], return_counts=True)
plt.figure(figsize=(10,7))
plt.bar(labels, counts, align='center', color='green', width=0.2)
plt.xlabel('Sentiment')
plt.ylabel('Counts')
plt.title('Sentiment Analysis')
plt.gca().set_xticks(labels)
plt.savefig('chart_sentimen', dpi=100)
plt.show()

df['Sentiment'] = sentiment
df['sentiment_angka']=hasil


twtid = df[0].tolist()
stmnt = df['sentiment_angka'].tolist()

# analisa.input_sentiment(twtid, stmnt)

#close connection
analisa.close_connection

#Exploratory Data

df_user = pd.DataFrame(user)
df[2]=pd.to_datetime(df[2])
df['new_date'] = df[2].dt.date

s = df.groupby('new_date').size()
fig, ax = plt.subplots(figsize=(14,7))
ax.plot(s.index, s, marker='o')
ax.set_xlabel('Date')
ax.set_ylabel('Count')
ax.set_title('Data Crawling Counts per Date')

for i,j in zip(s.index,s):
    ax.annotate( str(j),xy=(i,j),fontsize=12)
plt.xticks(rotation=35)
plt.savefig('grafik crawling', dpi=100)
plt.show()


status = []
for value in df_user[7]:
    if value == 1:
        status.append('verified')
    else:
        status.append('not verified')
df_user[8]=status

df_group = df_user.groupby(8).size()

fig, ax = plt.subplots(figsize=(10,8))
ax.bar(df_group.index, df_group, label='Sentiment', width=0.2, color='darkseagreen')
plt.savefig('vstatus', dpi=100)
plt.show()

df_ids = pd.DataFrame(sentimen_id)

df_user[4]=pd.to_datetime(df_user[4])
df_user['new_date'] = df_user[4].dt.year

j = df_user.groupby('new_date').size()
fig, ax = plt.subplots(figsize=(14,7))
ax.bar(j.index, j, color='purple')
ax.set_xlabel('Year Created')
ax.set_ylabel('Count')
ax.set_title('Account Created Date Count')
plt.savefig('acc_created', dpi=100)
plt.show()

r = df_user.groupby(['new_date', 8]).size().unstack(level=-1).reset_index()

fig, ax = plt.subplots(figsize=(14,7))
ax.bar(r['new_date'], r['not verified'], label='not verified')
ax.bar(r['new_date'], r['verified'], bottom=r['not verified'], label='verified')
ax.set_xlabel('Year')
ax.set_ylabel('Count')
ax.set_title('Account Created Date Count')
plt.legend()
plt.savefig('acc_created status', dpi=100)
plt.show()

pos_kata = ' '.join(pos_kata)
wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(pos_kata)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('dict_pos_kata', dpi=100)
plt.show()

neg_kata = ' '.join(neg_kata)
wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(neg_kata)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('dict_neg_kata', dpi=100)
plt.show()
