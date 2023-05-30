import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
#import nltk
#from sklearn.feature_extraction import text
#from sklearn.metrics import accuracy_score
import os
from num2words import num2words
#import re 
#from hashlib import new
import random
import streamlit as st
from PIL import Image
from random import choice
from string import ascii_lowercase, digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# streamlit title and text
st.title("Swapsies App")
st.write("Enter an item description to swap")


# check in correct directory
print(os.listdir())
print(os.getcwd())

#dataset_2 = "/Users/rainietamakloe/CodingProjects/final_year_project/dataset_files/styles_2"

# current directory containing styles file to be opened, need to be changed for other pc
dir = "/Users/rainietamakloe/CodingProjects/final_year_project/dataset_files/"
# directory for the images so they can be stored locally in the correct file 
dir_img = "/Users/rainietamakloe/CodingProjects/final_year_project/dataset_files/images/"

# changed directory here to open csv file
os.chdir(dir)
file = open('styles_2.csv')


# read csv file, which contains all data
df = pd.read_csv(file)
df = df.dropna()  #drop any empty values
df = df.drop_duplicates() #drop any duplicate valyes

#drop all other coloumns not being used
df = df.drop(['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage'], axis = 1)

# convert dataframe into dictionary so i can interate through each value
df_dict = df.set_index('id').T.to_dict('list')


# drop all colums except id and product name then convert to dict then find description with num and remove from dict
#print(f" old value is {df_dict[28049]}")

for key, value in df_dict.items():
    # empty string to store changed values
    placeholder_value = ""
    i = 0
    for word in value[i].split(): # split display name into individual words
        if word.isdigit():
            word = num2words(word)
            placeholder_value += word + " "
        else:
            placeholder_value += word + " "
        if i != len(df_dict[key]) - 1: # increment to end of current display name
                 i += 1
        df_dict[key] = placeholder_value  # new changed or unchaged display name saved

#print(f" new value is {df_dict[28049]}")

# convert back to dataframe
new_df = pd.DataFrame.from_dict(df_dict, orient='index')
new_df.columns = ["ProductDisplayName"]

# reset index of dataframe
new_df = new_df.rename_axis('id').reset_index()

#import a copy of dataframe with sub category labels
df_label = pd.read_csv('styles_2.csv')
df_label.dropna(inplace=True)
df_label.drop([ 'gender', 'articleType', 'masterCategory', 'baseColour', 'season', 'year', 'usage', 'productDisplayName'], axis = 1, inplace=True)


# merge processed dataframe and new label dataframe into one. Order not changed so items align
df_3 = pd.merge(new_df, df_label, on="id")

category_total = df_3['subCategory'].value_counts()
# find the most prominent categories inside the subcategory column and drop the rest
# keeps the categories at a good number and each needs to contain a substantial amount of items for classifier
new_sub = category_total.drop(labels = ['Shoe Accessories',
 'Lips', 'Saree', 'Scarves', 
 'Wallets', 'Apparel Set', 'Headwear', 'Mufflers', 'Skin Care', 'Makeup',
 'Free Gifts', 'Ties', 'Accessories', 'Nails', 'Beauty Accessories',
 'Water Bottle', 'Skin', 'Eyes', 'Bath and Body', 'Gloves',
 'Sports Accessories', 'Cufflinks', 'Sports Equipment', 'Stoles', 'Hair',
 'Perfumes', 'Home Furnishing', 'Umbrellas', 'Wristbands', 'Vouchers'])

#define new column
sub_apparel = new_sub.index
#define new dataframe with only chosen sub category items
df_4 = df_3[df_3['subCategory'].isin(sub_apparel)]

#split data into labelled and unlabelled for classifier
df_labelled = df_4.iloc[:22030,:]
df_unlabelled = df_4.iloc[22030:,:]

#drop sub category item on unlabelled dataframe
df_unlabelled.drop(['subCategory'],axis=1, inplace=True)

#defining my own stop words, used numbers since they were prominent and will add no value
new_stopwords = ["one",
                 "two",
                 "three",
                 "four",
                 "five",
                 "six",
                 "seven",
                 "eight",
                 "nine",
                 "ten",
                 "eleven",
                 "twelve",
                 "thirteen",
                 "fourteen",
                 "fifteen",
                 "sixteen",
                 "seventeen",
                 "eighteen",
                 "nineteen",
                 "twenty",
                 "thirty",
                 "fourty",
                 "fifty",
                 "sixty",
                 "seventy",
                 "eighty",
                 "ninety",
                 "hundred",
                 "thousand",
                 "Red",
                 "Orange",
                 "Yellow",
                 "Green",
                 "Blue",
                 "Pink",
                 "Purple",
                 "Brown",
                 "Black",
                 "Grey",
                 "White",
                 ]
#added above list to original list of stopwords 
#my_stopwords = text.ENGLISH_STOP_WORDS.union(new_stopwords)
#my_stopwords = list(my_stopwords)

# split dataset for classifier into two variables
X = df_labelled['ProductDisplayName']
y = df_labelled['subCategory']
# eighty twenty split and data is shuffled
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

#logisitic regression section

# test and train set converted into vectors
tfidf_vectorizer = TfidfVectorizer(stop_words=new_stopwords) 
tfidf_Xtrain = tfidf_vectorizer.fit_transform(X_train)
tfidf_Xtest = tfidf_vectorizer.transform(X_test)

# regularization = shrinks coefficients in regression formula. When decision boundary 
#is very complex, the sigmoid function becomes complex so regularization helps reduce this
#so the model is more generalized and less specific to training data, whhich mean better results
# regularization avoids overfitting where the model is good with training data but not testing

# C is Inverse of regularization strength, default is 1. goes up in increments of 
# solver is an algorithm used for optimization of calculating the coefficients
# solver creates the model function by finding the best parameters with the least error
# penalty is the type of reguarization, L2 by default
lr_tfidf = LogisticRegression(solver = 'lbfgs', C=10, multi_class='ovr', max_iter=1000)
# max iterations set to large number so the algorithm can run over and over without throwing an error

# training data fit to lr model
lr_tfidf.fit(tfidf_Xtrain, y_train)  

#Predict y value for test dataset, array of predicted labels
y_predict = lr_tfidf.predict(tfidf_Xtest)

# builds a text report showing the main classification metrics using actual correct labels, y_test, and predicted labels
print(classification_report(y_test,y_predict))

# precision is the number of correct results divided by the number of all returned results.
# precision essentially is the models ability to have no false positives e.g prediced bottomwear or eyewear as bag
# recall is the number of positive/correct results which were correctly classified
# recall is the models ability to have no false negatives e.g predicted a bag as bottomwear or eyewear
# f1 score calculates f measure, weighted harmonic mean of the precision and recall
# support is number of samples each metric is calculated on
# accuracy is the sum of true positives and true negatives divided by the total number of samples
# macro avg = average of precision, recall and f1 score
# weighted avg = average of precision, recall and f1 score with respect to amount of samples in each class

#global new_unlabelled_df
def add_item(user_input):
    
    item_id = random.randint(100,100000) # generate id
    id_list = df_unlabelled['id'].tolist()

    if item_id in id_list:
        item_id = random.randint(100,100000) # if id exists generate a new id
    
    unlabelled_input_df = pd.DataFrame({  #adds content to dataframe
        "id": item_id,
        "ProductDisplayName": user_input,
    }, index = [df_unlabelled.shape[0] + 1])
    
    new_unlabelled_df = pd.concat([df_unlabelled, unlabelled_input_df]) # joins both dataframes together

    #print(item_id)
    id_item = new_unlabelled_df.loc[new_unlabelled_df['ProductDisplayName'] == user_input, 'id'].values[0]
    #print(id_item)

    return id_item

# randomly generated emails to represent the contact details of each user
emails = ascii_lowercase + digits
contact = [''.join(choice(emails) for _ in range(4)) for _ in range(len(df_unlabelled))]
# looping to add email tag to strings
for i in range(len(contact)):
  contact[i] = contact[i] + str('@gmail.com')
# column added to the database to be fetched later
df_unlabelled.insert(loc=len(df_unlabelled.columns), column='OwnerContact', value=contact)


def recommender(user_input):
  # append user input to unlabelled database so it can be classified
  id = random.randint(10000,50000)

  unlabelled_input_df = pd.DataFrame({
      "id": id,
      "ProductDisplayName": user_input,
  }, index = [df_unlabelled.shape[0] + 1])

  # append user item to unlabelled database
  new_unlabelled_df = pd.concat([df_unlabelled, unlabelled_input_df])

  #print(new_unlabelled_df)

  # get the id of the item that was just added

  #new_unlabelled_df = df_unlabelled.append(unlabelled_input_df)

  # testing unlabelled data with user input
  X_test = new_unlabelled_df['ProductDisplayName']

  # unseen unlabelled data is transformed usinf tfidf
  unseen_vec = tfidf_vectorizer.transform(X_test)
  # uses the unseen unlabelled data and predicts it with classifier
  y_predict = lr_tfidf.predict(unseen_vec)  
  # calculates the probability of each item being correctly classified
  #y_prob = lr_tfidf.predict_proba(unseen_vec)[:,1]

   # prints probabilities in dataframe
  #new_unlabelled_df['predict_prob']= y_prob
  # list of predicted categories made into dataframe column
  new_unlabelled_df['subCategory']= y_predict
  # all of the unseen unlabelled data that has been classified
  final = new_unlabelled_df[['ProductDisplayName','subCategory']].reset_index(drop=True)
  #print(final.tail())


  #fetch user input classified category item so it can be used only to recommend
  userinput_category = final.loc[final['ProductDisplayName'] == user_input, 'subCategory'].iloc[0]

  #print(userinput_category)

  #new classified database with items which match user inputs category
  userinput_df = final.loc[final['subCategory'] == userinput_category]
  # dropping duplicated that may be present
  userinput_df = userinput_df.drop_duplicates(subset = "ProductDisplayName")

  #print(userinput_df.head())
  
  tfidf = TfidfVectorizer(stop_words=new_stopwords) 
  # the product display names are fit onto the tf idf model
  # creates vectors for each word in a matrix
  tfidf_matrix = tfidf.fit_transform(userinput_df['ProductDisplayName'])

  # the matrix is there twice so that each point can be compared to every other point
  cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

  #print(cos_sim)
  print(np.shape(cos_sim)) 
  # resetting the index so it can be fetched later
  userinput_df = userinput_df.rename_axis('id').reset_index()
  #print(userinput_df.head())

  #######
  # then find sim score of item and all corresponding sim scores
  #######
  #fecthes index of item that fucntion receives and turns into list type
  index = userinput_df[userinput_df['ProductDisplayName'] == user_input].index.values
  # turns list type into int type for cos sim matrix
  index = index[0]
  #turns cos sim matrix into a tuple/list type to make it easier to fetch/loop/see values
  similarity_values = enumerate(cos_sim[index])
  sim_values = list(similarity_values)
  print("id of user input item is" + str(index))
  print(len(tfidf.vocabulary_))
  #sorts cos sim values using python sorted func which takes sim vals 
  # and key as lambda gets index 1 which is cos sim val and reverse so highest at top
  sim_values_sorted = sorted(sim_values, key = lambda x: x[1], reverse = True)
  # saves 2nd to 12th val since very first value is itself
  sim_values_list = sim_values_sorted[1:11]
  for i in sim_values_list:
      print(i)
  # saves first val in sim val list/tuple and loops thru each to get index of each
  similarity_val_index = [i[0] for i in sim_values_list]
  # saves output in new dataframe with the indexes of sim values
  user_df_output = (userinput_df['ProductDisplayName'].iloc[similarity_val_index])
  
  #######
  # then print results in terminal
  #######
  print(user_df_output)
  #st.write(user_df_output)
  # chnage directory to access images
  os.chdir(dir_img)
  # convert results to list
  results_list = user_df_output.tolist()

  results_id_list = []
  results_list_contact = []
  # looping through list and getting the id of each item in results list from older dataframe
  for i in range(len(results_list)):
      results_id = new_df.loc[new_df['ProductDisplayName'] == results_list[i], 'id'].iloc[0]
      results_id_list.append(results_id)
      #fetches the contact email for each item
      results_contact = df_unlabelled.loc[df_unlabelled['ProductDisplayName'] == results_list[i], 'OwnerContact'].iloc[0]
      results_list_contact.append(results_contact)
      image = Image.open(str(results_id) + '.jpg')
      #st.image(image, caption = results_list[i], width=100)
      st.image(image, caption = str(results_list[i]) + " \n Contact: \n" + str(results_list_contact[i]), width=100)
      #lobal swapItem
      #swapItem = st.button("Request Swap", key=i)
      #image.show()


#print(results_id_list)

input = st.text_input("Item Description", key=input)


result = st.button("Find a Recommendation")

# error handling input so it is long enough and not empty
if result:
    if len(input) == 0:
        st.error("please enter a description")
    elif len(input) < 10:
        st.error("please enter a longer description")
    else:
        recommender(input)



st.sidebar.success("Select a page above.")

#recommender("blue striped top")
