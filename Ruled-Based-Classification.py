# Customer Potential Revenue Estimation with Rule-Based Classification
#############################################
# Business Problem
#############################################
# A gaming company wants to create level-based new customer personas and estimate
#how much potential revenue they can generate based on these new customer definitions by using some characteristics of their customers.
#For example: The company wants to estimate how much revenue an IOS user who is 25 years old and from Turkey can generate on average.

#############################################
# Dataset Story
#############################################
# The Persona.csv dataset contains the prices of products sold by an international gaming company and some demographic information of users who
#bought these products. The dataset consists of records generated in each sales transaction. 
#This means that the table is not deduplicated. In other words, a user with certain demographic characteristics may have made multiple purchases.

#Price: The amount spent by the customer
#Source: The type of device the customer connected from
#Sex: Gender of the customer
#Country: The country of the customer
#Age: Age of the customer

################# Before Application  #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# After Application  #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


# Project Task
#############################################
# TASK 1: Answer the following questions.
#############################################
# Q1: Read the persona.csv file and show the general information about the dataset.

# let's import the libraries and read the csv file
import pandas as pd
pd.set_option("display.max_rows", None)
df = pd.read_csv("WEEK02/persona.csv")
df.shape
df.head()

# Q2: How many unique SOURCE are there? What are their frequencies?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()


# Q3: How many unique PRICEs are there?
df["PRICE"].unique()
df["PRICE"].nunique()
df["PRICE"].value_counts()

#  How many sales were made from which PRICE?
df["PRICE"].value_counts()  #we use value_counts() command for the find how many 

# Q5: How many sales were made from which country?
df["COUNTRY"].nunique()
df["COUNTRY"].value_counts()

# Q6: How much was earned in total from sales by country?
df.groupby("COUNTRY")["PRICE"].sum()
df.groupby("COUNTRY").agg({"PRICE":"sum"})

# Q7: What are the sales numbers according to SOURCE types?
df.groupby("SOURCE")["PRICE"].count()
df["SOURCE"].value_counts()

# Q8: What are the PRICE averages by country?
df.groupby("COUNTRY")["PRICE"].mean()
df.groupby("COUNTRY").agg({"PRICE":"mean"})

# Q9: What are the PRICE averages by SOURCE?
df.groupby("SOURCE")["PRICE"].mean()
df.groupby("SOURCE").agg({"PRICE":"mean"})

#  Q10: What are the PRICE averages in the COUNTRY-SOURCE breakdown?
df.groupby(["SOURCE","COUNTRY"])["PRICE"].mean()
df.groupby(["SOURCE","COUNTRY"]).agg({"PRICE":"mean"})

#############################################
# TASK 2: What are the average earnings in breakdown of COUNTRY, SOURCE, SEX, AGE?

df.groupby(["SOURCE","COUNTRY","SEX","AGE"])["PRICE"].mean()

#############################################
# # TASK 3: Sort the output by PRICE.
# To see the output of the previous question better, apply the sort_values method to PRICE in descending order.
# Save the output as agg_df.

agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE",ascending=False)  
#to better see the output, we applied the sort_values method to PRICE in descending order

#############################################
# TASK 4: Convert the names in the index to variable names.
#############################################
# All variables except PRICE in the output of the third question are index names.
# Convert these names to variable names
# Hint: reset_index()
# agg_df.reset_index(inplace=True)

agg_df = agg_df.reset_index() #we use reset_index() to change the index names
agg_df.head()

#############################################
# TASK 5: Convert AGE variable to categorical variable and add it to agg_df.
#############################################
# Convert the numeric variable age to a categorical variable.
# Create the intervals in whatever way you think will be persuasive.
# For example: '0_18', '19_23', '24_30', '31_40', '41_70'

agg_df["AGE"].max()
# Let's specify where the AGE variable will be divided from:
bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]

# Let's express what the nomenclature will be in response to the dividing points:
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]


# divide the AGE
agg_df["age_cat"] = pd.cut(agg_df["AGE"],bins,labels=mylabels)
agg_df.head()

#############################################
# TASK 6: Define new level based customers and add them as variables to the dataset.
# Define a variable named customers_level_based and add this variable to the dataset.
# CAUTION!
# After creating customers_level_based values with list comp, these values need to be deduplicated.
# For example, it could be more than one of the following: USA_ANDROID_MALE_0_18
# It is necessary to take them to groupby and get the price average.

#customers_level_based
#variable names:
# how do we access the observation values?

[row[0].upper() + "_" +row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]

# We want to put the VALUES of the COUNTRY, SOURCE, SEX and age_cat variables side by side and combine them with the hyphen.
# We do this by list comphrensions
# Let's perform the operation in such a way that we choose the observation values in the above cycle that we need:


agg_df["customers_level_based"] = [row[0].upper() + "_" +row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]
agg_df.head()

agg_df["customers_level_based_2"] = agg_df[["COUNTRY","SOURCE","SEX","age_cat"]].agg(lambda x:'_'.join(x).upper(),axis=1 )
agg_df.head()


# Let's remove unnecessary variables:
agg_df = agg_df[["customers_level_based","PRICE"]]
agg_df.head()



# We are close to our original request, but there is a small problem. There will be many identical segments.
# for example, there may be many numbers from the USA_ANDROID_MALE_0_18 segment.
# lets check

agg_df["customers_level_based"].value_counts()

#Therefore, after making a groupby segment, we should take the price averages and deduplicate the segments.
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})
agg_df.head()

# because of groupby, it is located in level_base_index
# it is located in the customers_level_based index. Let's turn this into a variable.

agg_df = agg_df.reset_index()
agg_df.head()
# let's check it out. we expect each persona to have 1:


agg_df["customers_level_based"].value_counts()

#############################################
# TASK 7: Segment new customers (USA_ANDROID_MALE_0_18).
# Segment by PRICE,
# add segments to agg_df with "SEGMENT" naming,
# describe the segments,

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"],4,labels= ["D","C","B","A"])
agg_df.head()

#############################################
# TASK 8: Classify the new customers and estimate how much income they can bring.
#############################################
# # Which segment does a 33-year-old Turkish woman using ANDROID belongs to and how much income is expected to earn on average?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"]== new_user]

# In which segment and on average how much income would a 35-year-old French woman using iOS expect to earn?
new_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"]== new_user]

