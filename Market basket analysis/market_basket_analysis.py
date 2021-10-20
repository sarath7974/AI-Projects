"""
Table of Contents
(1)Data Preprocessing
(2)Exploratory Data Analysis and Visualisation

    - Top 10 countries with most transactions
    - Sales and Revenue Analysis
    - Customer Analysis
    - Create Basket Data for Customers
    - Data Encoding
(3)Apply apriori
    - Generate Frequent Itemset
(2)Association Rule Mining
    - Generate Rules
"""

#Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from pandas.plotting import parallel_coordinates
from IPython.display import display

#Reading data
data = pd.read_excel("online_retail_data.xlsx",engine = "openpyxl")
print("SHAPE OF DATA :",data.shape)
data.head()

data.info()
data.describe()

#Data Preprocessing
data=data.dropna()                                   #drop null values
data["Description"]=data["Description"].str.strip()  #remove spaces from description
data.drop(data[data["Quantity"]<=0].index)           #remove negative values from quantity
data['Invoice']=data["Invoice"].astype("str")        #convert invoice to string type
data=data[~data["Invoice"].str.contains("C")]        #uses bitwise negation operation to remove all the credit transactions
data.rename(columns={"Price":"Unitprice"},inplace=True)   # Rename price to Unitprice
data["Totalprice"]=data["Unitprice"]*data["Quantity"]     # Adding total price
print("SHAPE OF DATA :",data.shape)
data.head(10)

data.nunique()   #Number of unique values of each label

#EDA
#Top 10 countries with most transactions
def top_transactions(data):
    """
    Input :
    data: Input dataframe
    _______________________________
    Output: Top 10 Transaction
    _______________________________
    """
    top_10_transaction = pd.DataFrame(data.groupby("Country").nunique().sort_values("Invoice",ascending=False).head(10))
    top_10_transaction.reset_index(inplace=True)
    # Visualize top 10 Countries with Most Transactions
    top_10_transaction.plot(kind='pie', y = 'Invoice', autopct='%1.0f%%',labels=top_10_transaction['Country'],figsize=(10,10),title="Top 10 Countries With Most Transactions")
    display(top_10_transaction)

top_transactions(data)

#Top 10 Most Bought Items
def most_bought_items(data):
    """
    Input  :
    data: Input Dataframe
    _____________________________________________
    Output : Most Bought item in Total Transaction
    ______________________________________________
    """
    df=pd.DataFrame(data.groupby("Description")["Quantity"].sum())
    most_bought=df.sort_values("Quantity", axis = 0, ascending = False)[:10]
    most_bought.reset_index(inplace=True)
    #Visualize Most Bought Items
    most_bought.plot(kind="bar",x="Description",y="Quantity",figsize=(12,6),xlabel="Item_Name",ylabel="Count",title="Most Bought Items")
    display(most_bought)

most_bought_items(data)

#Total sales and revenue in months
def sales_in_months(data):
    """
    Input :
    data : Input Dataframe
    __________________________________
    Output: Sales in each Month
    __________________________________
    """
    total_sales=data.groupby(pd.Grouper(key="InvoiceDate",freq='M')).sum()
    total_sales.reset_index(inplace=True)
    total_sales.drop(["Unitprice","Customer ID"],axis=1,inplace=True)
    display(total_sales)
    total_sales.plot( x="InvoiceDate" , y=["Quantity","Totalprice"],figsize=(12,6),grid=True ,title="Total Sales in Each  Months",style="-o")
sales_in_months(data)

#Top 10 Customers
def top_10_customers(data):
    """
    Input :
    data: Input Dataframe
    ___________________________________________________
    Output: Top 10 Customers in whole transaction
    ____________________________________________________
    """
    customers=pd.DataFrame(data.groupby("Customer ID")["Quantity"].sum()).sort_values("Quantity", axis = 0, ascending = False)
    customers.reset_index(inplace=True)
    top_customers = customers.head(10)
    top_customers.plot(kind="bar",x="Customer ID",y="Quantity",figsize=(12,6),xlabel="Customer ID",ylabel="Quantity",title="Top 10 Customers")
    display(top_customers)

customer_analysis = top_10_customers(data)

#Sales per Hours
fig= plt.subplots(figsize=(12, 6))
data.groupby(data["InvoiceDate"].dt.hour)["Quantity"].mean().plot(kind='bar',title="Total Sales per Hour",xlabel="Hour of the day",ylabel="Quantity")

#Create Basket Data
#Create a basket data using transactions from United Kingdom
mybasket = (data[data['Country'] =="United Kingdom"]
          .groupby(['Invoice', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('Invoice'))
display("Basket itemsets",mybasket)

#Encode the Data
#Converting positive values to 1 and negative values to 0
def encoder(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_data = mybasket.applymap(encoder)                        #Apply encoding to dataframe
basket_data.drop("POSTAGE",inplace=True,axis=1)
basket_data = basket_data[(basket_data > 0).sum(axis=1) >= 2]   #filter out the transactions that have bought more than 1 items only
basket_data

#Training the model
#Generate Frequent Itemset
def frequent(basket_items):
    """
    Input :
    basket_items : Items in the basket for each transaction
    _____________________________________________________
    Output:
    frequent_itemsets: Frequently bought itemset in the whole transaction
    _______________________________________________________
    """
    frequent_data = apriori(basket_items, min_support=0.03, use_colnames=True)
    frequent_itemsets=frequent_data.sort_values("support", axis = 0, ascending = False)
    frequent_itemsets.reset_index(drop=True,inplace = True)
    return frequent_itemsets

frequent_itemset = frequent(basket_data)
frequent_itemset.head()

#plot top 10 most frequently bought items
frequent_bought=frequent_itemset.head(10)
frequent_bought.plot(kind="bar",x="itemsets",y="support",figsize=(12,6),xlabel="Item_Name",ylabel="support",title="Top 10 Most Frequently Bought Items")

#Find association between fequently bought items
rules = association_rules(frequent_itemset, metric="lift", min_threshold=1)
display(rules)

#plot leverage vs confidence
fig=plt.figure(figsize=(10,6))
plt.scatter(rules["leverage"], rules["confidence"], alpha=0.5)
plt.xlabel("leveraget")
plt.ylabel("confidence")
plt.title("leverage vs Confidence")
plt.show()

#plot support and confidence
fig = plt.figure(figsize=(10,6))
sns.scatterplot(x="support",y="confidence",data=rules,size="lift")
plt.title("Support vs Confidence")
plt.show()

#plot support vs lift
fig=plt.figure(figsize=(10,6))
plt.scatter(rules["support"], rules["lift"], alpha=0.5)
plt.xlabel("support")
plt.ylabel("lift")
plt.title("Support vs Lift")
plt.show()

#Visualise Heatmap with Lift
# Convert antecedents and consequents into strings
rules['antecedents'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['consequents'] = rules['consequents'].apply(lambda a: ','.join(list(a)))
# Relation between association rules represented using Lift value
fig=plt.figure(figsize=(10,8))
rule_data = rules.pivot(index="antecedents",columns="consequents",values="support")
sns.heatmap(rule_data,annot=True,cmap="magma",linecolor="black",linewidths=.01)

#Parallel Coordinate Plot
rules_1 = association_rules(frequent_itemset, metric="lift", min_threshold=1)
def rules_to_coordinates(rules):
    rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent)[0])
    rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent)[0])
    rules['rule'] = rules.index
    return rules[['antecedent','consequent','rule']]
# Convert rules into coordinates suitable for use in a parallel coordinates plot
coords = rules_to_coordinates(rules_1)
# Generate parallel coordinates plot
plt.figure(figsize=(4,8))
parallel_coordinates(coords, 'rule')
plt.legend([])
plt.grid(True)
plt.show()
