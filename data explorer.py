# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.ensemble import  RandomForestClassifier

filename=input("Enter the Name Of the File You Want To Explore : ")
print("\n------ Fetching/Reading The Data From The Data Set")
s_data = pd.read_csv(filename)

print("------ Printing the first 5 rows from the dataset to check the whether the data has been fetched properly or not :")
print(s_data.head(5)) 

pd.set_option("display.max.columns", None)

print("\n------ Total Number Of Rows and Colums In The Dataset :\n",s_data.shape)

print("\n------ See the all the Columns Present In The Dataset :\n")
print(s_data.columns.values )

print("\n------ The Information about each Column of the dataset (including their datatypes and the type of values (Null/Not Null) :\n")
print("\n------ Checking whether the Dataset has any Data Quality Issues such as Having NULL Values\n")
print(s_data.info())

print("\n------ Finding out the count, mean, standard deviation, minimum and maximum values and the quantiles of the data in the Dataset :\n")
print(s_data.describe())

print("\n------ To check the Missing Values In the Dataset :\n")
sns.heatmap(s_data.isnull(),cbar=False,yticklabels=False)
plt.title("Check the Missing Values In the Dataset")
plt.grid(True,color='r')
plt.show()

print("\n------ Dropping the columns 'resultQualifier.notation' and 'codedResultInterpretation.interpretation':\n")
s_data=s_data.drop('resultQualifier.notation',axis=1)
s_data=s_data.drop('codedResultInterpretation.interpretation',axis=1)

print("\n------ To check the Missing Values In the Dataset :\n")
sns.heatmap(s_data.isnull(),cbar=False,yticklabels=False)
plt.title("Check the Missing Values In the Dataset")
plt.grid(True,color='r')
plt.show()

print("\n------ Finding out the Correlations Between The Different Variables Using the HEAT MAP :\n")
correlations=s_data.corr()
sns.heatmap(correlations,annot=True)
plt.title("Correation Between The Different Variables :\n")
plt.show()

print("\n------ Finding out that the Data Of Which Areas Are Present In the Dataset :\n")
print(s_data['sample.samplingPoint.label'].unique())

print("\n------ Counting the Records Of Each Area Present In The Dataset :\n")
print(s_data['sample.samplingPoint.label'].value_counts())

print("\n------ Finding and Counting the different Sampling Material Types Present In the Dataset :\n")
print(s_data['sample.sampledMaterialType.label'].value_counts())

print("\n------ Finding and Counting each Sampling Purposes Present In the Dataset :\n")
print(s_data['sample.purpose.label'].value_counts())

print("\n------ Finding and Counting the different Determinand Units Present In the Dataset :\n")
print(s_data['determinand.unit.label'].value_counts())

print("\n------ Finding and Counting each Determinants Present In the Dataset :\n")
print(s_data['determinand.definition'].value_counts())

print("\n------ Finding Out that How Many Samples are Compliance Samples In the Dataset :\n")
print(s_data['sample.isComplianceSample'].value_counts())

plt.figure(figsize=(17,9))
plt.title('Comparison between the samples (Whether Compliance or not) based on the result and the dterminand unit')
sns.scatterplot(x=s_data['result'],y=s_data['determinand.unit.label'],hue =s_data['sample.isComplianceSample'],s=50)
plt.show()








