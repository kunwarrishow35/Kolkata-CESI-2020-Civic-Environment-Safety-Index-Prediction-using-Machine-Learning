import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

#________________Loading and editing AQI dataset_____________________
df = pd.read_excel("kolkata_aqi_2020.xlsx")
new_df = df.copy()
print(new_df.head())
print(new_df.columns)

mapping1 = {
    "Good": 5,
    "Satisfactory": 4,
    "Moderate": 3,
    "Poor": 2,
    "Very Poor": 1,
    "Severe": 1
}

new_df["EQI_Score"] = new_df["AQI_Bucket"].map(mapping1)
print(new_df.head())


new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")  #converts the written dates to actual real time dates so we can access by date, month or year
new_df["Month"] = new_df["Date"].dt.month  #from each date take only month                      #errors=coerce handles absurd data and return NaN instead of crashing

print(new_df[["Date", "Month", "EQI_Score"]].head())  

#___________________Crime dataSet______________________________________

df2 = pd.read_excel("crimes_kolkata_2020.xlsx")
crime_df = df2.copy()
print(crime_df.head())

crime_df["Date of Occurrence"] = pd.to_datetime(crime_df["Date of Occurrence"], errors="coerce")
crime_df["Month"] = crime_df["Date of Occurrence"].dt.month 
print(crime_df.head())

print(crime_df["Crime Description"].unique())   #all types of crimes

#mapping them
mapping2 = {
    # Very violent / life-threatening (score 1)
    "HOMICIDE": 1,
    "KIDNAPPING": 1,
    "SEXUAL ASSAULT": 1,
    "ROBBERY": 1,
    "ASSAULT": 1,
    "DOMESTIC VIOLENCE": 1,

    # Serious crimes (score 2)
    "ARSON": 2,
    "FIREARM OFFENSE": 2,
    "EXTORTION": 2,
    "BURGLARY": 2,
    "VEHICLE - STOLEN": 2,
    "DRUG OFFENSE": 2,

    # Moderate crimes (score 3)
    "FRAUD": 3,
    "IDENTITY THEFT": 3,
    "CYBERCRIME": 3,
    "COUNTERFEITING": 3,
    "ILLEGAL POSSESSION": 3,
    "SHOPLIFTING": 3,

    # Lower-level public order offenses (score 4)
    "VANDALISM": 4,
    "TRAFFIC VIOLATION": 4,
    "PUBLIC INTOXICATION": 4,
}

crime_df["Crime_Safety_Score"] = crime_df["Crime Description"].map(mapping2)

print(crime_df[["Crime Description", "Crime_Safety_Score"]].head())


#Avg EQI per month
air_month = new_df.groupby("Month")["EQI_Score"].mean().reset_index()
print(air_month)

#Crime safety per month
crime_month = crime_df.groupby("Month")["Crime_Safety_Score"].mean().reset_index()
print(crime_month)

#Combining tigether on month
combined = pd.merge(air_month, crime_month, on="Month")
print(combined)

combined["CESI_Score"] = (combined["EQI_Score"] + combined["Crime_Safety_Score"]) / 2    #CSEI = Civic Environment and Safety Index
    


def cesi_label(score):
    if score >= 3.5:
        return "Good"
    elif score >= 2.5:
        return "Moderate"
    else:
        return "Poor"

combined["CESI_Label"] = combined["CESI_Score"].apply(cesi_label)
print(combined)

#making predictions
X = combined[["EQI_Score", "Crime_Safety_Score"]]
y = combined["CESI_Label"]

log_model = LogisticRegression()
log_model.fit(X,y)
print("Logistic Regression Prediction", log_model.predict(X))

#Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X,y)
print("Random Forest prediction", rf_model.predict(X))

#Vizualization
#Heat map
plt.figure(figsize=(6,4))
sns.heatmap(combined[['EQI_Score', "Crime_Safety_Score", "CESI_Score"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#line plot
plt.figure(figsize=(8,4))
plt.plot(combined["Month"], combined["Crime_Safety_Score"], label="CESI")

plt.xlabel("Month")
plt.ylabel("Score")
plt.title("Month trend of Score")
plt.legend()
plt.show()

#bar chat
plt.figure(figsize=(5,4))
sns.countplot(x=combined["CESI_Label"])
plt.title("distribution of CESI catagory")
plt.show()



#making predictions
eqi = float(input("Enter EQI: "))
crime=float(input("Enter crime score: "))
Input = [[eqi, crime]]
print("your probable CESI is", rf_model.predict(Input)) 






