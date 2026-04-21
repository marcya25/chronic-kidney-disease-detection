# %% [markdown]
# # Data  Loading

# %%
import pandas as pd
data=pd.read_csv('kidney_disease.csv')
data

# %% [markdown]
# # Exploratory Data Analysis

# %%
data.info()

# %%
data.describe()

# %%
data.head() # first five rows



# %%
data.tail() # last five rows

# %%
data.dtypes

# %%
data.shape  # dimension;number of rows and colums

# %% [markdown]
#  # Data Cleaning

# %%
data.isnull  #to check for missing values


# %%

total_missing = data.isnull().sum().sum()
print("Total missing values in dataset:", total_missing)

# %%
datanew=data.dropna() #removing missing values and storing it in a new variable called datanew

# %%
datanew.describe()

# %%
datanew.shape

# %% [markdown]
# # Data Visualization

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(x="dm",y="age",hue="classification",data=datanew)
plt.grid()
plt.title("a graph of diabetes against age with a hue of classification")
plt.savefig("A.png")

# %%
counts = datanew['classification'].value_counts()

# Plot with custom colors
counts.plot(kind='bar', color=['blue', 'red'])  

plt.title('Distribution of classification')
plt.xlabel('classification')
plt.ylabel('Count')
plt.tight_layout()
plt.title("a graph number of ckd and notckd of the classification")
plt.savefig("B.png")
plt.show()

# %%
sns.boxplot(x="bp",y="sg",data=datanew)
plt.xlabel("bp")
plt.ylabel("sg")
plt.title("A box plot of bp against sg")
plt.grid()

plt.savefig("C.png")
plt.show()

# %%
import numpy as np

corr = datanew.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap="RdBu")
plt.title("a Heat Map of the numeric features" )
plt.savefig("D.png")
plt.show()

# %%
sns.lineplot(x = "age",y = "su",data = datanew)
plt.grid()
plt.title("A line plot age against urine sugar level")
plt.xlabel("Age")
plt.ylabel("urine sugar level")
plt.savefig("A line plot  of age and urine sugar level.png") 
plt.show()

# %%
sns.histplot(datanew["classification"], kde=True)
plt.grid()
plt.title("a graph of diabetes against age with a hue of classification")

plt.show()

# %%

plt.grid()
plt.scatter(datanew["age"],datanew["bp"],color='green')
plt.title("A scatter graph of age and blood pressure")
plt.xlabel("age")
plt.ylabel("blood pressure")
plt.savefig("A scatter  graph of age and blood pressure.png")
plt.show()

# %%
sns.scatterplot(x="bp", y="sod", hue="classification", data=datanew)
plt.grid()
plt.title("A scatter plot of bp against sod")

plt.savefig("f.png")
plt.show()

# %%


sns.lineplot(x="bp", y="sc", data=datanew)
plt.grid()
plt.title("A line plot of pb against sc")
plt.savefig("g.png")
plt.show()

# %%
ckd_data = datanew[datanew["classification"] == "ckd"]

# Count anemia cases
ane_counts = ckd_data["ane"].value_counts()

plt.figure()
plt.pie(ane_counts, labels=ane_counts.index, autopct="%1.1f%%")
plt.title("Anemia Distribution Among CKD Patients")



plt.savefig("anemia distribution among ckd patients.png")
plt.show()

# %% [markdown]
# #  covering objects to numbers

# %%
import pandas as pd

# Convert numeric-like object columns
datanew['pcv'] = pd.to_numeric(datanew['pcv'], errors='coerce')
datanew['wc'] = pd.to_numeric(datanew['wc'], errors='coerce')
datanew['rc'] = pd.to_numeric(datanew['rc'], errors='coerce')

# Convert yes/no columns to 1/0
datanew['htn'] = datanew['htn'].map({'yes': 1, 'no': 0})
datanew['dm'] = datanew['dm'].map({'yes': 1, 'no': 0})
datanew['cad'] = datanew['cad'].map({'yes': 1, 'no': 0})
datanew['pe'] = datanew['pe'].map({'yes': 1, 'no': 0})
datanew['ane'] = datanew['ane'].map({'yes': 1, 'no': 0})

# Convert appetite (good/poor)
datanew['appet'] = datanew['appet'].map({'good': 1, 'poor': 0})

# Convert classification (target variable)
datanew['classification'] = datanew['classification'].map({'ckd': 1, 'notckd': 0})

# Check datatypes
print(datanew.dtypes)

# %% [markdown]
# # Building the linear regression model

# %% [markdown]
# # spliting the dataset

# %%
from sklearn.model_selection import train_test_split# sk learn learns pattern and make predictions

X = datanew[['sc','al','hemo','bp','dm','htn','age','appet','ane']]  # choose relevant features
y = datanew['classification']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  #split datat and keeps it splitted
)

# %% [markdown]
# # scaling features

# %%
from sklearn.preprocessing import StandardScaler  #normalization

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# # training linear regression model

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)



# %%
import joblib

joblib.dump(model, "ckd_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# %% [markdown]
# # evaluation of model

# %%
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", model.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %% [markdown]
# # Building detection model

# %%
def early_ckd_detection(sc, al, hemo, bp, dm, htn, age, appet, ane):
    input_data = [[sc, al, hemo, bp, dm, htn, age, appet, ane]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return "High Risk of CKD" if prediction[0] == 1 else "Low Risk of CKD"

# %% [markdown]
# # testing the model

# %%
result = early_ckd_detection(
    sc=1.8, 
    al=4, 
    hemo=9, 
    bp=150, 
    dm=1, 
    htn=1, 
    age=60, 
    appet=0, 
    ane=1
)
print(result)  # Should print "High Risk of CKD" or "Low Risk of CKD"

# %%
result = early_ckd_detection(
    sc=0.5, 
    al=3, 
    hemo=3, 
    bp=78, 
    dm=0, 
    htn=1, 
    age=50, 
    appet=1, 
    ane=1)
print(result)

# %%
import joblib

joblib.dump(model, "ckd_model.pkl")
joblib.dump(scaler, "scaler.pkl")


