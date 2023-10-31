
import pandas as pd
import os
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score




le_country = LabelEncoder() 
le_ages = LabelEncoder()
script_dir = os.path.dirname(__file__) 
rel_path = "netflix_titles.csv"
abs_file_path = os.path.join(script_dir, rel_path)

df = pd.read_csv(abs_file_path)


df = df[df['rating'].isin(['PG-13', 'TV-MA', 'PG', 'TV-14', 'TV-PG', 'TV-Y', 'R', 'TV-G', 'TV-Y7', 'G', 'NC-17', 'NR', 'TV-Y7-FV', 'UR'])]

ratings_ages = {
    'PG-13': 'Teens',
    'TV-MA': 'Adults',
    'TV-14': 'Teens',
    'TV-Y7': 'Older Kids',
    'PG': 'Older Kids',
    'R': 'Adults',
    'TV-PG': 'Older Kids',
    'TV-Y': 'Kids',
    'TV-G': 'Kids',
    'G': 'Kids',
    'NC-17': 'Adults',
    'NR': 'Adults',
    'TV-Y7-FV': 'Older Kids',
    'UR': 'Adults'
    }

df['ages'] = df['rating'].replace(ratings_ages)
df['ages'].unique()


df['ages']= le_ages.fit_transform(df['ages'])
le_ages_mapping = dict(zip(le_ages.classes_, le_ages.transform(le_ages.classes_)))
print(le_ages_mapping)
df['country'] = le_country.fit_transform(df['country'])
le_country_mapping = dict(zip(le_country.classes_, le_country.transform(le_country.classes_)))

   
X = df[['country']]
y = df['ages']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  

X = df[['country']]
y = df['ages']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)


model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)



unique_countries = df['country'].unique()


country_age_predictions = {}

for encoded_country in unique_countries:
    
    encoded_country_2D = encoded_country.reshape(1, -1)

   
    predicted_age_code = model.predict(encoded_country_2D)

    
    predicted_age_label = le_ages.inverse_transform(predicted_age_code)

   
    original_country_label = le_country.inverse_transform([encoded_country])[0]

    country_age_predictions[original_country_label] = predicted_age_label[0]


for country, age_category in country_age_predictions.items():
    print(f"Predicted age category for {country}: {age_category}")








