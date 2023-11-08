
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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns
import warnings

# Suppress the warning
warnings.filterwarnings("ignore", category=UserWarning)
def get_first_element(str):
     return str.split(',')[0]


le_country = LabelEncoder() 
le_ages = LabelEncoder()
script_dir = os.path.dirname(__file__) 
rel_path = "netflix_titles.csv"
abs_file_path = os.path.join(script_dir, rel_path)

df = pd.read_csv(abs_file_path)
data = pd.read_csv(abs_file_path)

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

df = df.filter(items=['country', 'ages'])
df = df.dropna()



df['ages']= le_ages.fit_transform(df['ages'])
le_ages_mapping = dict(zip(le_ages.classes_, le_ages.transform(le_ages.classes_)))
print(le_ages_mapping)
df['country'] = le_country.fit_transform(df['country'])
le_country_mapping = dict(zip(le_country.classes_, le_country.transform(le_country.classes_)))




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

predicted_age_dictionary = {country: age_category for country, age_category in country_age_predictions.items()}




countries = list(predicted_age_dictionary.keys())
age_categories = list(predicted_age_dictionary.values())

       
country = [item.split(',')[0] for item in countries]
    

    
print(country)
print(age_categories)
'''
fig, ax = plt.subplots(figsize=(100,6))


bar_chart = ax.bar(country, age_categories)


plt.xlabel('Countries')
plt.ylabel('Age Category')
plt.title('Predicted Age Categories by Country')
plt.xticks(rotation=90)  


plt.tight_layout()
plt.show()
'''


data= data['listed_in'].astype(str).apply(lambda s : s.replace('&',' ').replace(',', ' ').split()) 

test = data
mlb = MultiLabelBinarizer()
res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_)
corr = res.corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(35, 34))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


import networkx as nx

stocks = corr.index.values
cor_matrix = np.asmatrix(corr)
G = nx.DiGraph(cor_matrix)
G = nx.relabel_nodes(G,lambda x: stocks[x])
G.edges(data=True)

def create_corr_network(G, corr_direction, min_correlation):
    H = G.copy()
    for stock1, stock2, weight in G.edges(data=True):
        if corr_direction == "positive":
            if weight["weight"] <0 or weight["weight"] < min_correlation:
                H.remove_edge(stock1, stock2)
        else:
            if weight["weight"] >=0 or weight["weight"] > min_correlation:
                H.remove_edge(stock1, stock2)
                
    edges,weights = zip(*nx.get_edge_attributes(H,'weight').items())
    weights = tuple([(1+abs(x))**2 for x in weights])
    d = nx.degree(H)
    nodelist, node_sizes = zip(*d)
    positions=nx.circular_layout(H)
    
    plt.figure(figsize=(10,10), dpi=72)
    
    nx.draw_networkx_nodes(H,positions,node_color='#DA70D6',nodelist=nodelist,
                           node_size=tuple([x**2 for x in node_sizes]),alpha=0.8)
    
    nx.draw_networkx_labels(H, positions, font_size=8, 
                            font_family='sans-serif')
    
    if corr_direction == "positive": edge_colour = plt.cm.GnBu 
    else: edge_colour = plt.cm.PuRd
        
    nx.draw_networkx_edges(H, positions, edgelist=edges,style='solid',
                          width=weights, edge_color = weights, edge_cmap = edge_colour,
                          edge_vmin = min(weights), edge_vmax=max(weights))
    plt.axis('off')
    plt.show() 
    
create_corr_network(G, 'positive', 0.3)
create_corr_network(G, 'positive', -0.3)


