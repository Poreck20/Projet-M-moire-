#!/usr/bin/env python
# coding: utf-8

# ## I- Importation des données 

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib as plt 
import os 
import datetime as dt 
#!pip install pyproj
from pyproj import Proj, transform 
from scipy.spatial import cKDTree
import seaborn as sb 
import matplotlib.pyplot as plt 
import datetime as dt
import statsmodels.api as sm 


# In[170]:


cd "C:\Documents alpha\Projet memoire\Projet memoire" 


# ### 1- Importation des données de DVF

# In[171]:


dvf_2020 = pd.read_csv("C:/Documents alpha/Projet memoire/Projet memoire/full.csv/full_2020.csv", sep=",", encoding="utf-8", parse_dates=["date_mutation"])
dvf_2020.head()


# In[173]:


dvf_2020[dvf_2020["nombre_pieces_principales"]>100]


# In[172]:


dvf_2021 = pd.read_csv("C:/Documents alpha/Projet memoire/Projet memoire/full.csv/full_2021.csv", sep=",", encoding="utf-8",  parse_dates=["date_mutation"])
dvf_2021.head()


# In[5]:


dvf_2022 = pd.read_csv("C:/Documents alpha/Projet memoire/Projet memoire/full.csv/full_2022.csv", sep=",", encoding="utf-8",  parse_dates=["date_mutation"])
dvf_2022.head()


# In[6]:


dvf_2023 = pd.read_csv("C:/Documents alpha/Projet memoire/Projet memoire/full.csv/full_2023.csv", sep=",", encoding="utf-8",  parse_dates=["date_mutation"])
dvf_2023.head()


# In[7]:


dvf_2024 = pd.read_csv("C:/Documents alpha/Projet memoire/Projet memoire/full.csv/full_2024.csv", sep=",", encoding="utf-8",  parse_dates=["date_mutation"])
dvf_2024.head()


# In[8]:


dvf = pd.concat([dvf_2020, dvf_2021, dvf_2022, dvf_2023, dvf_2024])
dvf.tail()


# In[9]:


dvf.dtypes


# #### Nettoyage

# In[10]:


dvf1=dvf[["id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere","nom_commune", "code_departement", "type_local",
"code_type_local", "surface_reelle_bati", "nombre_pieces_principales", "surface_terrain", "latitude", "longitude"]]


# In[36]:


dvf1=dvf1[
    (dvf1["nature_mutation"]=="Vente") & 
    (dvf1["type_local"].isin(["Appartement", "Maison"])) & 
    (dvf1["code_departement"].isin([75, 77, 78, 91, 92, 93, 94, 95]))]
dvf1.sample(2)


# ### 2- Importation des données des stations et gares 

# In[97]:


df_gare= pd.read_excel("C:/Documents alpha/Projet memoire/Projet memoire/gares_de_idf.xlsx")
df_gare.sample(5)


# In[98]:


df_gare.dtypes


# In[99]:


df_gare=df_gare[["codeunique", "nom_long", "Geo Point", "mode_"]]
df_gare["latitude"] = df_gare["Geo Point"].apply(lambda x : float(x.split(',')[0]))
df_gare["longitude"] = df_gare["Geo Point"].apply(lambda x : float(x.split(',')[1]))
df_gare.head()


# ### 3- Importation des données de police et gendarmerie 

# In[77]:


police = pd.read_csv("C:/Documents alpha/Projet memoire/Projet memoire/police.csv", sep=";", encoding="utf-8")
police.head()
len(police)


# In[78]:


gendarmerie= pd.read_csv("C:/Documents alpha/Projet memoire/Projet memoire/gendarmerie.csv", sep=";", encoding="utf-8")
gendarmerie.head()
len(gendarmerie)


# In[79]:


pg = pd.concat([police, gendarmerie])
pg.sample(4)


# In[18]:


pg.dtypes


# In[80]:


pg = pg[["service", "departement", "commune",  "geocodage_y_GPS", "geocodage_x_GPS"]]


# In[129]:


pg=pg[pg["departement"].isin(["75", "77", "78", "91", "92", "93", "94", "95"])]
print(len(pg))
pg = pg.rename(columns={"geocodage_x_GPS" : "longitude", "geocodage_y_GPS" : "latitude"})
pg = pg.reset_index(drop=True)
pg.head(3)


# ### 4- Importations des données de lieux touristiques 

# In[138]:


tourisme = pd.read_excel("C:/Documents alpha/Projet memoire/Projet memoire/principaux-sites-touristiques-en-ile-de-france0.xlsx")
tourisme.head(2)


# In[139]:


tourisme.dtypes


# In[140]:


tourisme=tourisme[["typo_niv3", "dep", "Geo Point"]]
tourisme.head()


# In[141]:


tourisme['latitude'] = tourisme['Geo Point'].apply(lambda x: float(x.split(',')[0]))
tourisme['longitude'] = tourisme['Geo Point'].apply(lambda x: float(x.split(',')[1]))
tourisme.head()


# ### 5- Importations des données des grands lieux de commerce 

# In[158]:


commerce = pd.read_csv("C:/Documents alpha/Projet memoire/Projet memoire/Centre commerciaux.csv", sep=";", encoding="utf-8")
commerce.sample(2)


# In[159]:


commerce.dtypes


# In[160]:


commerce = commerce[["shop", "name", "addr-city", "the_geom"]]
commerce.sample(3)


# In[161]:


commerce = commerce[commerce["shop"].isin(["supermarket"])]
commerce.sample(3)


# In[162]:


commerce['x'] = commerce['the_geom'].str.extract(r'POINT \(([^ ]+)')[0].astype(float)
commerce['y'] = commerce['the_geom'].str.extract(r'POINT \([^ ]+ ([^ ]+)\)')[0].astype(float)
commerce.head()


# In[163]:


get_ipython().system('pip install pyproj')
from pyproj import Proj, transform 

proj_3857 = Proj('EPSG:3857') 
proj_4326 = Proj('EPSG:4326')

def convertir_coords(row):
    lon, lat = transform(proj_3857, proj_4326, row['y'], row['x'])
    return pd.Series({'longitude': lon, 'latitude': lat})


commerce[['longitude', 'latitude']] = commerce.apply(convertir_coords, axis=1)


# In[164]:


commerce = commerce[["shop", "name", "addr-city","latitude",  "longitude"]]

commerce = commerce.dropna(subset=["latitude", "longitude"])
commerce = commerce.reset_index(drop=True)

commerce.head()


# ## II- Calcul de la distance 

# ### 1- Distance par rapport aux gares 

# In[101]:


# Traitement valeur manquante 
dvf1 = dvf1.dropna(subset=['latitude', 'longitude'])
df_gare= df_gare.dropna(subset=['latitude', 'longitude'])


# In[111]:


from scipy.spatial import cKDTree


# In[119]:


gares_coords = df_gare[["latitude", "longitude"]].values

tree_gares = cKDTree(gares_coords)


biens_coords = dvf1[['latitude', 'longitude']].values

distances_min = []
gare = []
for bien_coord in biens_coords:

    dist, idx = tree_gares.query(bien_coord) # Rechercher la distance minimale à la gare la plus proche
    distances_min.append(dist)
    
    gare.append(df_gare['nom_long'][idx])

dvf1['dist_gare'] = distances_min
dvf1["gare_proche"] = gare

dvf1.head(2)


# ### 2- Distance par rapport aux Commissariats et Gendarmeries

# In[130]:


pg_coords = pg[["latitude", "longitude"]].values

tree_pg = cKDTree(pg_coords)


biens_coords = dvf1[['latitude', 'longitude']].values

distances_min = []
police = []
for bien_coord in biens_coords:

    dist, idx = tree_pg.query(bien_coord) # Rechercher la distance minimale à la police la plus proche
    distances_min.append(dist)
    
    police.append(pg['service'][idx])

dvf1['dist_police'] = distances_min
dvf1["police_proche"] = police

dvf1.head(2)


# ###  3- Distance par rapport aux lieux touristiques

# In[142]:


tourisme_coords = tourisme[["latitude", "longitude"]].values

tree_tourisme = cKDTree(tourisme_coords)


biens_coords = dvf1[['latitude', 'longitude']].values

distances_min = []
lieux = []
for bien_coord in biens_coords:

    dist, idx = tree_tourisme.query(bien_coord) # Rechercher la distance minimale au touriste le plus proche
    distances_min.append(dist)
    
    lieux.append(tourisme['typo_niv3'][idx])

dvf1['dist_tourisme'] = distances_min
dvf1["tourisme_proche"] = lieux

dvf1.head(3)


# ### 4- Distance par rapport aux Centres Commerciaux 

# In[166]:


commerce_coords = commerce[["latitude", "longitude"]].values

tree_commerce = cKDTree(commerce_coords)


biens_coords = dvf1[['latitude', 'longitude']].values

distances_min = []
centre = []
for bien_coord in biens_coords:

    dist, idx = tree_commerce.query(bien_coord) # Rechercher la distance minimale au touriste le plus proche
    distances_min.append(dist)
    
    centre.append(commerce['name'][idx])

dvf1['dist_commerce'] = distances_min
dvf1["commerce_proche"] = centre

dvf1.head(5)


# In[168]:


dvf1["dist_moyenne"] = (dvf1["dist_gare"] + dvf1["dist_police"] + dvf1["dist_tourisme"] + dvf1["dist_commerce"])/4


# In[171]:


dvf1.sample(5)


# In[ ]:


# BASE DE DONNEE FINALE 

dvf1.to_csv("C:/Documents alpha/Projet memoire/Projet memoire/Base de donne finale/table_finale.csv", index=False, encoding="utf-8")


# In[142]:


table_finale = pd.read_csv("C:/Documents alpha/Projet memoire/Projet memoire/Base de donne finale/table_finale.csv", sep=",", encoding="utf-8")


# In[168]:


table_finale.sample(5)


# In[146]:


table_finale.dtypes


# In[51]:


table_finale= table_finale.dropna(subset=["valeur_fonciere"])


# In[52]:


table_finale["prix_par_m2"] =table_finale["valeur_fonciere"]/table_finale["surface_reelle_bati"]


# ## III- Statistique descriptives 

# ### 1- Statistiques générales

# In[147]:


table_finale.describe().T


# In[169]:


table_finale[table_finale["nombre_pieces_principales"]>=50]


# In[162]:


sb.displot(table_finale, x= "nombre_pieces_principales", kind="kde")


# ### 2- Distribution des differentes variables 

# In[8]:


sb.displot(table_finale, x="valeur_fonciere", kind="kde")
plt.title("Distribution de la valeur fonciere")


# In[179]:


sb.displot(table_finale, x="log_valeur_fonciere", kind='kde')
plt.title("Distribution du log de la valeur foncicere")


# In[287]:


sb.displot(table_finale, x="dist_moyenne", kind="kde")


# In[294]:


sb.displot(table_finale, x=np.log(table_finale["dist_gare"]), kind="kde")


# In[9]:


sb.displot(table_finale, x=np.log(table_finale["dist_commerce"]), kind="kde")


# In[298]:


sb.displot(table_finale, x=np.log(table_finale["dist_police"]), kind="kde")


# In[299]:


sb.displot(table_finale, x=np.log(table_finale["dist_tourisme"]), kind="kde")


# In[292]:


sb.displot(table_finale, x=np.log(table_finale["surface_reelle_bati"]), kind="kde")


# In[305]:


sb.displot(table_finale, x=np.log(table_finale["surface_terrain"]), kind="kde")


# In[9]:


sb.scatterplot(table_finale, x="valeur_fonciere", y="surface_reelle_bati",  hue="type_local")


# In[238]:


sb.scatterplot(table_finale, x="valeur_fonciere", y="surface_terrain",  hue="type_local")


# In[61]:


sb.countplot(data=table_finale, x="code_departement", hue="type_local")


# ### 3- Analyse du prix de l'immobilier 

# In[55]:


sb.barplot(data=table_finale, x="code_departement", y="valeur_fonciere", errorbar=None, estimator=sum)


# In[177]:


sb.barplot(data=table_finale, x="type_local", y="valeur_fonciere", errorbar=None)


# ### 4- Influence de la distance sur le prix 

# In[13]:


table_dist = table_finale[["valeur_fonciere", "dist_gare","dist_commerce", "dist_tourisme", "dist_police"]]
corr = table_dist.corr()
sb.heatmap(corr,annot=True)
print(corr)


# In[80]:


table_local = test[["valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales", "surface_terrain"]]
corr2 = table_local.corr()
sb.heatmap(corr2, annot=True)
print(corr2)


# ### 5- Analyse par localisation géographique

# In[107]:


# Prix par commune 
table_finale.groupby('nom_commune')['valeur_fonciere'].mean().sort_values(ascending=False)


# ### 6- Analyse temporelle

# In[122]:


table_finale["annee_mutation"] =pd.to_datetime(table_finale["date_mutation"]).dt.year
table_finale.groupby('annee_mutation')['id_mutation'].count().plot(kind='bar', title='Nombre de ventes par mois')


# In[124]:


table_finale['mois_mutation'] = pd.to_datetime(table_finale['date_mutation']).dt.month
table_finale.groupby('mois_mutation')['id_mutation'].count().plot(kind='bar', title='Nombre de ventes par mois')


# ### 6- Segementation par infrastructure proches

# In[129]:


# Prix moyen par infrasturcture 
table_finale.groupby('gare_proche')['valeur_fonciere'].mean().sort_values(ascending=False)


# In[147]:


# Distance moyenne par infrastructure

print(table_finale.groupby('type_local')[['dist_gare', 'dist_police', 'dist_tourisme', 'dist_commerce']].mean())

print(table_finale.groupby('code_departement')[['dist_gare', 'dist_police', 'dist_tourisme', 'dist_commerce']].mean())


# ### 7- Comparaisons inter-départementales

# In[24]:


table_finale.groupby('code_departement')['surface_reelle_bati'].mean()


# In[162]:


sb.barplot(data=table_finale, x="code_departement", y="surface_reelle_bati")


# In[59]:


code_departement = [75, 77, 78, 91, 92, 93, 94, 95]
for col in code_departement :
    table_finale[f"{col}"]= (table_finale["code_departement"] == col).astype(int)
table_finale.sample(5)


# ## IV- Modèle Econometrique 

# In[121]:


table_finale2 = table_finale.dropna()
y = table_finale2["prix_par_m2"]
x = table_finale2[["surface_reelle_bati", "surface_terrain","nombre_pieces_principales", "dist_gare", 
                   "dist_commerce","dist_tourisme", "dist_police", 
                   "75", "77", "78", "91", "92", "93", "94"]]

X = sm.add_constant(x)

model = sm.OLS(y, X).fit()

print(model.summary())


# ### Modèle en logarithme

# In[123]:


for col in ["prix_par_m2", "valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales", "surface_terrain", 
            "dist_gare", "dist_commerce", "dist_tourisme", "dist_police"]:
    table_finale2[f"log_{col}"] = np.log(table_finale2[col])


# In[66]:


table_finale2.dtypes


# In[129]:


y = table_finale2["log_prix_par_m2"]
x = table_finale2[["log_surface_reelle_bati", "log_surface_terrain", "log_dist_gare",
                   "log_dist_commerce","log_dist_tourisme", "log_dist_police",  
                   "75", "77", "78", "91", "92", "93", "94"]]
X = sm.add_constant(x)

model =sm.OLS(y,X).fit()

print(model.summary())


# In[131]:


for col in ["prix_par_m2", "nombre_pieces_principales","surface_reelle_bati", "nombre_pieces_principales", "surface_terrain", 
            "dist_gare", "dist_commerce", "dist_tourisme", "dist_police"]:
    table_finale2[f"carre_{col}"] = table_finale2[col]**2
table_finale2.head()


# In[134]:


y = table_finale2["log_prix_par_m2"]
x = table_finale2[["surface_reelle_bati", "nombre_pieces_principales",
                   "carre_surface_reelle_bati", "surface_terrain", 
                  "dist_commerce", "carre_dist_commerce", "dist_tourisme", "dist_police","carre_dist_police",  
                   "75", "77", "78", "91", "92", "93", "94"]]

X = sm.add_constant(x)

model = sm.OLS(y, X).fit() 

print(model.summary())


# ## 6- Sans valeurs aberante

# In[136]:


# Borne inferieur 
table_finale3 = table_finale2
for col in ["log_prix_par_m2", "log_surface_reelle_bati", "log_surface_terrain", 
            "log_nombre_pieces_principales", "log_dist_gare",
                   "log_dist_commerce","log_dist_tourisme", "log_dist_police"]:
    table_finale3[f"borneInf_{col}"] = table_finale2[col].mean() - 3*table_finale2[col].std()
table_finale3.head()


# In[138]:


# Borne Superieur 
for col in ["log_prix_par_m2", "log_surface_reelle_bati", "log_surface_terrain", "log_dist_gare","log_nombre_pieces_principales",
                   "log_dist_commerce","log_dist_tourisme", "log_dist_police"]:
    table_finale3[f"borneSup_{col}"] = table_finale3[col].mean() + 3*table_finale3[col].std()
table_finale3.head()


# In[139]:


len(table_finale3)


# In[140]:


for col  in ["log_prix_par_m2", "log_surface_reelle_bati", "log_surface_terrain","log_nombre_pieces_principales", "log_dist_gare",
                   "log_dist_commerce","log_dist_tourisme", "log_dist_police"]:
    table_finale3 = table_finale3[(table_finale3[f"borneInf_{col}"] < table_finale3[col]) & (table_finale3[col] < table_finale3[f"borneSup_{col}"])] 


# In[141]:


print(len(table_finale3))


# In[98]:


sb.displot(data= table_finale2, x = "log_prix_par_m2" , kind = "kde")
plt.title("Distribution de prix avec aberante")
sb.displot(data= table_finale3, x = "log_prix_par_m2" , kind = "kde")
plt.title("Distribution de prix sans aberante") 


# In[79]:


y = table_finale3["log_prix_par_m2"]
x = table_finale3[["log_surface_reelle_bati", "log_surface_terrain", "log_dist_gare",
                   "log_dist_commerce", "log_dist_tourisme", "log_dist_police", 
                   "75", "77", "78", "91", "92", "93", "94"]]

X = sm.add_constant(x)

model = sm.OLS(y, X).fit()

print(model.summary())


# In[113]:


sb.displot(model.resid, kind="kde")
plt.title("Distribution des residus")


# ## V- Machine learning 

# In[80]:


import sklearn
from sklearn.linear_model import LinearRegression    
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# In[81]:


X = table_finale3[["log_surface_reelle_bati", "log_surface_terrain", "log_dist_gare",
                   "log_dist_commerce","log_dist_tourisme", "log_dist_police", 
                   "75", "77", "78", "91", "92", "93", "94"]]
y = table_finale3["log_prix_par_m2"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression() 

model.fit(X_train, y_train)        # entrainement 
test = model.score(X_test, y_test) # test 
prediction = model.predict(X_test) # prediction 

print("Score du modele:", test) # R2
print("R2:", r2_score(y_test, prediction)) # equivalent a 


# In[82]:


# Avec Random forest 
X = table_finale3[["log_surface_reelle_bati", "log_surface_terrain", "log_dist_gare",
                   "log_dist_commerce","log_dist_tourisme", "log_dist_police", 
                   "75", "77", "78", "91", "92", "93", "94"]]
y = table_finale3["log_prix_par_m2"]

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = RandomForestRegressor(n_estimators= 120, random_state=42)
model.fit(X_train, y_train) 
test = model.score(X_test, y_test)
prediction = model.predict(X_test)
print("Score", test)


# In[83]:


sb.scatterplot(x=y_test, y=prediction, alpha=0.3, color="purple") 
plt.ylabel("Valeur predites")
plt.xlabel("Valeur Reelle")


# In[94]:


residus = y_test - prediction
sb.displot(residus, kind="kde")
plt.xlabel("residus")
plt.title("Distribution des residus")


# In[110]:


importance = model.feature_importances_
feature_names= X.columns
df = pd.DataFrame({"variable" : feature_names, "importance" : importance}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 6))
sb.barplot(y = "variable", x = "importance", data =df, palette='viridis')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




