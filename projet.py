import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_brut=pd.read_csv('C:/Users/orthi/Documents/projet-traitement-donnée/Teen_Mental_Health_Dataset.csv')


print(df_brut.head())


df = pd.get_dummies(df_brut, columns=['social_interaction_level','gender','platform_usage'], dtype=int)
df.info
print(df)




# Afficher la répartition des heures de sommeil

sns.histplot(df['sleep_hours'], kde=True) # kde=True ajoute la courbe de tendance
plt.title("Répartition des heures de sommeil")
plt.show()

sns.boxplot(x='depression_label', y='stress_level', data=df)

plt.title("Niveau de stress selon la dépression")
plt.show()
#scatter= nuage de point hue='depression_label' permet de colorer les points selon qu'ils sont dépressifs ou non !

sns.scatterplot(x='screen_time_before_sleep', y='sleep_hours', hue='depression_label', data=df)
plt.title("Sommeil vs Temps d'écran")
plt.show()
# Attention : df.corr() ne marche que sur des colonnes contenant des nombres !
matrice_correlation = df.corr()

# Créer une "Heatmap" (Carte de chaleur)
plt.figure(figsize=(10, 8)) # Agrandir l'image
sns.heatmap(matrice_correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation de la santé mentale")
plt.show()
