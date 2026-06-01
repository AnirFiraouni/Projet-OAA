import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_brut=pd.read_csv('./Teen_Mental_Health_Dataset.csv')


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

#On définit les variable correle avec la depression.
variables_correlees = [
    ('sleep_hours', 'Heures de sommeil'),
    ('daily_social_media_hours', 'Heures sur les réseaux sociaux'),
    ('stress_level', 'Niveau de stress'),
    ('anxiety_level', "Niveau d'anxiété")
]

sns.set_theme(style="whitegrid", palette="pastel")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# J'ai retiré le y=1.02 qui poussait le titre hors de la fenêtre
fig.suptitle('Relations entre la Dépression et ses variables corrélées', fontsize=16, fontweight='bold')

axes = axes.flatten()

df['Etat_Depressif'] = df['depression_label'].map({0: 'Non (0)', 1: 'Oui (1)'})

for i, (var, title) in enumerate(variables_correlees):
    
    sns.boxplot(x='Etat_Depressif', y=var, data=df, ax=axes[i], palette=['#a1c9f4', '#ffb482'])
    
    # Amélioration : taille des points réduite (size=3) et opacité ajustée (alpha=0.2)
    sns.stripplot(x='Etat_Depressif', y=var, data=df, ax=axes[i], color='black', alpha=0.2, size=3, jitter=True)
    
    # Amélioration : pad=10 décolle légèrement le titre de la boîte
    axes[i].set_title(f"{title} selon l'état dépressif", fontsize=12, pad=10)
    
    # Amélioration : on supprime le label X "Dépression" qui prend de la place pour rien (Oui/Non suffit)
    axes[i].set_xlabel('')
    axes[i].set_ylabel(title, fontsize=10)

plt.tight_layout()

# LA CORRECTION CLÉ : On force des espaces manuels après le tight_layout
# top=0.90 laisse de la place pour le titre principal
# hspace=0.30 écarte la ligne du haut de la ligne du bas
fig.subplots_adjust(top=0.90, hspace=0.30) 

plt.show()