#!/usr/bin/env python
# coding: utf-8

# <a id="top"></a>
# <div style="text-align: center; background: #ff8c00; font-family: 'Montserrat', sans-serif; color: white; padding: 15px; font-size: 30px; font-weight: bold; line-height: 1; border-radius: 20px 20px 0 0; margin-bottom: 20px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);">🚀 Classification de SMS: détection des spams 🚀</div>
# <div style="text-align: center;">
#     <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*mbFBPcPUJD-53v3h.png">
# </div>
# 

# <a id="1"></a>
# <div style="text-align: center; background: #1ED760; font-family: 'Trebuchet MS', Arial, sans-serif; color: white; padding: 15px; font-size: 26px; font-weight: bold; line-height: 1; border-radius: 50% 0 50% 0 / 40px; margin-bottom: 20px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">1. Introduction</div>
# 

# <a id="2"></a>
# <div style="text-align: center; background: #1ED760; font-family: 'Trebuchet MS', Arial, sans-serif; color: white; padding: 15px; font-size: 26px; font-weight: bold; line-height: 1; border-radius: 50% 0 50% 0 / 40px; margin-bottom: 20px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">2. Problem Statement</div>
# 

# « L'objectif principal de ce notebook est de développer un modèle prédictif capable de classifier avec précision les emails en fonction de leur label : <span style="background-color: #FF5733; color: white; font-weight: bold; padding: 3px 6px; border-radius: 3px;">ecole</span>, <span style="background-color: #FF5733; color: white; font-weight: bold; padding: 3px 6px; border-radius: 3px;">pro</span> ou <span style="background-color: #FF5733; color: white; font-weight: bold; padding: 3px 6px; border-radius: 3px;">sport</span>. Nous utiliserons le jeu de données local `dataset_emails_classification_2k.csv`, qui contient des objets et corps d'emails annotés avec leurs labels respectifs. »

# <a id="3"></a>
# <div style="text-align: center; background: #1ED760; font-family: 'Trebuchet MS', Arial, sans-serif; color: white; padding: 15px; font-size: 26px; font-weight: bold; line-height: 1; border-radius: 50% 0 50% 0 / 40px; margin-bottom: 20px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">3. Data Checks to Perform 🔍</div>

# à faire pour plus tard

# <div style="text-align: center; background:  #FF00CC; font-family: 'Trebuchet MS', Arial, sans-serif; color: white; padding: 15px; font-size: 26px; font-weight: bold; line-height: 1; border-radius: 50% 0 50% 0 / 40px; margin-bottom: 20px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">3.1 Import Necessary Libraries</div>
# 

# In[1]:


# Importation des librairi nécessaires
import numpy as np        # For numerical operations
import pandas as pd       # For data manipulation and analysis
import matplotlib.pyplot as plt  # For data visualization
get_ipython().run_line_magic('matplotlib', 'inline')

# Importing WordCloud for text visualization
from wordcloud import WordCloud

# Importing NLTK for natural language processing
import nltk
from nltk.corpus import stopwords    # For stopwords


# Downloading NLTK data (utiliser pour tous ce qui n'est pas deep learning)
nltk.download('stopwords')   # Downloading stopwords data
nltk.download('punkt')       # Downloading tokenizer data


# In[2]:


# Charger le dataset local multiclasse (subject/body/label).
# Utiliser UTF-8 pour rester compatible avec les textes en français.
df = pd.read_csv('dataset_emails_classification_2k_harder.csv', encoding='utf-8')


# In[3]:


styled_df = df.head()
styled_df = styled_df.style.set_table_styles([
    {"selector": "th", "props": [("color", 'black'), ("background-color", "#FF00CC")]}
])
styled_df


# In[4]:


df.info()


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>4.2 | Drop the Columns</b></span>

# In[5]:


# Conserver uniquement les colonnes attendues pour le dataset multiclasse.
# Utiliser une copie pour éviter les warnings de chained-assignment.
df = df[['subject', 'body', 'label']].copy()


# In[6]:


styled_df = df.head(5).style


# Modify the color and background color of the table headers (th)
styled_df.set_table_styles([
    {"selector": "th", "props": [("color", 'Black'), ("background-color", "#FF00CC"), ('font-weight', 'bold')]}
])


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>4.3 | Rename the Column</b></span>

# In[7]:


# Construire une colonne texte unique à partir de subject + body pour le NLP.
# Garder le nom `text` pour rester compatible avec les cellules suivantes.
df['text'] = (df['subject'].fillna('') + ' ' + df['body'].fillna('')).str.strip()


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>4.4 | Convert the target variable</b></span>

# In[8]:


from sklearn.preprocessing import LabelEncoder

# Encoder les labels multiclasses (ecole/pro/sport) en cibles numériques.
# Conserver un mapping lisible pour interpréter les métriques et la matrice.
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['label'])
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print('Label mapping:', label_mapping)


# In[9]:


styled_df = df.head().style


# Modify the color and background color of the table headers (th)
styled_df.set_table_styles([
    {"selector": "th", "props": [("color", 'Black'), ("background-color", "#FF00CC"), ('font-weight', 'bold')]}
])


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>4.5 | Check Missing values</b></span>

# In[10]:


#checking missing values
df.isnull().sum()


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>4.6 | Check Duplicate values</b></span>

# In[11]:


#check duplicate values
df.duplicated().sum()


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>4.7 | Remove Duplicate values</b></span>

# In[12]:


#remove Duplicate
df = df.drop_duplicates(keep = 'first')


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>4.8 | Shape of the Dataset</b></span>

# In[13]:


df.shape


# <a id="5"></a>
# <div style="text-align: center; background:  #FF00CC; font-family: 'Trebuchet MS', Arial, sans-serif; color: white; padding: 15px; font-size: 26px; font-weight: bold; line-height: 1; border-radius: 50% 0 50% 0 / 40px; margin-bottom: 20px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">5. EDA</div>
# 

# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>5.1 | Percentage of Ham and Spam</b></span>

# In[14]:


# Calculer la distribution multiclasses à partir des labels texte d'origine.
# Le tri garde un ordre de classes stable entre les exécutions.
values = df['label'].value_counts().sort_index()

import matplotlib.pyplot as plt

# Définir une couleur par label (3 classes dans ce dataset).
# Un léger explode sur chaque part améliore la lisibilité.
colors = ['#FF5733', '#33FF57', '#3357FF']
explode = [0.02] * len(values)

# Créer la figure avec un fond blanc
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('white')

# Créer le camembert multiclasse avec des labels dynamiques.
# `values.index.tolist()` garantit l'alignement labels/valeurs.
wedges, texts, autotexts = ax.pie(
    values.values,
    labels=values.index.tolist(),
    autopct='%0.2f%%',
    startangle=90,
    colors=colors[:len(values)],
    wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
    explode=explode,
    shadow=True
)

# Personnaliser les propriétés du texte
for text, autotext in zip(texts, autotexts):
    text.set(size=14, weight='bold')
    autotext.set(size=14, weight='bold')

# Ajouter un titre adapté aux labels multiclasses
ax.set_title('Email Classification by Label', fontsize=16, fontweight='bold')

# Garder un ratio égal pour afficher un cercle
ax.axis('equal')

# Afficher le graphique
plt.show()


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>5.2 | Text Length and Structure Analysis</b></span>

# In[15]:


df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentence'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))


# In[16]:


df[['num_characters', 'num_words', 'num_sentence']].describe()


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>5.3 | Summary Statistics for Legitimate Messages</b></span>

# In[17]:


# Afficher les statistiques descriptives de chaque classe dans un seul tableau.
# Cela remplace les résumés binaires ham/spam.
df.groupby('label')[['num_characters', 'num_words', 'num_sentence']].describe()


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>5.4 | Summary Statistics for Spam Messages</b></span>

# In[18]:


# Comparer les moyennes des features de structure de texte sur toutes les classes.
# Les valeurs arrondies facilitent une lecture rapide des écarts.
df.groupby('label')[['num_characters', 'num_words', 'num_sentence']].mean().round(2)


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>5.5 | Character Length Distribution for Legitimate and Spam Messages</b></span>

# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt

# Tracer la distribution des longueurs de texte pour tous les labels sur un même graphe.
# `common_norm=False` garde des distributions comparables entre classes.
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x='num_characters',
    hue='label',
    kde=True,
    element='step',
    stat='density',
    common_norm=False
)

# Ajouter des labels et un titre adaptés au multiclasse
plt.xlabel('Number of Characters', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Distribution of Number of Characters by Label', fontsize=16, fontweight='bold')

# Personnaliser l'apparence du graphique
sns.set(style='whitegrid')

# Afficher le graphique
plt.show()


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>5.6 | Word Count Distribution for Legitimate and Spam Messages</b></span>

# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt

# Tracer la distribution du nombre de mots pour tous les labels sur un même graphe.
# Cela évite les comparaisons uniquement binaires et garde l'intention EDA.
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x='num_words',
    hue='label',
    kde=True,
    element='step',
    stat='density',
    common_norm=False
)

# Ajouter des labels et un titre adaptés au multiclasse
plt.xlabel('Number of Words', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Distribution of Number of Words by Label', fontsize=16, fontweight='bold')

# Personnaliser l'apparence du graphique
sns.set(style='whitegrid')

# Afficher le graphique
plt.show()


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>5.7 | Pairplot for Data Visualization</b></span>

# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt

# Créer un pairplot multiclasse en utilisant les labels texte.
# Définir 3 marqueurs pour couvrir les 3 classes du dataset.
sns.set(style='ticks', color_codes=True)
g = sns.pairplot(df, hue='label', diag_kind='kde', markers=["o", "s", "D"])

# Définir un titre et ajuster son positionnement.
# Conserver une légende automatique pour éviter les mappings binaires en dur.
g.fig.suptitle("Pairplot des variables par label", fontsize=16, fontweight='bold')
plt.subplots_adjust(top=0.95)
g._legend.set_title('Label')

# Afficher le pairplot
plt.show()


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>5.8 | Coorelation</b></span>

# In[22]:


df[['target','num_characters', 'num_words', 'num_sentence']].corr()


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt

# Sélectionner les colonnes pour la matrice de corrélation
correlation_matrix = df[['target', 'num_characters', 'num_words', 'num_sentence']].corr()

# Créer une heatmap avec une mise en forme personnalisée
plt.figure(figsize=(10, 6))
sns.set(font_scale=1.2)  # Ajuster l'échelle de police pour une meilleure lisibilité
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")

# Définir un titre pour la heatmap
plt.title("Correlation Heatmap", fontsize=16, fontweight='bold')

# Tourner les labels de l'axe x pour une meilleure lisibilité
plt.xticks(rotation=45)

# Afficher la heatmap
plt.show()


# <a id="6"></a>
# <div style="text-align: center; background:  #FF00CC; font-family: 'Trebuchet MS', Arial, sans-serif; color: white; padding: 15px; font-size: 26px; font-weight: bold; line-height: 1; border-radius: 50% 0 50% 0 / 40px; margin-bottom: 20px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">6. Data Preprocessing</div>
# 

# In[24]:


# Importing the Porter Stemmer for text stemming
from nltk.stem.porter import PorterStemmer

# Importing the string module for handling special characters
import string

# Creating an instance of the Porter Stemmer
ps = PorterStemmer()

# Lowercase transformation and text preprocessing function
def transform_text(text):
    # Transform the text to lowercase
    text = text.lower()

    # Tokenization using NLTK
    text = nltk.word_tokenize(text)

    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Removing stop words and punctuation
    text = y[:]
    y.clear()

    # Loop through the tokens and remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming using Porter Stemmer
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    # Join the processed tokens back into a single string
    return " ".join(y)


# In[25]:


transform_text('Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...')


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>6.1 | Creating a New Column: 'transformed_text'</b></span>

# In[26]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[27]:


styled_df = df.head(5).style


# Modify the color and background color of the table headers (th)
styled_df.set_table_styles([
    {"selector": "th", "props": [("color", 'Black'), ("background-color", "#FF00CC"), ('font-weight', 'bold')]}
])


# In[28]:


wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

# Construire un wordcloud par label pour respecter l'objectif multiclasse.
# Un ordre trié fixe garde la position des panneaux stable entre exécutions.
labels_sorted = sorted(df['label'].unique())
fig, axes = plt.subplots(1, len(labels_sorted), figsize=(18, 6))
axes = np.atleast_1d(axes)

for ax, current_label in zip(axes, labels_sorted):
    # Concaténer uniquement le texte transformé de la classe courante.
    # Générer puis afficher l'importance du vocabulaire par classe.
    text_blob = df[df['label'] == current_label]['transformed_text'].str.cat(sep=' ')
    wc_img = wc.generate(text_blob)
    ax.imshow(wc_img)
    ax.set_title(f'WordCloud: {current_label}', fontsize=12, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.show()


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>6.2 | Word Cloud for Not spam Messages</b></span>

# In[29]:


# Construire un wordcloud global pour visualiser le vocabulaire commun.
# Cela complète les wordclouds par classe avec une vue d'ensemble.
all_wc = wc.generate(df['transformed_text'].str.cat(sep=' '))
plt.figure(figsize=(15, 6))
plt.imshow(all_wc)
plt.title('WordCloud: All Labels Combined', fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>6.3 | Find top 30 words of spam</b></span>

# In[30]:


# Utiliser un label de référence pour l'analyse de fréquence en multiclasse.
# Garder le nom de variable existant évite de casser les cellules suivantes.
reference_label = sorted(df['label'].unique())[0]
spam_carpos = []
for sentence in df[df['label'] == reference_label]['transformed_text'].tolist():
    for word in sentence.split():
        spam_carpos.append(word)

print(f'Reference label used for spam_carpos: {reference_label}')


# In[31]:


from collections import Counter
filter_df = pd.DataFrame(Counter(spam_carpos).most_common(30))


# In[32]:


sns.barplot(
    data = filter_df,
    x = 0,
    y = 1,
    hue = 0,
    palette = 'bright',
    legend = False)
plt.xticks(rotation = 90)
plt.show()


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>6.4 | Find top 30 words of Not spam Messages</b></span>

# In[33]:


# Utiliser un deuxième label pour comparer les fréquences en multiclasse.
# Garder le nom de variable existant évite de casser les cellules suivantes.
comparison_label = sorted(df['label'].unique())[1]
ham_carpos = []
for sentence in df[df['label'] == comparison_label]['transformed_text'].tolist():
    for word in sentence.split():
        ham_carpos.append(word)

print(f'Comparison label used for ham_carpos: {comparison_label}')


# In[34]:


filter_ham_df = pd.DataFrame(Counter(ham_carpos).most_common(30))


# In[35]:


sns.barplot(
    data = filter_ham_df, 
    x = 0,               # On utilise l'index de la colonne (mots)
    y = 1,               # On utilise l'index de la colonne (fréquences)
    hue = 0,             # On colore selon les mots pour activer la palette
    palette = 'cool', 
    legend = False       # On cache la légende car elle est inutile ici
)
plt.xticks(rotation = 90)
plt.show()


# <a id="7"></a>
# <div style="text-align: center; background: linear-gradient(to right, #FF00CC, #660066); font-family: 'Trebuchet MS', Arial, sans-serif; color: white; padding: 15px; font-size: 26px; font-weight: bold; line-height: 1; border-radius: 50% 0 50% 0 / 40px; margin-bottom: 20px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">7. Model Building</div>
# 

# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>7.1 | Initializing CountVectorizer and TfidfVectorizer</b></span>

# In[36]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Garder CountVectorizer pour des tests rapides et l'inspection des features.
# Utiliser TF-IDF comme représentation principale pour l'entraînement multiclasse.
cv = CountVectorizer()
tfid = TfidfVectorizer(max_features=3000)


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>7.2 | Dependent and Independent Variable</b></span>

# In[37]:


# Vectoriser le texte nettoyé avec TF-IDF pour créer les features d'entraînement.
# Utiliser la cible multiclasse encodée (0..n_classes-1) comme labels.
X = tfid.fit_transform(df['transformed_text']).toarray()
y = df['target'].values


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>7.3 | Split into Train and Test Data</b></span>

# In[38]:


from sklearn.model_selection import train_test_split

# Stratifie pour conserver l'équilibre des classes entre train et test.
# Garder random_state fixe pour des comparaisons reproductibles.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=2, stratify=y
)


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>7.4 | Import the Models</b></span>

# In[39]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>7.5 | Initialize the Models</b></span>

# In[40]:


svc = SVC(kernel="sigmoid", gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)

# Utiliser un solveur compatible multiclasse (liblinear ne gère pas 3 classes).
# Garder seulement les paramètres compatibles avec cette version de scikit-learn.
lrc = LogisticRegression(solver='lbfgs', max_iter=2000)

rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)


# In[41]:


clfs = {
    'SVC': svc,
    'KNN': knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'Adaboost': abc,
    'Bgc': bc,
    'ETC': etc,
    'GBDT': gbdt,
    'xgb': xgb

}


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>7.7 |Train the Models</b></span>

# In[42]:


from sklearn.metrics import accuracy_score, precision_score

def train_classifier(clfs, X_train, y_train, X_test, y_test):
    # Entraîner un classifieur et calculer des métriques multiclasses standard.
    # La macro-precision donne le même poids à chaque classe.
    clfs.fit(X_train, y_train)
    y_pred = clfs.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    return accuracy, precision


# In[43]:


def update_performance(method_name, y_true, y_pred):
    # Calculer les métriques dans la fonction pour garder un pipeline réutilisable.
    # La macro-precision est robuste pour comparer en multiclasse.
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)

    # Créer une ligne puis l'ajouter au tableau de bord global.
    # Garder les mêmes noms de colonnes évite de casser les graphiques ensuite.
    new_row = pd.DataFrame({
        'Méthode': [method_name],
        'Accuracy': [acc],
        'Precision': [prec]
    })

    global performance_df
    performance_df = pd.concat([performance_df, new_row], ignore_index=True)
    print(f"✅ {method_name} ajouté au tableau avec Accuracy: {acc:.4f} et Macro-Precision: {prec:.4f}")


# In[44]:


# Initialisation du tableau de bord avec les types de données définis
performance_df = pd.DataFrame({
    'Méthode': pd.Series(dtype='str'),
    'Accuracy': pd.Series(dtype='float'),
    'Precision': pd.Series(dtype='float')
})

for name, clf in clfs.items():
    # 1. Entraînement et calcul des prédictions
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 2. Sauvegarde automatique dans le tableau global via notre fonction
    update_performance(name, y_test, y_pred)

# Affichage du tableau final avec TOUS les modèles (R1, R2 + les nouveaux)
display(performance_df)


# <a id="8"></a>
# <div style="text-align: center; background: linear-gradient(to right, #FF00CC, #660066); font-family: 'Trebuchet MS', Arial, sans-serif; color: white; padding: 15px; font-size: 26px; font-weight: bold; line-height: 1; border-radius: 50% 0 50% 0 / 40px; margin-bottom: 20px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">8. Deep learning</div>
# 

# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>8.1 | préparation des données</b></span>

# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>8.1.1 | tokenization</b></span>

# In[45]:


from tensorflow.keras.preprocessing.text import Tokenizer

# 1. On crée l'objet
mon_tokenizer = Tokenizer(num_words=3000)

# 2. On lui apprend les mots de TON tableaux
mon_tokenizer.fit_on_texts(df['transformed_text'])

# 3. ON L'UTILISE pour transformer
sequences = mon_tokenizer.texts_to_sequences(df['transformed_text'])


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>8.2 | séparation stat données</b></span>

# In[46]:


from tensorflow.keras.preprocessing.sequence import pad_sequences

# On transforme tes listes de chiffres en un tableau fixe de 50 colonnes
X_deep = pad_sequences(sequences, maxlen=50, padding='post')
# On prépare tes 3 colonnes de stats (déjà prêtes dans ton df)
X_stats = df[['num_characters', 'num_words', 'num_sentence']].values


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>8.3 | construction du model</b></span>

# In[47]:


from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Dropout
from tensorflow.keras.models import Model

# Définir dynamiquement le nombre de classes depuis les labels encodés.
# Cela garde le modèle aligné avec le dataset actuel.
num_classes = len(encoder.classes_)

# --- BRANCHE 1 : LE TEXTE (X_deep) ---
input_text = Input(shape=(50,), name="Entree_Texte")
x = Embedding(input_dim=3000, output_dim=64)(input_text)
x = LSTM(32)(x)
x = Dropout(0.2)(x)

# --- BRANCHE 2 : LES STATS (Longueur, Mots, Phrases) ---
input_stats = Input(shape=(3,), name="Entree_Stats")
y = Dense(8, activation='relu')(input_stats)

# --- FUSION DES DEUX BRANCHES ---
combined = concatenate([x, y])

# --- SORTIE MULTICLASSE ---
z = Dense(16, activation='relu')(combined)
output = Dense(num_classes, activation='softmax')(z)

# Construire et compiler avec une loss sparse multiclasse.
# `sparse_categorical_crossentropy` attend des cibles entières encodées.
model_complet = Model(inputs=[input_text, input_stats], outputs=output)
model_complet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>8.3 | split des nouvelles données</b></span>

# In[48]:


from sklearn.model_selection import train_test_split

# On splitte tout en même temps : X_deep, X_stats et la cible y
X_train_deep, X_test_deep, X_train_stats, X_test_stats, y_train, y_test = train_test_split(
    X_deep, X_stats, df['target'], test_size=0.2, random_state=2
)


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>8.3 | apprentissage</b></span>

# In[49]:


# On lance l'apprentissage
history = model_complet.fit(
    x=[X_train_deep, X_train_stats], # Les deux entrées d'entraînement
    y=y_train,                        # La cible (0 ou 1)
    epochs=10,                        # On passe 10 fois sur les données
    batch_size=32,                    # On traite les SMS par paquets de 32
    validation_data=([X_test_deep, X_test_stats], y_test) # Pour tester en temps réel
)


# In[50]:


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Succès Entraînement')
plt.plot(history.history['val_accuracy'], label='Succès Test')
plt.title('Performance du modèle fusionné')
plt.ylabel('Précision')
plt.xlabel('Époque')
plt.legend()
plt.show()


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>8.4 | validation sur les donnée test</b></span>

# In[51]:


# Prédire les probabilités de classe puis convertir en ids de classe.
# `argmax` est nécessaire pour des sorties softmax multiclasses.
y_pred_proba_1 = model_complet.predict([X_test_deep, X_test_stats])
y_pred_1 = np.argmax(y_pred_proba_1, axis=1)

# Sauvegarder les métriques multiclasses dans le tableau partagé.
# Réutiliser la même fonction aide à garder des comparaisons cohérentes.
update_performance('LSTM Hybrid (R1)', y_test, y_pred_1)

# display(performance_df)


# In[52]:


from sklearn.metrics import confusion_matrix, classification_report
# 3. On crée la matrice
cm = confusion_matrix(y_test, y_pred_1)

# 4. On l'affiche joliment
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédiction')
plt.ylabel('Réalité')
plt.title('Matrice de Confusion : Modèle Fusionné')
plt.show()


# <a id="9"></a>
# <div style="text-align: center; background:  #FF00CC; font-family: 'Trebuchet MS', Arial, sans-serif; color: white; padding: 15px; font-size: 26px; font-weight: bold; line-height: 1; border-radius: 50% 0 50% 0 / 40px; margin-bottom: 20px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">9. finetuning</div>
# 

# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>9.1 | choix du model pré-entrainer</b></span>

# ## 📊 Comparaison des modèles BERT
# 
# | Modèle | Paramètres | Taille | Vitesse | Performance | Cas d'usage |
# |--------|-----------|--------|---------|------------|-----------|
# | **BERT** (original) | 110M | 440 MB | 🐢 Lent | ⭐⭐⭐⭐⭐ | Production |
# | **DistilBERT** (actuellement) | 66M | 268 MB | 🚗 Moyen | ⭐⭐⭐⭐⭐ | **Bon compromis** ✅ |
# | **TinyBERT** | 14M | 56 MB | ⚡⚡ Extrêmement rapide | ⭐⭐⭐ | Très léger |
# | **ALBERT-small** | 12M | 48 MB | ⚡⚡ Extrêmement rapide | ⭐⭐⭐ | Très compact |

# In[53]:


from transformers import AutoTokenizer, AutoModel
import torch

def load_model(model_choice):
    models_dict = {
        'bert': 'bert-base-uncased',
        'distilbert': 'distilbert-base-uncased',
        'albert': 'albert-base-v2',
        'tinybert': 'huawei-noah/TinyBERT_General_6L_768D'
    }

    model_name = models_dict.get(model_choice)
    if not model_name:
        print(f"❌ Modèle '{model_choice}' non trouvé!")
        return None, None

    print(f"🔄 Chargement de {model_choice} via PyTorch...")

    # Chargement du tokenizer et du modèle (AutoModel charge la version PyTorch par défaut)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Geler les paramètres (Transfer Learning) c'est ce qui permet le finetuning
    for param in model.parameters():
        param.requires_grad = False

    # On passe le modèle en mode évaluation (important pour BERT/Dropout)
    model.eval()

    return tokenizer, model

# Test
model_choice = 'distilbert'  # ou 'distilbert', 'bert', 'tinybert', 'albert'
tokenizer, bert_model = load_model(model_choice)
print(f"✅ {model_choice} chargé avec succès !")


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>9.2 | Tokenization propre au model</b></span>

# In[54]:


print("🔄 Tokenization des SMS...")

X_bert = [tokenizer.encode(text, max_length=50, padding='max_length', truncation=True) 
          for text in df['text']]
X_bert = np.array(X_bert)

print(f"✅ X_bert shape: {X_bert.shape}")

# Préparer les stats
X_stats = df[['num_characters', 'num_words', 'num_sentence']].values
y = df['target'].values

print(f"✅ X_stats shape: {X_stats.shape}")
print(f"✅ y shape: {y.shape}")


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>9.3 | split des données</b></span>

# In[55]:


from sklearn.model_selection import train_test_split

train_ids, test_ids, train_stats, test_stats, y_train, y_test = train_test_split(
    X_bert, X_stats, y, test_size=0.2, random_state=2
)

print(f"✅ Train: {train_ids.shape} | Test: {test_ids.shape}")


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>9.4 | précalcul pour accélérer</b></span>

# In[56]:


import torch

print("⏳ Pré-calcul des features (~10 min)...\n")

# Ensemble d'entraînement
train_ids_tensor = torch.tensor(train_ids)
with torch.no_grad():
    train_output = bert_model(train_ids_tensor)
train_cls = train_output[0][:, 0, :].numpy()

# Ensemble de test
test_ids_tensor = torch.tensor(test_ids)
with torch.no_grad():
    test_output = bert_model(test_ids_tensor)
test_cls = test_output[0][:, 0, :].numpy()

print(f"✅ train_cls: {train_cls.shape}")
print(f"✅ test_cls: {test_cls.shape}")


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>9.5 | construction du model</b></span>

# In[57]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout
from tensorflow.keras.models import Model

# Garder un nombre de classes dynamique selon l'encodage actuel.
# Cela évite de figer une sortie binaire dans une tâche multiclasse.
num_classes = len(encoder.classes_)

# Entrée 1 : Features BERT
input_bert = Input(shape=(768,), name="bert_features")

# Entrée 2 : Stats
input_stats_layer = Input(shape=(3,), name="input_stats")
stats_feat = Dense(8, activation='relu')(input_stats_layer)

# Fusion
combined = concatenate([input_bert, stats_feat])

# Couches finales
x = Dense(32, activation='relu')(combined)
x = Dropout(0.2)(x)
output = Dense(num_classes, activation='softmax')(x)

# Modèle multiclasse avec loss sparse catégorielle.
# Les labels entiers du LabelEncoder sont directement compatibles.
model = Model(inputs=[input_bert, input_stats_layer], outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✅ Modèle créé !")
print(f"   Total params: {model.count_params():,}")


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>9.6 | entrainement des couche spécifique</b></span>

# In[58]:


print("🚀 Entraînement...\n")

history = model.fit(
    x=[train_cls, train_stats],
    y=y_train,
    epochs=3,
    batch_size=32,
    validation_data=([test_cls, test_stats], y_test),
    verbose=1
)

print("\n✅ Entraînement terminé !")


# > <span style='font-size:15px; font-family:Verdana;color: #FF00CC;'><b>9.7 | classification des données test</b></span>

# In[59]:


from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report

# Prédire les probabilités puis convertir en ids de classe.
# Argmax est requis pour des sorties softmax multiclasses.
y_pred_proba = model.predict([test_cls, test_stats])
y_pred = np.argmax(y_pred_proba, axis=1)

# Utiliser la macro-precision pour pondérer chaque classe de façon égale.
# zero_division=0 évite les erreurs si une classe n'est jamais prédite.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Macro-Precision: {precision:.4f}")

# Matrice de confusion multiclasse
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Confusion Matrix:")
print(cm)

# Rapport multiclasse avec noms de classes dynamiques
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=list(encoder.classes_), zero_division=0))


# In[60]:


update_performance('BERT Hybrid (R2)', y_test, y_pred)


# ## 📊 Comparaison des modèles (version multiclasse)
# 
# Les scores affichés dans `performance_df` utilisent maintenant :
# - `Accuracy`
# - `Precision` en **macro** (adaptée aux 3 labels)
# 
# > Remarque : les anciens chiffres binaires (HAM/SPAM) ne sont plus la référence pour ce notebook.

# <a id="10"></a>
# <div style="text-align: center; background:  #FF00CC; font-family: 'Trebuchet MS', Arial, sans-serif; color: white; padding: 15px; font-size: 26px; font-weight: bold; line-height: 1; border-radius: 50% 0 50% 0 / 40px; margin-bottom: 20px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">10. Evaluate the Models</div>
# 

# In[61]:


import matplotlib.pyplot as plt
import seaborn as sns

# 1. On transforme le tableau pour qu'il soit lisible par Seaborn (format "long")
df_melted = performance_df.melt(id_vars='Méthode', var_name='Métrique', value_name='Score')

# 2. Création du graphique
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

ax = sns.barplot(data=df_melted, x='Méthode', y='Score', hue='Métrique', palette='magma')

# 3. Personnalisation
plt.title('Comparaison des Performances : Deep Learning vs Classique', fontsize=16)
plt.ylim(0.8, 1.0)  # On zoom entre 80% et 100% pour mieux voir les différences
plt.ylabel('Score (0-1)')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Ajout des valeurs au-dessus des barres
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.3f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points',
                fontsize=10)

plt.tight_layout()
plt.show()


# <div style="background-color:white;font-size:15px;font-family:Georgia;border-style: solid;border-color: #FF00CC;border-width:3px;padding:10px;margin: 1px;color:#254E58;overflow:hidden"> 
# 
# <h4><b style = 'color: red;'>Amélioration possible :
# 
# 
# <b>👉 cross validation a l'aide de KFold (réalise plusieur réseau est fait une moyenne pondération des résultats) <br>
# <b>👉 cross validation pondération des différents résultats de différentes IA <br>
# <b>👉  mesure de temps et de ressource des IA <br>
# <b>👉 Augmenter les données réelles ou fictives (ex: EDA : synonymes, suppression aléatoire de mots non essentiels).<br>
# <b>👉 Tester des modèles FR (CamemBERT, FlauBERT) ou multilingues selon la langue des mails.<br>
# <b>👉 En multiclasses, ajuster plutôt l'entraînement (features, architecture, régularisation) car la décision finale se fait via `argmax` et non un seuil binaire à 0.5.<br>

# <div style="background-color:white;font-size:15px;font-family:Georgia;border-style: solid;border-color: #FF00CC;border-width:3px;padding:10px;margin: 1px;color:#254E58;overflow:hidden"> 
# 
# <h4><b style = 'color: red;'>source :
# 
# 
# <b>👉https://www.kaggle.com/code/zabihullah18/email-spam-detection <br>
# <b>👉https://www.kaggle.com/code/dkhalidashik/spam-text-classification-using-deep-learning<br>
#     
