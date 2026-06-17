# 🌍 Rechauffement-planete

[License: MIT](https://opensource.org/licenses/MIT)
[Python 3.9+](https://www.python.org/downloads/)
[Streamlit](https://streamlit.io/)

**Analyse des données climatiques mondiales : évolutions d'un phénomène planétaire**

> Projet de Data Analyse - Formation DA Janvier 2024

---

## 📌 À propos du projet

Ce projet analyse les **variations de température mondiale** à partir des données de la NASA pour comprendre et prédire l'évolution du réchauffement climatique. L'application permet de visualiser, analyser et modéliser les tendances climatiques à travers différentes régions du monde.

### 🎯 Objectifs principaux

- **Constater** les variations de température observées à travers le monde, significativement plus élevées qu'à l'époque préindustrielle (1850-1900)
- **Identifier** les paramètres naturels et liés à l'activité humaine expliquant ces variations
- **Analyser** les forces explicatives de ces paramètres dans un modèle descriptif
- **Prédire** les variations futures en manipulant ces paramètres

### 🌐 Contexte scientifique

Le réchauffement climatique, observé depuis le milieu du 19ème siècle, a des conséquences déjà visibles :

- Événements météorologiques extrêmes plus fréquents et intenses (sécheresses, inondations, tempêtes)
- Fonte des glaciers et élévation du niveau de la mer
- Perturbations des écosystèmes terrestres et marins
- Modifications des cycles biologiques et des chaînes alimentaires

---

## ✨ Fonctionnalités


| Section                     | Description                                                                         |
| --------------------------- | ----------------------------------------------------------------------------------- |
| **Introduction**            | Contexte général, technique, économique et scientifique du réchauffement climatique |
| **Collecte et exploration** | Chargement et première analyse des datasets climatiques                             |
| **Preprocessing**           | Nettoyage, transformation et préparation des données                                |
| **Visualisation**           | Graphiques interactifs des tendances climatiques par continent et pays              |
| **Modèles supervisés**      | Analyse des corrélations entre paramètres climatiques                               |
| **Séries temporelles**      | Modélisation SARIMAX et Holt-Winters pour prédire les températures futures          |


---

## 🛠 Technologies utilisées


| Technologie                                     | Version | Usage                                            |
| ----------------------------------------------- | ------- | ------------------------------------------------ |
| [**Python**](https://www.python.org/)           | 3.9+    | Langage principal                                |
| [**Streamlit**](https://streamlit.io/)          | 1.28+   | Framework web pour l'interface utilisateur       |
| [**Pandas**](https://pandas.pydata.org/)        | Latest  | Manipulation et analyse des données              |
| [**NumPy**](https://numpy.org/)                 | Latest  | Calculs numériques                               |
| [**Matplotlib**](https://matplotlib.org/)       | Latest  | Visualisation statique                           |
| [**Plotly**](https://plotly.com/python/)        | Latest  | Visualisation interactive                        |
| [**Statsmodels**](https://www.statsmodels.org/) | Latest  | Modélisation statistique (SARIMAX, Holt-Winters) |


---

## 📦 Structure du projet

```
Rechauffement-planete/
├── presentation.py          # Application Streamlit principale
├── requirements.txt         # Dépendances Python
├── README.md                # Documentation du projet
├── LICENSE                  # Licence MIT
├── .devcontainer/           # Configuration Dev Container (VS Code)
└── ressources/              # Données et visualisations
    ├── dataset.csv          # Dataset principal des températures
    ├── MONDE.csv            # Données mondiales agrégées
    ├── MONDE2011.csv        # Données mondiales (2011+)
    ├── MONDE_12_22.csv      # Données mondiales (2012-2022)
    ├── GISTEMP.csv          # Données NASA GISS
    ├── GHCN.csv             # Données Global Historical Climatology Network
    ├── *.png                # Visualisations générées
    └── ...
```

---

## 📊 Données

### Sources principales


| Dataset         | Source    | Description                                        |
| --------------- | --------- | -------------------------------------------------- |
| **dataset.csv** | NASA GISS | Températures mondiales par pays et continent       |
| **MONDE.csv**   | NASA GISS | Températures mondiales moyennes (série temporelle) |
| **GISTEMP.csv** | NASA GISS | Anomalies de température globale                   |
| **GHCN.csv**    | NOAA      | Données historiques de stations météorologiques    |


### Variables clés

- **Températures** : Anomalies par rapport à la période de référence (1951-1980)
- **Période** : Données historiques depuis 1850
- **Granularité** : Mensuelle, annuelle, par pays/continent
- **Paramètres** : CO₂ atmosphérique, activité solaire, aérosols, etc.

---

## 🚀 Installation et exécution

### Prérequis

- Python 3.9 ou supérieur
- pip (gestionnaire de paquets Python)
- Git (optionnel, pour cloner le dépôt)

### 1. Cloner le dépôt

```bash
git clone https://github.com/DamienS3/Rechauffement-planete.git
cd Rechauffement-planete
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Avec venv (intégré à Python)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows

# Ou avec conda
conda create -n rechauffement python=3.9
conda activate rechauffement
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer l'application

```bash
streamlit run presentation.py
```

L'application sera disponible à l'adresse : [**http://localhost:8501**](http://localhost:8501)

---

## 🔬 Méthodologie

### 1. Collecte et exploration

- Chargement des datasets CSV depuis le dossier `ressources/`
- Identification des variables : pays, continents, dates, températures
- Statistiques descriptives initiales

### 2. Preprocessing

- Gestion des valeurs manquantes (imputation)
- Normalisation des données
- Transformation des séries temporelles
- Sélection des features pertinentes

### 3. Visualisation

- **Cartes thermiques** : Évolution des températures par région
- **Séries temporelles** : Tendances annuelles et mensuelles
- **Distributions** : Analyse des anomalies de température
- **Corrélations** : Relations entre paramètres climatiques

### 4. Modélisation

#### Modèles de séries temporelles


| Modèle           | Description                             | Usage                                    |
| ---------------- | --------------------------------------- | ---------------------------------------- |
| **SARIMAX**      | Modèle autorégressif saisonnier intégré | Prédiction des températures futures      |
| **Holt-Winters** | Lissage exponentiel triple              | Détection des tendances et saisonnalités |


#### Évaluation

- **Métriques** : RMSE (Root Mean Square Error), MAE (Mean Absolute Error)
- **Validation** : Split temporel (train/test)
- **Visualisation** : Comparaison prédictions vs valeurs réelles

---

## 📈 Résultats

### Principales découvertes

1. \\\*\\\*Augmentation globale\\\*\\\* : Les températures mondiales ont augmenté de \\\~1.2°C depuis l'ère préindustrielle
2. \\\*\\\*Accélération récente\\\*\\\* : Le rythme du réchauffement s'est accéléré depuis les années 1980
3. \\\*\\\*Variabilité régionale\\\*\\\* : Certaines régions (Arctique) se réchauffent plus vite que la moyenne
4. \\\*\\\*Corrélation CO₂\\\*\\\* : Forte corrélation entre concentration de CO₂ et augmentation des températures
5. \\\*\\\*Prédictions\\\*\\\* : Les modèles prévoient une poursuite de la hausse des températures

### Visualisations disponibles

- Cartes des anomalies de température (NASA GISS)
- Graphiques d'évolution par continent
- Décomposition des séries temporelles (tendance, saisonnalité, résidu)
- Matrices de corrélation entre paramètres
- Prédictions à 10 ans avec intervalles de confiance

---

## 📂 Fichiers importants


| Fichier                  | Description                               |
| ------------------------ | ----------------------------------------- |
| `presentation.py`        | Code principal de l'application Streamlit |
| `requirements.txt`       | Liste des dépendances Python              |
| `ressources/dataset.csv` | Dataset principal des températures        |
| `ressources/MONDE.csv`   | Série temporelle mondiale                 |


---

## 🤝 Contribution

Les contributions sont les bienvenues ! Voici comment contribuer :

1. Forker le dépôt
2. Créer une branche pour votre fonctionnalité (\\\`git checkout -b feature/amazing-feature\\\`)
3. Commiter vos changements (\\\`git commit -m 'Add some amazing feature'\\\`)
4. Pousser vers la branche (\\\`git push origin feature/amazing-feature\\\`)
5. Ouvrir une Pull Request

### Idées de contributions

- Ajouter de nouveaux datasets climatiques
- Implémenter d'autres modèles de prédiction (LSTM, Prophet, etc.)
- Améliorer les visualisations interactives
- Ajouter des tests unitaires
- Optimiser les performances du code

---

## 👥 Auteurs


| Auteur                        | Rôle         | LinkedIn                                                 |
| ----------------------------- | ------------ | -------------------------------------------------------- |
| **Sébastien Lagarde-Corrado** | Data Analyst | [LinkedIn](https://www.linkedin.com/in/slagardecorrado/) |
| **Damien Selosse**            | Data Analyst | [LinkedIn](https://www.linkedin.com/in/damienselosse/)   |


---

## 🏆 Remerciements

- **Formation Data Scientest** - Pour le cadre pédagogique
- **NASA GISS** - Pour les données climatiques de qualité
- **NOAA** - Pour les données historiques
- **Communauté Open Source** - Pour les outils utilisés (Streamlit, Pandas, Statsmodels, etc.)

---

## 📜 License

Ce projet est sous license **MIT** - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 📞 Contact

Pour toute question ou suggestion, n'hésitez pas à nous contacter :

- **Sébastien Lagarde-Corrado** : [LinkedIn](https://www.linkedin.com/in/slagardecorrado/)
- **Damien Selosse** : [LinkedIn](https://www.linkedin.com/in/damienselosse/)

---

## 🔗 Liens utiles

- [Dataset NASA GISS](https://data.giss.nasa.gov/gistemp/)
- [NOAA Global Historical Climatology Network](https://www.ncei.noaa.gov/access/search/)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
