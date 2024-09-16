import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
# Pour éviter d'avoir les messages warning
import warnings
warnings.filterwarnings('ignore')

# Chargement des datasets et listes
dataset = pd.read_csv('./ressources/dataset.csv', sep=";")
monde = pd.read_csv("./ressources/MONDE.csv", sep=";").set_index('date')

continents_list = dataset['Continent_EN'].unique()
countries_list = dataset['Name_EN'].unique()

# Page config
st.set_page_config(page_title="Présentation SLC&DS", layout="wide")
background="https://as1.ftcdn.net/v2/jpg/00/34/75/54/1000_F_34755489_duiIuPfqZNtYgrSGFZAyjg5KyMV2Euai.jpg"

#Sections
sections=["Introduction",
        "Collecte et exploration des données",
        "Preprocessing et analyse des données",
        "Visualisation des données",
        "Modèles supervisés",
        "Modèles séries temporelles et prédictions",
        "Crédits"]


#============
# SIDEBAR
#============

with st.sidebar:
    # Zone d'en-tête
    with st.container(border=True):
        st.image("https://datascientest.com/wp-content/uploads/2022/03/logo-2021.png")
        st.markdown("""
            
            <div data-testid="stCaptionContainer" class="st-emotion-cache-1g1z3k2 e1nzilvr5">
            <div style="text-align: center;">
            <p>Formation DA - Janvier 2024</p>
            </div>
            <p><span style='font-size:larger;'>Un projet mené par :</span></p>
            </div>
            """,unsafe_allow_html=True)
        
        IN="https://media.licdn.com/dms/image/v2/D4E0BAQFkiMXPKAXo0Q/company-logo_100_100/company-logo_100_100/0/1719404287274/linkedin_social_selling_logo?e=1733961600&v=beta&t=u98lpxNPMVejUKouuUlO1CPO1j892JKpjWYG7F4dVhw"
        autors1=st.columns((.5,.8,8))
        autors1[1].image(IN, width=20)
        autors1[2].markdown("[Sébastien Lagarde-Corrado](https://www.linkedin.com/in/slagardecorrado/)", unsafe_allow_html=True)
        autors2=st.columns((.5,.8,8))
        autors2[1].image(IN, width=20)
        autors2[2].markdown("[Damien Selosse](https://www.linkedin.com/in/damienselosse/)", unsafe_allow_html=True)
    
    # Séparation illustrée
    #with st.container():
        st.image("https://data.giss.nasa.gov/tmp/gistemp/NMAPS/tmp_GHCNv4_ERSSTv5_1200km_Anom_7_2024_2024_1951_1980_100_180_90_0_2_/amaps.png")
        col1,col2=st.columns((4.5,5.5))
        with col1:
            with st.popover("🔎 Zoom"):
                st.image("https://data.giss.nasa.gov/tmp/gistemp/NMAPS/tmp_GHCNv4_ERSSTv5_1200km_Anom_7_2024_2024_1951_1980_100_180_90_0_2_/amaps.png")
        with col2:
            with st.popover("🎥  Évolution"):
                st.video("https://data.giss.nasa.gov/gistemp/animations/TEMPANOMALY_05_2023_pdiff.mp4")
    
    #Sommaire
    with st.container(border=True):
        st.markdown("""
                    <span style='font-size=xx-large;'><b>Sommaire</b></span>
                    """, unsafe_allow_html=True)
        page=st.radio("** **", sections)

#============
# MAIN PAGE
#============

# Bandeau de titre
with st.container():
    st.markdown(f"""
                <style>
                .container_slc {{
                    width: 100%;
                    height:100px;
                    overflow: hidden;
                }}
                .image {{
                    width: calc(100% + 150px);
                    height: auto;
                    clip-path: inset(0 0 0 150px);
                    transform: translateX(-150px) translateY(-80%);
                }}
                </style>
                <div class="container_slc">
                <img src={background} alt="Description de l'image" class="image">
                </div>
                """, unsafe_allow_html=True)
    #st.image(background)
    st.title("Le réchauffement climatique : évolutions d'un phénomène planétaire")
    #st.caption("Projet de Data Analyse des températures terrestres")
    st.divider()

# INTRODUCTION
if page == sections[0] :
    with st.container():
        st.header(f"{sections[0]}")
        cols=st.columns((3,7))
        with cols[0]:
            st.title("")
            #st.write("")
            st.image("https://media.set.or.th/set/Images/2024/Jun/thailand-focus-2024-img-04.jpg")
        with cols[1]:
            intro_tabs = st.tabs(["Contexte général",
                            "Du point de vue technique",
                            "Du point de vue économique",
                            "Du point de vue scientifique"])
            with intro_tabs[0]:
                # Contexte général
                st.markdown("""**L'analyse des données climatiques est essentielle** pour comprendre les causes et les conséquences du réchauffement climatique,
                        ainsi que pour élaborer des stratégies d'atténuation et d'adaptation. Les scientifiques utilisent des modèles climatiques sophistiqués pour simuler 
                        les processus physiques et biologiques qui régissent le climat, et pour faire des prévisions sur les tendances futures.""")
                st.markdown("""Globalement observé depuis le milieu du 19ème siècle, les conséquences du réchauffement climatique **sont déjà visibles** dans de nombreuses régions du monde. 
                        Les événements météorologiques extrêmes, tels que les sécheresses, les inondations et les tempêtes, sont devenus **plus fréquents et plus intenses**. 
                        La fonte des glaciers et des calottes glaciaires contribue à l'élévation du niveau de la mer, menaçant les communautés côtières et les écosystèmes marins. 
                        Les changements de température et de précipitations affectent également les écosystèmes terrestres, en particulier les forêts et les zones humides,
                        et entraînent des **perturbations dans les cycles biologiques et les chaînes alimentaires**.""")
            with intro_tabs[1]:
                # Du point de vue technique
                st.write("""L'analyse des données climatiques est un domaine d'application clé de l'IA. En utilisant des techniques d'apprentissage automatique, 
                        les apprenants sur ce projet pourront identifier les sources pertinentes et les mettre en œuvre pour bâtir des prédictions de températures. 
                        Les modèles climatiques peuvent également être utilisés pour simuler les impacts du changement climatique sur les écosystèmes, les économies et les sociétés.""")
                st.write("""Cependant, l'analyse des données climatiques présente également des défis techniques importants. Les données sont souvent incomplètes, bruyantes et sujettes à des erreurs de mesure. 
                        De plus, les modèles climatiques sont complexes et nécessitent une grande puissance de calcul pour être simulés.""")
            with intro_tabs[2]:
                # Du point de vue économique
                st.write("""Le réchauffement climatique a des implications économiques considérables. Les coûts liés aux événements météorologiques extrêmes, tels que les ouragans, les inondations et les sécheresses, 
                        sont en augmentation constante. Selon le Groupe d'experts intergouvernemental sur l'évolution du climat (GIEC), les dommages causés par les catastrophes naturelles liées au climat ont augmenté 
                        de 50 % depuis les années 1970.""")
                st.write("""En outre, le réchauffement climatique menace les moyens de subsistance de millions de personnes dans le monde, en particulier dans les régions les plus vulnérables telles que les petits États 
                        insulaires en développement (PEID) et les pays les moins avancés (PMA). Les impacts économiques du changement climatique sont donc une préoccupation majeure pour les décideurs politiques, 
                        les entreprises, les communautés locales et plus généralement chacun d’entre nous.""")
            with intro_tabs[3]:
                # Du point de vue scientifique
                st.write("""Le réchauffement climatique est un phénomène complexe qui implique de multiples facteurs et processus interdépendants. Les scientifiques du Groupe d’experts intergouvernemental sur l’évolution 
                        du climat (Giec) ont établi que l'augmentation des concentrations de gaz à effet de serre (GES) dans l'atmosphère, en particulier le dioxyde de carbone (CO2), est la cause principale du réchauffement 
                        global observé depuis le milieu du XIXe siècle.""")
                st.write("""Les GES absorbent et émettent le rayonnement infrarouge émis par la Terre, créant ainsi un effet de serre naturel qui maintient la température de la planète à un niveau stable. 
                        Cependant, l'augmentation des émissions de GES due à l'activité humaine, en particulier la combustion de combustibles fossiles tels que le charbon, le pétrole et le gaz naturel, a entraîné une 
                        augmentation de l'effet de serre et une élévation de la température moyenne de la planète.""")
        #st.subheader("Objectifs:")
        st.write("")
        st.markdown("**Dans ce projet, nous avons pu :**\n"
                    "- **constater** à l’aide du jeu de données de la NASA que les variations de températures observées à travers le monde sont significativement plus élevées qu’à l’époque préindustrielle (1850-1900)\n"
                    "- **identifier** quelques paramètres naturels et liés à l’activité humaine permettent d’expliquer les variations observées\n"
                    "- **analyser** les forces explicatives de ces paramètres dans un modèle descriptif et manipuler ces paramètres pour prédire des variations futures.")
        
# DONNEES COLLECTE DESCRIPTION
if page == sections[1] :
    with st.container():
        st.header(f"{sections[1]}")
        st.subheader("Jeux de données")
        with st.expander("Collecte des jeux de données initiaux & complémentaires"):
            
            st.subheader("Jeux de données initiaux")
            
            st.markdown("""1️⃣ <b>Températures de surface (NASA - GISTEMP v4)</b>
                        - (Source: https://data.giss.nasa.gov/gistemp)
                        """, unsafe_allow_html=True)
            sub_cols_i=st.columns(3)
            with sub_cols_i[0]:
                with st.popover("Températures de surface (stations météo NOAA GHCN v4 et océans ERRST-v4)"):
                    st.write("Moyennes de températures (mensuelle, saisonnière ou annuelle) mondiales ou par hémisphère ou par zone, de 1880 à aujourd'hui.")
                    st.write("✅ précision chronologique importante (échelle mensuelle) et grande période couverte (1880-2024)")
                    st.write("❌ précision géographique faible (échelle zonale)")
                    st.image("./ressources/GHCN.png")
            with sub_cols_i[1]:
                with st.popover("Températures de surface (satellites Atmospheric Infra-Red Sounder - AIRS, v5 - v6 et v7)"):
                        st.write("Moyennes de températures (mensuelle, saisonnière ou annuelle) mondiales ou par hémisphère ou par zone, de 2002 à aujourd'hui.")
                        st.write("✅ précision chronologique importante (échelle mensuelle)")
                        st.write("❌ précision géographique faible (échelle zonale) et peu d’antériorité chronologique (2002-2024)")
                        st.image("./ressources/AirV6.png")
                        st.write("Ici on peut déjà observer des écarts jusqu’à + 0,3°C alors que la période de référence appartient à l’ère post-industrielle. \n"
                                "Les différentes versions, v5, v6 et v7 nous donnent des résultats différents mais la tendance reste la même : "
                                "augmentation des anomalies de températures avec une accélération depuis 2014 en dépit des épisodes Niña sur la période.")
            with sub_cols_i[2]:
                with st.popover("Anomalies climatiques mensuelles moyennes par coordonnées GPS de 1961 à 2010"):
                    st.write("✅ précision chronologique importante (échelle mensuelle) et géographique importante (échelle GPS)")
                    st.write("❌ période trop restreinte et manque d’actualité (1961-2010)")                    
                    st.image("./ressources/GISTEMPU.png")
                    
            st.markdown("""2️⃣ <b>Données mondiales CO2 et gaz à effet de serre (Our World in Data CO2 and Greenhouse Gas Emissions dataset)</b>
                        - (Source: https://github.com/owid/co2-dat)
                        """, unsafe_allow_html=True)
            with st.popover("plus..."):
                st.write("Grand dataset 48 058 lignes x 79 colonnes")
                st.write("✅ précision géographique importante (échelle pays +regroupements) : 192 pays avec code ISO")
                st.write("✅ précision chronologique importante (échelle annuelle) : de 1750 à 2022")
                    
            st.subheader("Jeux de données complémentaires")
            
            st.markdown("""3️⃣ <b>Températures de surface GISS (v4)</b>
                        - (Source: https://data.giss.nasa.gov/gistemp/station_data_v4_globe)
                        """, unsafe_allow_html=True)
            with st.popover("plus..."):
                st.write("Grand dataset 1 243 611 lignes x 21 colonnes (après retraitement partiel du dataset original de 1­ 271­ 506 lignes x 99 colonnes dans la version du 08/03/2024)")
                st.write("✅ précision géographique importante (échelle station) : liste de 22 141 Stations d'enregistrement météorologique à travers 216 pays")
                st.write("✅ précision chronologique importante (échelle mensuelle) : de 1851 à 2022")
                    
            st.markdown("""4️⃣ <b>Localisations géographiques</b>
                        """, unsafe_allow_html=True)
            sub_cols_c = st.columns(2)
            with sub_cols_c[0]:
                with st.popover("FIPS 10-4 Codes and history"):
                    st.markdown("""Association des codes FIPS et codes ISO 3166 - (Source: http://efele.net/maps/fips-10/data)""")
                    st.markdown("""<div style="text-align: justify;">
                                <b>FIPS 10-4</b> est une ancienne norme américaine, incluse dans le Federal Information Processing Standard (FIPS) intitulée Countries, Dependencies, Areas of Special Sovereignty, and Their Principal Administrative Divisions, ce qui se traduit par « Pays, dépendances, zones de souveraineté particulière, et leurs principales subdivisions administratives », caduque depuis le retrait de la norme américaine FIPS 10-4 le 2 septembre 2008.
                                </div>""", unsafe_allow_html=True)
                    st.markdown("""<div style="text-align: justify;">
                                <b>ISO 3166</b> (ICS n° 01.140.30) est une norme ISO de codage des pays et de leurs subdivisions. Cette norme définit des codes pour la quasi-totalité des pays du monde, y compris pour certains territoires (îles en général), non habités de façon permanente. Chacune de ces entités reçoit ainsi un code à deux lettres, un code à trois lettres et un code numérique."
                                </div>""", unsafe_allow_html=True)
                    st.write("") # cette ligne permet de forcer la largeur du pop up (auto-focus sur le contenu) pour éviter une contraction liée à l'alignement justifié
            with sub_cols_c[1]:
                with st.popover("Codes ISO 3166 et regroupements géographiques"):
                    st.markdown("""Association des codes ISO 3166 selon les zones géographiques, dont les continents - (Sources: https://www.donneesmondiales.com/codes-pays.php et https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv)""")
                    st.markdown("""Données générales incluses sur tous les pays :
                                <ul><li>Nom du pays (anglais)</li>
                                <li>Nom du pays (français)</li>
                                <li>Nom du pays en langue locale</li>
                                <li>Code de pays selon ISO 3166-1 (2 lettres)</li>
                                <li>Capitale</li>
                                <li>Continent</li>
                                <li>Population</li>
                                <li>Superficie (km²)</li>
                                <li>Longueur de la côte (km)</li>
                                <li>Forme de gouvernement</li>
                                <li>Monnaie</li>
                                <li>Abréviation de la monnaie</li>
                                <li>Indicatif du pays</li>
                                <li>Taux de natalité (pour 1000 habitants/an)</li>
                                <li>Taux de mortalité (pour 1000 habitants/an)</li></ul>
                                """, unsafe_allow_html=True)
            
            st.markdown("""5️⃣ <b>Évolution de la concentration de CO2 atmosphérique mondiale</b>
                        - (Source: https://gml.noaa.gov/ccgg/trends/data.htm)
                        """, unsafe_allow_html=True)
            with st.popover("plus..."):
                st.write("Petit dataset 66 lignes x 3 colonnes")
                st.write("✅ Précision géographique faible (échelle mondiale) et chronologique réduite (échelle annuelle) : de 1959 à 2023")
                st.image("./ressources/Co2Atm.png")


        st.subheader("Analyse des jeux de données")
        with st.expander("Analyses préliminaires - Données d'identification des pays"):
            st.subheader("Données d'identification des pays")
            st.markdown("""La mise en correspondance des codes FIPS avec les Codes ISO de pays, puis avec les valeurs de regroupements géographiques en continents a imposé quelques vérifications:
                    <ul><li>absences de correspondance entre code FIPS et code ISO</li>
                    <li>regroupement des codes FIPS rattachés à un même pays</li>
                    <li>association manuelle des codes ISO sans correspondance avec un continent</li>
                    <li>subdivision du continent américain en amérique du nord et amérique du sud</li></ul>
                    """, unsafe_allow_html=True)

        with st.expander("Analyses préliminaires - de températures de surface GISS (v4)"):
            st.subheader("Données de températures de surface GISS (v4)")
            
            cols_analyse_T1 = st.columns((9,1))
            with cols_analyse_T1[0]:
                st.markdown("""
                            <div style="text-align: justify;">
                            Le jeu de données ne présente qu'un sous-ensemble de la liste complète des stations: ne sont utilisées que des <b>stations avec des enregistrements de temps raisonnablement longs et mesurés de manière cohérente</b>. 
                            Ce sous-ensemble de la liste des stations qui contribuent aux produits finaux peut légèrement changer à chaque mise à jour, car le nombre de stations supprimées en raison de la brièveté de leur enregistrement de température peut diminuer lorsque de nouvelles données sont ajoutées.
                            </div>
                                """, unsafe_allow_html=True)
                st.write("")
            with cols_analyse_T1[1]:
                with st.popover("➕"): #Homogénéisation des données"):
                    st.write("Dans le cadre de l'homogénéisation, toutes les stations avec moins de 20 ans de données sont supprimées (comme le montre la partie (a) de la figure ci-dessous).")
                    st.write("Les chiffres ci-dessous indiquent:")
                    cols = st.columns(3)
                    with cols[0]:
                        st.write("a) le nombre de stations dont la durée d'enregistrement est d'au moins N années en fonction de N")
                        st.image("https://data.giss.nasa.gov/gistemp/station_data_v4_globe/station_record_length.png")
                    with cols[1]:
                        st.write("b) le nombre de stations de reporting en fonction du temps")
                        st.image("https://data.giss.nasa.gov/gistemp/station_data_v4_globe/number_of_stations.png")
                    with cols[2]:
                        st.write("c) le pourcentage de la superficie hémisphérique située à moins de 1 200 km d'une station de déclaration.")
                        st.image("https://data.giss.nasa.gov/gistemp/station_data_v4_globe/coverage.png")
                st.write("")
            st.write("")
            
            cols_analyse_T2 = st.columns((9,1))
            with cols_analyse_T2[0]:
                st.markdown("""
                    <div style="text-align: justify;">
                    La distribution du nombre de stations par pays est relativement homogène (médiane de 1O stations par pays et EIQ=26), la moyenne étant plus élevée car les grands pays en disposent davantage que les plus petits (les États-Unis en recensent 10 270). 
                    En dehors des grands pays pouvant présenter de grandes variabilités climatiques et une répartition hétérogène des stations sur leur territoire, tant dans un plan horizontal qu'en fonction des reliefs, <b>les données d'enregistrement de chaque station pourront donc être moyennées pour chaque pays</b>.
                    </div>
                    """, unsafe_allow_html=True)
                st.write("")
            with cols_analyse_T2[1]:
                with st.popover("➕"): #Distribution du nombre de stations météorologiques par pays"):
                    st.image("./ressources/DistribTStations.png")
                    st.write("Pour 25% des pays, il y existe au-moins 3 stations, 50% des pays disposent de 1 à 10 stations et 75% des pays disposent de 1 à 29 stations d’enregistrement météorologique. Pour des raisons de visibilité, le graphe ne figure pas les pays de plus de 400 stations.")
            st.write("")
            
            cols_analyse_T3 = st.columns((9,1))
            with cols_analyse_T3[0]:
                st.markdown("""
                    <div style="text-align: justify;">
                    Le regroupement des données mensuelles des stations de chacun des 215 pays permet de <b>réduire le taux de données manquantes, sans modifier la distribution</b> (non uniforme) des années d’enregistrement (fidèle à celle utilisée pour l’analyse GISS v4 : augmentation rapide du nombre mondial de stations d’enregistrement (cumulant 20 ans de données) à partir des années 1950 et jusqu'en 2024). 
                    </div>
                    """, unsafe_allow_html=True)
                st.write("")
            with cols_analyse_T3[1]:
                with st.popover("➕"): #Distribution des années d'enregistrement des stations météorologiques"):
                    sub_cols = st.columns(2)
                    with sub_cols[0]:
                        st.image("./ressources/DistribTY.png")
                        st.markdown("""
                            <div style="text-align: justify;">
                            Si le nombre de pays participant aux mesures de températures augmente régulièrement entre les années 1880 et 1950, ce n’est qu’à partir de ces années que le nombre de pays impliqués a rapidement atteint son maximum et s’est stabilisé depuis.
                            </div>
                            """, unsafe_allow_html=True)
                    with sub_cols[1]:
                        st.image("./ressources/DistribTYC.png")
                        st.markdown("""
                            <div style="text-align: justify;">
                            Pour la majorité des pays (75%), les données d’enregistrement des températures s’étalent sur 76 à 137 années (médiane 105 années). 
                            Le nombre moyen d’années d’enregistrement par pays est de 103,7 ans (écart-type: 34,1, valeurs compriss entre 23 et 145 ans) ; seuls 43 pays (20%) présentent le nombre maximal d'années d'enregistrements entre 1880 et 2024.
                            </div>
                            """, unsafe_allow_html=True)
                    st.write("")
            st.write("")
            
            cols_analyse_T4 = st.columns((9,1))
            with cols_analyse_T4[0]:
                st.markdown("""
                    <div style="text-align: justify;">
                    Si le nombre d'années d'enregistrements est relativement homogène d'un pays à l'autre, néanmoins, les données mensuelles collectées au cours de l'année peuvent être disparates. 
                    Même si le taux de valeurs manquantes reste autour de 8% pour l'ensemble des mois pour tous les pays entre 1880 et 2024, les stations ne fournissent pas un nombre homogène d'enregistrements selon les mois «chauds» ou «froids» de l'année. 
                    <b>La saisonalité doit donc être considérée pour résumer annuellement les données mensuelles d'enregistrements</b>.
                    </div>
                    """, unsafe_allow_html=True)
                st.write("")
            with cols_analyse_T4[1]:
                with st.popover("➕"): #Distribution mensuelle des enregistrements des stations météorologiques"):
                    st.image("./ressources/DistribTMNan.png")
                    st.image("./ressources/DistribTM.png")
                    st.write("Le graphique montre une répartition des températures homogène sur l’ensemble des mois de l’année, avec une température médiane autour de 20°C. Néanmoins, les températures des mois «chauds» sont plus homogènes et concentrées entre 15 et 25°C, alors que les températures des mois «froids» sont plus hétérogènes selon les pays.")
            st.write("")
            
            cols_analyse_T5 = st.columns((9,1))
            with cols_analyse_T5[0]:
                st.markdown("""
                    <div style="text-align: justify;">
                    L'agrégation des données de température par année et par pays (pour les rapprochements avec les variables explicatives selon cette ventilation) permet d'abaisser le taux de valeurs manquantes à 4,23% (957 enregistrements annuels écartés, car ne présentant pas suffisamment de données mensuelles exploitables). 
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("""
                    La moyenne par année et par pays est donc légitime dès lors qu’elle vérifie :
                    <ul><li>le nombre : <b>au-moins 4 valeurs mensuelles disponibles</b></li>
                    <li>et la composition des mois : <b>un nombre équivalent (± 1) de mois «chauds»</b> (parmi avril, mai, juin, juillet, août et septembre) <b>et de mois «froids»</b> (parmi janvier, février, mars, octobre, novembre et décembre).</li></ul>
                    </div>
                    """, unsafe_allow_html=True)
            with cols_analyse_T5[1]:
                with st.popover("➕"): #Distribution mensuelle des enregistrements des stations météorologiques"):
                    sub_cols = st.columns(2)
                    with sub_cols[0]:
                        st.image("https://data.giss.nasa.gov/gistemp/faq/merra2_seas_anom.png")
                    with sub_cols[1]:
                        st.write("Les anomalies de températures par rapport au cycle saisonnier 1980-2015 dans MERRA2. Selon le cycle saisonnier de la température moyenne mondiale, en moyenne, juillet et août sont environ 3,6 °C plus chauds que décembre et janvier. "
                            "Ainsi, un réchauffement de +1°C en décembre serait exceptionnellement chaud pour ce mois, alors qu'il ne serait pas significatif en juillet.")
                    st.image("./ressources/WYAVGT.png")
            st.write("")

            cols_analyse_T6 = st.columns((9,.5,.5))
            with cols_analyse_T6[0]:
                st.markdown("""
                    <div style="text-align: justify;">
                    Bien que le phénomène soit général, l’accroissement des températures moyennes <b>se produit de manière différente selon les pays</b>, en termes de décours (augmentation précoce ou tardive), de régularité (linéaire ou par sauts) et de progression nette depuis l’ère pré-industrielle.
                    De plus, les variations climatiques normales entre pays, entre continents ou entre localisations hémisphériques rendent les comparaisons de températures difficiles, notamment en termes d'échelle.
                    </div>
                    """, unsafe_allow_html=True)
            with cols_analyse_T6[1]:
                with st.popover("➕"): #Évolutions de température moyenne par pays"):
                    st.image("./ressources/YAVGT_Europe.png")
            with cols_analyse_T6[2]:
                with st.popover("➕"): #Évolutions de température moyenne par continent"):
                    st.image("./ressources/YAVGT_Continents.png")
            st.write("")
        
            cols_analyse_T7 = st.columns((9,1))
            with cols_analyse_T7[0]:
                st.markdown("""
                    <div style="text-align: justify;">
                    Pour déterminer les anomalies de température, la <b>moyenne sur la période de référence 1951-1980</b> permet de calculer l'écart avec la température de l'année considérée.
                    Sur cette période, le taux de données manquantes n'est que de 1,58% (absence de température annuelle pour 1 seul pays) et permet de compléter les valeurs manquantes par la moyenne mondiale de l'année considérée.
                    </div>
                    """, unsafe_allow_html=True)
            with cols_analyse_T7[1]:
                with st.popover("➕"): #Température de référence moyenne par pays"):
                    st.image("./ressources/WREFT.png")
            st.write("")

            cols_analyse_T8 = st.columns((9,1))
            with cols_analyse_T8[0]:
                st.markdown("""
                    <div style="text-align: justify;">
                    Par rapport à la période de référence (1951-1980), les anomalies de température montrent <b>un réchauffement d'environ +1,5°C au cours des 5 dernières années</b>. 
                    De plus, après une augmentation continue initiée à la fin des années 1990, une accélaration accentue le phénomène depuis la fin des années 2010.
                    </div>
                    """, unsafe_allow_html=True)
            with cols_analyse_T8[1]:
                with st.popover("➕"): #Température de référence moyenne par pays"):
                    st.image("./ressources/WYANOT1.png")
                    st.image("./ressources/WYANOT2.png")
            st.write("")

            cols_analyse_T9 = st.columns((9,1))
            with cols_analyse_T9[0]:
                st.markdown("""
                    <div style="text-align: justify;">
                    Cas particulier de l'antarctique: ce pays-continent montre une évolution de ses températures moyennes contraire à l'évolution globalement constatée. Comparé à d'autres pays «froids», la distribution des températures semble anormale et sera écartée (d'autant plus qu'aucune donnée démographique ou de production de CO2 n'y est associée).
                    </div>
                    """, unsafe_allow_html=True)
            with cols_analyse_T9[1]:
                with st.popover("➕"): #Antarctique"):
                    st.image("./ressources/ANTAR01.png")
                    st.image("./ressources/ANTAR02.png")
                    st.image("./ressources/ANTAR03.png")
            st.write("")


        with st.expander("Analyses préliminaires - Données CO2, gaz à effet de serre et compléments"):
            st.subheader("Données CO2, gaz à effet de serre et compléments")

            cols_analyse_C1 = st.columns((9,.5,.5))
            with cols_analyse_C1[0]:
                st.markdown("""
                    <div style="text-align: justify;">
                    Le jeux de données comporte de <b>nombreuses valeurs manquantes</b>:
                    <ul><li>20 des 78 variables (26%) ont un taux de valeurs manquantes de 80% et plus.</li>
                    <li>28 des 78 variables (36%) ont un taux de valeurs manquantes de 60% et plus.</li></ul>
                    Parmi les variables trop peu renseignées, figurent celles les plus susceptibles d'expliquer le réchauffement observé: méthane (methane), protoxyde d'azote (nitrous_oxide), g.e.s par habitant (ghg_per_capita), CO2 issu d'autres industries (other_industry_co2). 
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("""
                    <div style="text-align: justify;">
                    Néanmoins, certaines variables calculées (ou déduite) à partir d'autres variables héritent également des pays et/ou années sans valeur, et propagent ou augmentent le taux de valeurs manquantes.
                    Parce qu'elles peuvent être restaurées, <b>les variables résultant d'un calcul pourront donc être écartées. De même pour les regroupements de pays</b>.
                    </div>
                    """, unsafe_allow_html=True)
            with cols_analyse_C1[1]:
                with st.popover("➕"): #Taux de Nan initial"):
                    st.image("./ressources/CO2_Nan00a.png")
            with cols_analyse_C1[2]:
                with st.popover("➕"): #Taux de Nan corrigé"):
                    st.image("./ressources/CO2_Nan00b.png")
            st.write("")

            cols_analyse_C2 = st.columns((9,1))
            with cols_analyse_C2[0]:
                st.markdown("""
                    <div style="text-align: justify;">
                    De plus, la répartition des valeurs manquantes n'est pas homogène au cours du temps: <b>à partir de 1950, le nombre de valeurs manquantes baisse considérablement</b>, à la fois pour les variables, mais aussi pour chaque pays.
                    </div>
                    """, unsafe_allow_html=True)
            with cols_analyse_C2[1]:
                with st.popover("➕"): #Taux de Nan selon les années"):
                    sub_cols=st.columns(3)
                    with sub_cols[0]:
                        st.image("./ressources/CO2_Nan01.png")
                    with sub_cols[1]:
                        st.image("./ressources/CO2_Nan02.png")
                    with sub_cols[2]:
                        st.image("./ressources/CO2_Nan03.png")
            st.write("")

            cols_analyse_C3 = st.columns((9,1))
            with cols_analyse_C3[0]:
                st.markdown("""
                    <div style="text-align: justify;">
                    L'homogénéité dans la distribution des données est inégale d'une variable à l'autre, selon les pays et selon la période considérée.
                    La réduction à la <b>période 1950-2022 permet d'homogénéiser les distributions des variables</b> ; les regroupements de pays à l'échelle d'un continent concentrent également des distributions plus homogènes.
                    Compte tenu de la grande variablilité des valeurs, la <b>transformation par RobustScaler</b> semble indiquée : notamment pour gérer la présence de valeurs aberrantes et/ou extrêmes et, compte tenu de la grande diversité des tendances évolutives des variables qui ne suivent pas nécessairement une distribution normale.
                    <b>Les variables avec une présence extrême d'outliers devront être abandonnées</b>.
                    </div>
                    """, unsafe_allow_html=True)
            with cols_analyse_C3[1]:
                with st.popover("➕"): #Distribution Avant et après transformation Robustscaler"):
                    st.write("Distribution des variables AVANT transformation")
                    st.image("./ressources/CO2_Distrib01.png")
                    st.write("Distribution des variables APRÈS transformation")
                    st.image("./ressources/CO2_Distrib02.png")
            st.write("")

            cols_analyse_C4 = st.columns((9,.5,.5))
            with cols_analyse_C4[0]:
                st.markdown("""
                    <div style="text-align: justify;">
                    Parmi les méthodes de <b>correction des valeurs manquantes</b>, une première approche consistait à identifier les courbes d'évolution mondiale pour chaque variable, puis à recalculer pour les variables correspondantes les paramètres de cette courbe pour chaque pays afin de compléter les valeurs manquantes.
                    Si cette méthode semblait plus fine qu'un remplacement basique par la moyenne ou la médiane, elle a été écartée du fait de l'apparition de sauts brutaux dans l'évolution temporelle des données.
                    De plus, il existe des discontinuités dans les années présentées pour certains pays ; sans traitement préalable, le rapprochement avec les données de températures ferait apparaître de nouvelles valeurs manquantes.
                    </div>
                    """, unsafe_allow_html=True)
            with cols_analyse_C4[1]:
                with st.popover("➕"): #Détermination monde"):
                    st.write("Détermination des courbes de tendance")
                    sub_cols=st.columns(2)
                    with sub_cols[0]:
                        st.image("./ressources/Meth1_monde01.png")
                        st.image("./ressources/Meth1_monde03.png")
                    with sub_cols[1]:
                        st.image("./ressources/Meth1_monde02.png")
                        st.image("./ressources/Meth1_monde03.png")
            with cols_analyse_C4[2]:
                with st.popover("➕"): #Correction exemples"):
                    st.write("Exemples de données complétées")
                    st.image("./ressources/Meth1_corr01.png")
                    st.image("./ressources/Meth1_corr02.png")
                    st.image("./ressources/Meth1_corr03.png")
            st.write("")

            cols_analyse_C5 = st.columns((9,.5,.5))
            with cols_analyse_C5[0]:
                st.markdown("""
                    <div style="text-align: justify;">
                    À partir de cette première approche, a été réalisé un <b>ajout des années manquantes pour obtenir une continuité chronologique</b> aux pays concernés afin de couvrir au-moins la période 1941-2022.
                    La méthode ayant le mieux complété les valeurs manquantes a consisté à effectuer, pour chaque pays (et chaque variable) :
                    <ol><li>Sur la période à partir de la première année avec des données manquantes :
                        <ol><li>une première interpolation polynomiale d'ordre 3 sur les données manquantes dans la période comportant des trous</li>
                        <li>une seconde interpolation polynomiale d'ordre 3 (éventuellement recalculée à partir des données complétées à l'étape précédente) pour compléter jusqu'en 2022.</li></ol></li>
                    <li>Sur la période en amont de la première année avec des données, une imputation par la valeur médiane recalculée après les transformations précédentes.</li></ol>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("""
                    <div style="text-align: justify;">
                    Les méthodes IterrativeImputer, KNNImputer ont été testées en comparaison de SimpleImputer ont été testées sans donner de meilleurs résultats.
                    </div>
                    """, unsafe_allow_html=True)
            with cols_analyse_C5[1]:
                with st.popover("➕"): #Illustration complétion"):
                    st.write("Méthode de complétion par phases successives")
                    st.image("./ressources/CO2_FillNan01.png")
            with cols_analyse_C5[2]:
                with st.popover("➕"): #Illustration autres méthodes"):
                    st.write("Comparaison avec autres méthodes")
                    st.image("./ressources/CO2_FillNan02.png")
                    st.image("./ressources/CO2_FillNan03.png")
            st.write("")


# PREPROCESSING
if page == sections[2] :
    with st.container():
        st.header(f"{sections[2]}")
        st.subheader("Données de température")
        with st.expander("Preprocessing Data Temperatures", expanded=True):
            st.markdown("""
            <div style="text-align: justify;">
            <ol>
            <li>Intégration des données de pays (Codes ISO, Noms, Continents) à partir des codes FIPS des stations météorologiques</li>
            <li>Suppression des données pour l'Antarctique</li>
            <li>Agrégation moyenne des données mensuelles des stations par pays</li>
            <li>Calcul de la moyenne annuelle par pays (YAVGT) en tenant compte:
                <ol><li>du nombre de mois supérieur ou égal à 4</li>
                <li>de l'équilibre des mois froids et chauds pour la moyenne</li></ol></li>
            <li>Complétion des valeurs manquantes de température moyenne annuelle (YAVGT) par:
                <ol><li>Ajout des années manquantes pour continuité à compter de 1941 pour chaque pays</li>
                <li>interpolation linéaire couvrant la période la première à la dernière année avec données manquantes</li>
                <li>imputation SimpleImputer Moyenne du reste des années avec données manquantes</li></ol></li>
            <li>Calcul de la température moyenne (REFT) sur la période de référence 1951-1980 pour chaque pays</li>
            <li>Calcul des anomalies de température (YANOT), écart entre YAVGT et REFT pour chaque année et pour chaque pays</li>
            <li>Sélection des données sur la plage 1950-2022</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            st.write("")

        st.subheader("Données CO2")
        with st.expander("Preprocessing Data CO2", expanded=True):
            st.markdown("""
            <div style="text-align: justify;">
            <ol>
            <li>Intégration des données de pays (Codes ISO, Noms, Continents) à partir des codes ISO présents dans le jeu de données de températures</li>
            <li>Suppression des données de pays issues du jeu de données de températures qui n'ont pas de correspondance dans le jeu CO2 (localités avec températures, mais sans population)</li>
            <li>Inclusion pour chaque pays des années manquantes pour la période 1950-2023</li>
            <li>Réduction des variables avec valeurs manquantes selon les motifs suivants:
                <ol><li>Les variables sont déjà des interprétations (exemple : temperature_change_from_co2)</li>
                <li>Les variables peuvent être recalculées à partir des autres variables (exemple ghg_per_capita) en respectant les unités tonnes / personnes
                <li>Les variables ont un taux de NaN proche de 75%</li></ol></li>
            <li>Complétion des valeurs manquantes des variables par pays par:
                <ol><li>une première interpolation polynomiale d'ordre 3 sur les données manquantes dans la période comportant des trous</li>
                <li>une seconde interpolation polynomiale d'ordre 3 (éventuellement recalculée à partir des données complétées à l'étape précédente) pour compléter jusqu'en 2022.</li>
                <li>une imputation SimpleImputer médiane recalculée après les transformations précédentes pour compléter la période en amont de la première année avec des données</li></ol></li>
            <li>Suppression des variables avec une présence extrême d'outliers (co2_including_luc) et avec valeurs manquantes ne pouvant être recalculées pour 40% des pays (flaring_co2)</li>
            <li>Sélection des données sur la plage 1950-2022</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        st.write("")

        st.subheader("Données CO2 atmosphérique")
        with st.expander("Preprocessing Data CO2 atmosphérique", expanded=True):
            st.markdown("""
            <div style="text-align: justify;">
            <ol>
            <li>Inclusion des années manquantes pour la période 1950-2024</li>
            <li>Complétion des valeurs manquantes par année par interpolation polynomiale d'ordre 2 :
                <ol><li>la valeur de départ pour l'année 1950 est arbitrairement remplacée par la valeur minimale pour procéder à l'interpolation</li>
                <li>après l'interpolation, la valeur pour l'année 1950 est de nouveau remplacée par la plus petite valeur obtenue sur la nouvelle série de données</li>
            <li>Intégration de chaque valeur annuelle mondiale pour chaque pays</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        st.write("")


        st.subheader("Ajustement des données")
        with st.expander("Preprocessing Ajustement des données", expanded=False):
            st.markdown("""
            <div style="text-align: justify;">
            <ol><li><b>Ajout d'indices de températures:</b>
                <ul><li>Calcul des moyennes de températures annuelles sur période glissante de 5 et 10 ans (YAVGTp5 et YAVGTp10)</li>
                <li>Calcul des moyennes des anomalies de températures annuelles sur période glissante de 5 et 10 ans (YANOTp5 et YANOTp10)</li>
                <li>Pour réaliser ces calculs, les données de températures (YAVGT) devaient exister à partir de 1941:
                    <ol><li>Ajout des années manquantes pour continuité à compter de 1941 pour chaque pays</li>
                    <li>interpolation linéaire couvrant la période la première à la dernière année avec données manquantes</li>
                    <li>imputation SimpleImputer Moyenne du reste des années avec données manquantes</li>
                    <li>Calcul des anomalies de température (YANOT) pour les années de la nouvelle période</li></ol></li>
            </ul>
            <li><b>Ajout de l'indice d'association Année-Pays:</b>
            création d'une variable (ISO_YEAR) permettant de concatener selon un entier au format ####YYYY.</li>
            <li><b>Réduction de la période d'analyse à 1988-2022 (35 ans) pour une meilleure captation du phénomène</b></li>
            <li><b>Transformation des données numériques par Robustscaler :</b>
                <ul><li>Les données numériques du jeu CO2 sont transformées dans leur globalité: actuellement, l'évolution de la démographie, du pib ou des autres émissions de g.e.s. ne dépend pas de la température annuelle moyenne et les données CO2 collectées n'ont pas été influencées par l'acquisition des données de températures.</li>
                <li>les données numériques des indices de températures sur les périodes -5 ans et -10 ans sont également transformées uniformément avant le split.</li></ul></li>
            <li><b>Séparation en jeu de test et jeu d'entraînement selon le ratio classique 80/20:</b>
                <ul><li>jeu d'entraînement : période 1988-2015</li>
                <li>jeu de test : période 2016-2022</li></ul></li>
            <li><b>Méthodes de sélection du transformer, des indices fournis, de l'année de séparation des jeux de données et de l'application des transformation avant split selon la MAE la plus faible.</b></li> 
                
            </ol>
            </div>
            """, unsafe_allow_html=True)
        st.write("")

        st.subheader("Analyse des données")
        with st.expander("Analyse des données", expanded=False):
            st.markdown("""
            <b>Distribution de la variable cible</b>
            """, unsafe_allow_html=True)
            st.image("./ressources/Distrib_Target.png")

            st.markdown("""
            <b>Interactions de la variable cible avec les autres variables</b>
            """, unsafe_allow_html=True)
            with st.popover("➕"): #Interactions"):
                st.image("./ressources/TargetInteractions.png")

            st.markdown("""
            <b>Matrices de corrélation</b>
            """, unsafe_allow_html=True)
            Mat_cols = st.columns(2)
            with Mat_cols[0]:
                st.image("./ressources/MatCorr01.png")
            with Mat_cols[1]:
                st.image("./ressources/MatCorr02.png")

            st.markdown("""
            <b>Analyse en composantes principales (ACP)</b>
            """, unsafe_allow_html=True)
            Mat_cols = st.columns(2)
            with Mat_cols[0]:
                st.image("./ressources/ACP01.png")
            with Mat_cols[1]:
                st.image("./ressources/ACP02.png")

            st.write("")
            

# VISUALISATIONS
if page == sections[3] :
    with st.container():
        st.header(f"{sections[3]}")
        st.image("https://www.nasa.gov/wp-content/uploads/2024/06/maytemp-line-big.gif")

# MODELES SUPERVISES
if page == sections[4] :
    with st.container():
        st.header(f"{sections[4]}")

        models_list=["Régression linéaire",
                    "Régression par arbres décisionnels",
                    "Régression Lasso",
                    "Régression ElasticNet",]

        st.write("Compte tenu du caractère continu de notre variable cible, les modèles de régression sont les mieux indiqués.")        
        selected_model = st.selectbox("Sélectionnez un modèle", models_list)
        if selected_model==models_list[0]:
            st.subheader("Modèle de régression linéaire")
            cols_model1 = st.columns((2,2,6))
            with cols_model1[0]:
                st.write("La performance du Modèle pour le set de Training")
                st.write("l'erreur RMSE est ",0.3938966928393961)
                st.write("l'erreur MAE est ",0.26540326184951857)
                st.write("le score R2 est ",0.7822704129630749)
            with cols_model1[1]:
                st.write("La performance du Modèle pour le set de Test")
                st.write("l'erreur RMSE est ",0.3792660596186801)
                st.write("l'erreur MAE est ",0.27306251902447154)
                st.write("le score R2 est ",0.8396576193597625)
            with cols_model1[2]:
                st.image("./ressources/Model01.png")
        st.write("")

        if selected_model==models_list[1]:
            st.subheader("Modèle de Regression par arbres décisionnels")
            cols_model2 = st.columns((4,6))
            with cols_model2[0]:
                sub_cols = st.columns(2)
                with sub_cols[0]:
                    st.write("La performance du Modèle pour le set de Training")
                    st.write("l'erreur RMSE est ",3.0084421802235703e-17)
                    st.write("l'erreur MAE est ",6.412281420128555e-18)
                    st.write("le score R2 est ",1.0)
                with sub_cols[1]:
                    st.write("La performance du Modèle pour le set de Test")
                    st.write("l'erreur RMSE est ",0.8325213156963459)
                    st.write("l'erreur MAE est ",0.5818021529324424)
                    st.write("le score R2 est ",0.22740642527833843)
                st.subheader("Importance des variables")
                st.image("./ressources/Model02b.png")
            with cols_model2[1]:
                st.image("./ressources/Model02a.png")



        if selected_model==models_list[2]:
            st.subheader("Modèle de Regression Lasso")
            cols_model1 = st.columns((2,2,6))
            with cols_model1[0]:
                st.write("La performance du Modèle pour le set de Training")
                st.write("l'erreur RMSE est ",0.39753930452815345)
                st.write("l'erreur MAE est ",0.26802089381505256)
                st.write("le score R2 est ",0.7782248267889735)
                st.write("Meilleur paramètre alpha sur jeu d'entrainement: {'alpha': ",1e-05,"}")
            with cols_model1[1]:
                st.write("La performance du Modèle pour le set de Test")
                st.write("l'erreur RMSE est ",0.38026193184820367)
                st.write("l'erreur MAE est ",0.2720336769464201)
                st.write("le score R2 est ",0.8388144636719062)
                st.write("Meilleur paramètre alpha sur jeu de test: {'alpha': ",1e-05,"}")
            with cols_model1[2]:
                st.image("./ressources/Model03.png")
            

        if selected_model==models_list[3]:
            st.subheader("Modèle de Regression ElasticNet")
            cols_model1 = st.columns((2,2,6))
            with cols_model1[0]:
                st.write("La performance du Modèle pour le set de Training")
                st.write("l'erreur RMSE est ",0.39751820415704603)
                st.write("l'erreur MAE est ",0.26802950055347224)
                st.write("le score R2 est ",0.7782483686839001)
                st.write("Meilleurs paramètres sur jeu d'entrainement: {'alpha': ",1e-05," 'l1_ratio': ",0.1,"}")
            with cols_model1[1]:
                st.write("La performance du Modèle pour le set de Test")
                st.write("l'erreur RMSE est ",0.3802528740381691)
                st.write("l'erreur MAE est ",0.2721035778453997)
                st.write("le score R2 est ",0.8388221424346626)
                st.write("Meilleur paramètre alpha sur jeu de test: {'alpha': ",1e-05," 'l1_ratio': ",1.0,"}")
            with cols_model1[2]:
                st.image("./ressources/Model04.png")


# SERIES TEMPORELLES
if page == sections[5] :

    def prediction_temperature(country):
        # 2. Copier la colonne "YEAR" en "date"
        dataset['date'] = dataset['YEAR']
        dataset['YEAR'] = pd.to_datetime(dataset['date'], format='%Y')
        # 3. Changer son format pour "%Y"
        dataset['date'] = pd.to_datetime(dataset['date'], format='%Y')
        # 4. Filtrer pour ne retenir que le pays choisi
        df_country = dataset[dataset['Name_EN'] == country]
        # 5. Passer la "date" en index
        df_country.set_index('date', inplace=True)
        # 6. Ne garder que quelques colonnes
        df_country = df_country[['YEAR', 'YAVGT', 'Name_EN']]

        # 7. Modèle SARIMAX
        model_sarimax = SARIMAX(df_country['YAVGT'], order=(1, 1, 1), seasonal_order=(0, 1, 1, 5))
        sarimax_fit = model_sarimax.fit(disp=False)
        sarimax_forecast = sarimax_fit.get_forecast(steps=10)
        sarimax_pred = sarimax_forecast.predicted_mean

        # 7. Modèle Holt-Winters
        model_hw = ExponentialSmoothing(df_country['YAVGT'], trend='mul', seasonal='mul', seasonal_periods=5)
        hw_fit = model_hw.fit()
        hw_forecast = hw_fit.forecast(steps=10)

        # Créer un graphique interactif avec Plotly
        fig = go.Figure()

        # Températures réelles
        fig.add_trace(go.Scatter(x=df_country.index, y=df_country['YAVGT'], mode='lines', name='Températures Réelles', line=dict(color='blue')))

        # Prévisions SARIMAX
        future_dates = pd.date_range(start=df_country.index[-1] + pd.DateOffset(years=1), periods=10, freq='Y')
        fig.add_trace(go.Scatter(x=future_dates, y=sarimax_pred, mode='lines', name='Prévisions SARIMAX', line=dict(color='orange')))

        # Prévisions Holt-Winters
        fig.add_trace(go.Scatter(x=future_dates, y=hw_forecast, mode='lines', name='Prévisions Holt-Winters', line=dict(color='green')))

        # Mise à jour du layout
        fig.update_layout(title=f'Prévisions de Températures pour {country}',
                          xaxis_title='Année',
                          yaxis_title='Température (°C)',
                          legend=dict(x=0, y=1))

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig)

    with st.container():
        st.header(f"{sections[5]}")
        st.markdown("Sélection du modèle et validation par la **RMSE**. D’abord testée sur un pays (la France) puis validées par la moyenne des températures mondiale.")
        st.dataframe(monde.head(5))
        st.image("./ressources/decomposition.png",
                caption='Les différentes tentatives de décomposition de la série temporelle ont permis de montrer une tendance de type “multiplicative” et une saisonnalité de 5 ans.',)#                use_column_width=True)
        st.markdown("#### Cette approche a été validée par l'autocorrélation.")
        st.image("./ressources/autocorrélation.png",
                caption='L autocorrélation a confirmé nos choix en multiplicatif et en saisonnalité',)#                use_column_width=True)
        st.markdown("#### Deux modèles se sont dégagés par leur performance SARIMAX HoltWinters")
        st.image("./ressources/RMSE.png",
                caption='Performance des modèles sur les moyennes annuelles modiales ',)#                use_column_width=True)
        st.markdown("#### Avantages et inconvénients pour chaque modèle :")
        st.markdown("- Interprétabilité du SARIMAX (nombreuses valeurs d’évaluation dans result.summary\n- Holt-Winters donne plus d’importance aux toutes dernières valeurs observées dans la série temporelle.")
        st.markdown("Le choix est fait de garder l'exécution de ces deux modèles avec un Trend Multiplicatif et une saisonnalité de 5 ans. En assumant que SARIMAX sous-évalue légèrement et que Holt-Winters surévalue légèrement. C'est un peu comme garder un intervale de confiance de 15% entre les deux prévisions ")
        st.markdown("### Prevision")
        # Sélection du pays à partir de la liste déroulante
        selected_country = st.selectbox('Sélectionnez un pays', countries_list)
        if st.button('Exécuter la prévision'):
            prediction_temperature(selected_country)
        #HWMonde = "./ressources/Holt-Winters-MONDE.png"
        #SARIMAXMonde = "./ressources/SARIMAX-MONDE.png"


# REMERCIEMENTS
if page == sections[6] :
    with st.container():
        st.header(f"{sections[6]}")
        st.subheader("Les auteurs")
        # Les auteurs
        cols = st.columns(2)
        with cols[0]:
            sub_cols=st.columns((1,3))
            with sub_cols[0]:
                st.image("./ressources/Sebastien.jpg")#, width=200)
            with sub_cols[1]:
                st.subheader("Sébastien LAGARDE-CORRADO")
                st.caption("Chargé d’études RH - CHU de Bordeaux")
            #st.divider()
                with st.popover("ℹ️"):
                    st.markdown("De formation scientifique, j’ai été sensibilisé à la méthodologie de la recherche et à une approche globale de systèmes complexes, notamment biologiques." 
                    "Ma sensibilité environnementale, aiguisée par ma parentalité et mon expérience en milieu hospitalier par les enjeux socio-économiques et stratégiques autour de la santé, m'a conduit à m'intéresser à la climatologie."
                    "La littérature disponible depuis la fin des années 1990 est aussi dense que variée, avec autant d’approches sérieuses que contestées. Cependant, le dernier rapport (6ème) d’évaluation publié en mars 2023 et les données du GIEC/IPCC (Intergouvernemental panel on climate change) m'ont été utiles.")
                    st.markdown("En dehors des implications individuelles, le CHU de Bordeaux s’est lancé en octobre 2022 dans un plan de transformation écologique et de sobriété selon 3 objectifs: limiter les impacts environnementaux et adapter le CHU aux crises écologiques, déployer de nouvelles compétences et créer une culture de la transformation écologique et inventer de nouvelles complémentarités entre les soins et l’approche écologique.")
        with cols[1]:
            sub_cols=st.columns((1,3))
            with sub_cols[0]:
                st.image("./ressources/Damien.png")
            with sub_cols[1]:
                st.subheader("Damien SELOSSE")
                st.caption("Direction de projet innovation - 109 l’innovation dans les veines")
                with st.popover("ℹ️"):
                    st.markdown("Depuis 10 ans dans le conseil en innovation, mes missions ont pour but d’améliorer l’efficacité des projets d’innovation. La maîtrise de méthodes et le management de l’innovation. La data analyse m’a toujours permis d’apporter au projet une structuration des connaissances créées mais aussi une ouverture vers l’exploration de solutions innovantes. Aujourd’hui, avec les apports des IA génératives, ces métiers de l’innovation changent. Mon objectif est d’en tirer le meilleur avantage pour les projets d’innovation. Ces projets aujourd’hui créent de la connaissance nouvelle, les modèles d’apprentissage devraient permettre d’en profiter et rendre cette connaissance actionnable pour mes clients.")
                    st.markdown("J’ai une bonne affinité pour ce sujet. Je suis particulièrement sensible aux actions concrètes réalisables par chacun de nous. J’ai obtenu la certification du CNED pour le [Super Badge du Climat et de la biodiversité](https://openbadgepassport.com/app/badge/info/575913)")
    st.divider()
    st.subheader("Remerciements DataScientest ")
    st.markdown("Nous vous remerçions pour toute l'aide que vous nous avez apportée durant notre formation, et en particulier **Yohan Cohen** notre tuteur.")
    st.markdown("Nous avons également une pensée particulière pour **Jérémy Bazille** (CHU d'Amiens) qui a été à nos côtés au démarrage du projet ; son évolution professionnelle ne lui ayant pas permis de le poursuivre et le finaliser avec nous.")
    st.divider()
    if "clap" not in st.session_state:
        st.session_state.clap = False
    def release_the_balloons():
        st.balloons()
    st.button("Merci de votre attention ! ", key="clap", type="primary", on_click=release_the_balloons, args=())
        
