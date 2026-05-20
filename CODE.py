import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import datetime
import io
import os

# ==============================================================================
# 1. CONFIGURATION DE LA PAGE & STYLE CUSTOMISÉ (CHARTE ROUGE, NOIR, BLANC)
# ==============================================================================
st.set_page_config(
    page_title="Afriland First Bank - BI Satisfaction Client",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injection de styles CSS personnalisés pour respecter strictement la charte couleur
st.markdown("""
    <style>
        /* Couleurs de la charte Afriland First Bank */
        :root {
            --primary-color: #D32F2F; /* Rouge Ardent */
            --secondary-color: #1A1A1A; /* Noir */
            --bg-color: #FFFFFF; /* Blanc */
        }
        
        .main-title {
            color: #D32F2F;
            font-size: 30px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 5px;
            text-transform: uppercase;
            border-bottom: 3px solid #1A1A1A;
            padding-bottom: 10px;
        }
        .section-subtitle {
            color: #1A1A1A;
            font-size: 16px;
            font-style: italic;
            text-align: center;
            margin-bottom: 25px;
        }
        
        /* Cartes KPI Stylisées */
        .kpi-card {
            background-color: #F8F9FA;
            border-left: 6px solid #D32F2F;
            border-right: 1px solid #E0E0E0;
            border-top: 1px solid #E0E0E0;
            border-bottom: 1px solid #E0E0E0;
            padding: 20px;
            border-radius: 4px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.05);
            text-align: center;
        }
        .kpi-value {
            font-size: 30px;
            font-weight: bold;
            color: #1A1A1A;
        }
        .kpi-label {
            font-size: 13px;
            color: #555555;
            font-weight: 600;
            margin-top: 5px;
            text-transform: uppercase;
        }
        
        /* Blocs d'alertes et de faits saillants */
        .insight-box {
            background-color: #FFF5F5;
            border: 1px solid #FEB2B2;
            border-left: 6px solid #D32F2F;
            padding: 18px;
            border-radius: 4px;
            margin-top: 15px;
            margin-bottom: 15px;
        }
        
        /* Boutons personnalisés */
        div.stButton > button:first-child {
            background-color: #D32F2F;
            color: white;
            border: 1px solid #1A1A1A;
            padding: 10px 24px;
            border-radius: 4px;
            font-weight: bold;
            width: 100%;
        }
        div.stButton > button:first-child:hover {
            background-color: #1A1A1A;
            color: white;
            border: 1px solid #D32F2F;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. FONCTION DE PRÉ-TRAITEMENT ET NETTOYAGE DES DONNÉES
# ==============================================================================
def load_and_preprocess(file_source):
    # Lecture adaptative pour neutraliser l'erreur de codec 'utf-8' sur les caractères accentués
    try:
        df = pd.read_csv(file_source, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_source, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(file_source, encoding='utf-8-sig')
    
    # Nettoyage préventif des espaces multiples et invisibles en début/fin des en-têtes de colonnes
    df.columns = df.columns.str.strip()
    
    # Cartographie de renommage stricte calquée sur vos colonnes réelles
    rename_dict = {
        'start': 'date_debut',
        'end': 'date_fin',
        'A quelle date avez-vous effectué votre dernière opération ?': 'date_operation',
        'Dans quelle agence avez-vous effectué cette opération ?': 'agence',
        'Dans quel type de compte avez-vous effectué cette opération ?': 'type_compte',
        'Quelle était l’opération effectuée ?': 'operation',
        "Quelle appréciation faites-vous du temps mis à l'agence de  ${S0Q0Tres satisfaisante} avant d’être servi ?": 'temps_attente',
        'Quelle appréciation faites-vous de la qualité de l’accueil de l’agent de guichet/caisse qui vous a reçu ?': 'accueil_agent',
        'Comment évaluez-vous l’effort que vous avez fourni avant d’être servi ? (prise de ticket, remplissage du bordereau, traitement de l’opération, etc.)': 'effort_client',
        'Globalement, comment avez-vous trouvé la qualité de service offerte par l’agence de  ${S0Q0Tres satisfaisante} au niveau de ses guichets/caisses ?': 'satisfaction_globale',
        'Globalement, comment avez-vous trouvé la qualité de service offerte par l’agence de  ${S0Q0Tres satisfaisante} au niveau de ses guichets/caisses ? ': 'satisfaction_globale',
        "Sur la base de votre expérience à l'issue de cette opération, sur une échelle de 1 à 10 jusqu’à combien seriez-vous prêt à recommander Afriland First Bank à un proche ?": 'nps_score',
        "Sur la base de votre expérience à l'issue de cette opération, sur une échelle de 1 à 10 jusqu’à combien seriez-vous prêt à recommander Afriland First Bank à un proche ? ": 'nps_score',
        'Qu’est-ce que vous n’avez pas apprécié dans le service ?': 'verbatim_negatif',
        'Que devons-nous améliorez dans ce service pour mieux vous satisfaire?': 'verbatim_amelioration',
        'Qu’est-ce qui vous a marqué positivement dans le service ?': 'verbatim_positif'
    }
    
    df = df.rename(columns=rename_dict)
    
    # Nettoyage et uniformisation du libellé des agences
    if 'agence' in df.columns:
        df['agence'] = df['agence'].astype(str).str.strip().str.upper()
        
    # Conversion sécurisée du score NPS en valeur numérique
    df['nps_score'] = pd.to_numeric(df['nps_score'], errors='coerce')
    
    # Segmentation officielle du Net Promoter Score
    def segmenter_nps(score):
        if pd.isna(score): return np.nan
        if score >= 9: return 'Promoteur'
        elif score >= 7: return 'Passif'
        else: return 'Détracteur'
        
    df['nps_class'] = df['nps_score'].apply(segmenter_nps)
    
    # Encodage numérique pour correspondances avec l'échelle de Likert
    likert_mapping = {
        'Très satisfaisante': 5, 'Satisfaisante': 4, 'Neutre': 3, 'Peu satisfaisante': 2, 'Pas du tout satisfaisante': 1,
        'Tres satisfaisante': 5, 
        'Très facile': 5, 'Facile': 4, 'Moyennement difficile': 3, 'Difficile': 2, 'Très difficile': 1
    }
    
    # Nettoyage des chaînes textuelles avant mapping pour éviter les échecs d'alignement
    for col in ['satisfaction_globale', 'temps_attente', 'accueil_agent', 'effort_client']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    df['score_satisfaction_num'] = df['satisfaction_globale'].map(likert_mapping)
    df['score_attente_num'] = df['temps_attente'].map(likert_mapping)
    df['score_accueil_num'] = df['accueil_agent'].map(likert_mapping)
    df['score_effort_num'] = df['effort_client'].map(likert_mapping)
    
    return df

# ==============================================================================
# 3. CONSTRUIRE LA BARRE LATÉRALE (SIDEBAR) & CHARGEMENT SÉCURISÉ
# ==============================================================================
with st.sidebar:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    
    # CORRECTION ICI : Gestion sécurisée de l'absence du fichier image sans planter l'application
    if os.path.exists("LOGO_AFRILAND.png"):
        st.image("LOGO_AFRILAND.png", width=70, caption="Afriland First Bank")
    else:
        # Affichage d'une icône alternative si le logo local est introuvable sur Streamlit Cloud
        st.markdown("<h1 style='text-align: center; margin:0;'>🏦</h1>", unsafe_allow_html=True)
        
    st.markdown("<h4 style='color: #D32F2F; margin-top:5px; font-weight:bold;'>PILOTAGE SATISFACTION</h4>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("📁 Flux d'entrée des données")
    uploaded_file = st.file_uploader(
        "Importer un nouveau fichier d'enquête (.csv)", 
        type=["csv"],
        help="Glissez-déposez ici le fichier de la semaine ou de la journée courante pour actualiser les indicateurs."
    )
    
    df_clean = None
    if uploaded_file is not None:
        try:
            df_clean = load_and_preprocess(uploaded_file)
            st.success("Données de l'enquête injectées avec succès !")
        except Exception as e:
            st.error(f"Erreur d'analyse du fichier : {e}")
    else:
        fichier_historique = "Exemple_donnée.xlsx - Feuil1.csv"
        if os.path.exists(fichier_historique):
            df_clean = load_and_preprocess(fichier_historique)
            st.info("Affichage basé sur les données historiques de référence.")
        else:
            st.warning("⚠️ Aucun fichier détecté. Veuillez importer un fichier d'enquête pour activer l'analyse.")
            
    st.markdown("---")
    
    st.subheader("🗺️ Menu Pilote")
    page = st.radio(
        "Sélectionner la granularité :",
        ["🏢 Vision Globale Nationale", "📍 Analyse Détaillée par Agence", "🔮 Modélisation & Prédictions S+1", "📝 Rapport de Synthèse & Téléchargement"]
    )

# ==============================================================================
# 4. EXÉCUTION DU TABLEAU DE BORD SI LES DONNÉES SONT DISPONIBLES
# ==============================================================================
if df_clean is None:
    st.markdown("<div class='main-title'>Plateforme d'Analyse Clientèle</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Veuillez charger votre fichier Excel converti en CSV à gauche pour commencer.</p>", unsafe_allow_html=True)
else:
    # --------------------------------------------------------------------------
    # PAGE 1 : VISION GLOBALE NATIONALE
    # --------------------------------------------------------------------------
    if page == "🏢 Vision Globale Nationale":
        st.markdown("<div class='main-title'>Réseau Cameroun - Tableau de Bord National</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Indicateurs macroscopiques consolidés de la performance de l'expérience client aux guichets</div>", unsafe_allow_html=True)
        
        total_rep = len(df_clean)
        
        nps_classes = df_clean['nps_class'].value_counts()
        promos = nps_classes.get('Promoteur', 0)
        detracs = nps_classes.get('Détracteur', 0)
        nps_valides = df_clean['nps_class'].dropna().count()
        nps_global = ((promos - detracs) / nps_valides * 100) if nps_valides > 0 else 0
        
        sat_counts = df_clean['satisfaction_globale'].value_counts()
        positives = sat_counts.get('Satisfaisante', 0) + sat_counts.get('Tres satisfaisante', 0) + sat_counts.get('Très satisfaisante', 0)
        tx_satisfaction = (positives / df_clean['satisfaction_globale'].dropna().count() * 100) if df_clean['satisfaction_globale'].dropna().count() > 0 else 0
        
        avg_accueil = df_clean['score_accueil_num'].mean()
        
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_rep}</div><div class='kpi-label'>Réponses Collectées</div></div>", unsafe_allow_html=True)
        with k2:
            color_nps = "#D32F2F" if nps_global < 0 else "#1A1A1A"
            st.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:{color_nps};'>{nps_global:.1f}</div><div class='kpi-label'>NPS Global Réseau</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{tx_satisfaction:.1f}%</div><div class='kpi-label'>Taux de Satisfaction</div></div>", unsafe_allow_html=True)
        with k4:
            val_acc = f"{avg_accueil:.2f} / 5" if not pd.isna(avg_accueil) else "N/A"
            st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{val_acc}</div><div class='kpi-label'>Moyenne Accueil Guichet</div></div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### 📊 Profil de la Satisfaction Globale")
            sat_dist = df_clean['satisfaction_globale'].value_counts().reset_index()
            sat_dist.columns = ['Appréciation', 'Volume']
            fig1 = px.bar(sat_dist, x='Volume', y='Appréciation', orientation='h', color_discrete_sequence=['#D32F2F'])
            fig1.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig1, use_container_width=True)
            
        with g2:
            st.markdown("#### 🎯 Répartition des Segments Net Promoter Score")
            nps_dist = df_clean['nps_class'].value_counts().reset_index()
            nps_dist.columns = ['Classe', 'Total']
            fig2 = px.pie(nps_dist, values='Total', names='Classe', color_discrete_map={'Promoteur': '#1A1A1A', 'Passif': '#CCCCCC', 'Détracteur': '#D32F2F'})
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 🔍 Analyse de la Satisfaction par Segment d'Activité")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Satisfaction Moyenne par Catégorie de Compte Client**")
            compte_res = df_clean.groupby('type_compte')['score_satisfaction_num'].mean().reset_index()
            fig3 = px.bar(compte_res, x='type_compte', y='score_satisfaction_num', color_discrete_sequence=['#1A1A1A'])
            fig3.update_layout(yaxis_range=[1,5], plot_bgcolor='white')
            st.plotly_chart(fig3, use_container_width=True)
        with s2:
            st.markdown("**Satisfaction Moyenne selon la Typologie d'Opération**")
            op_res = df_clean.groupby('operation')['score_satisfaction_num'].mean().reset_index().sort_values(by='score_satisfaction_num')
            fig4 = px.bar(op_res, y='operation', x='score_satisfaction_num', orientation='h', color_discrete_sequence=['#D32F2F'])
            fig4.update_layout(xaxis_range=[1,5], plot_bgcolor='white')
            st.plotly_chart(fig4, use_container_width=True)

    # --------------------------------------------------------------------------
    # PAGE 2 : ANALYSE DÉTAILLÉE PAR AGENCE
    # --------------------------------------------------------------------------
    elif page == "📍 Analyse Détaillée par Agence":
        st.markdown("<div class='main-title'>Diagnostic Pointilleux par Point de Vente</div>", unsafe_allow_html=True)
        
        liste_agences = sorted(df_clean['agence'].dropna().unique().tolist())
        agence_sel = st.selectbox("🎯 Sélectionner l'agence à auditer :", liste_agences)
        
        df_ag = df_clean[df_clean['agence'] == agence_sel]
        vol_ag = len(df_ag)
        
        st.markdown(f"### Performance Locale : <span style='color:#D32F2F;'>{agence_sel}</span> ({vol_ag} répondants)", unsafe_allow_html=True)
        
        if vol_ag < 5:
            st.markdown("<div class='insight-box'>⚠️ <b>Alerte de Représentativité :</b> Le volume d'échantillon pour cette agence est trop faible pour des conclusions statistiques définitives. Se référer principalement aux verbatims qualitatifs ci-dessous.</div>", unsafe_allow_html=True)
            
        nps_ag_counts = df_ag['nps_class'].value_counts()
        nps_local = ((nps_ag_counts.get('Promoteur', 0) - nps_ag_counts.get('Détracteur', 0)) / vol_ag * 100) if vol_ag > 0 else 0
        avg_att_ag = df_ag['score_attente_num'].mean()
        avg_acc_ag = df_ag['score_accueil_num'].mean()
        avg_eff_ag = df_ag['score_effort_num'].mean()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("NPS Local", f"{nps_local:.1f}")
        m2.metric("Note Attente", f"{avg_att_ag:.2f} / 5" if not pd.isna(avg_att_ag) else "N/A")
        m3.metric("Note Accueil", f"{avg_acc_ag:.2f} / 5" if not pd.isna(avg_acc_ag) else "N/A")
        m4.metric("Note Effort", f"{avg_eff_ag:.2f} / 5" if not pd.isna(avg_eff_ag) else "N/A")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔄 Comparatif du Parcours Client : Agence vs Moyenne Nationale")
        
        dims = ["Temps d'attente", "Qualité de l'accueil", "Effort client", "Satisfaction Globale"]
        scores_ag = [
            avg_att_ag if not pd.isna(avg_att_ag) else 1, 
            avg_acc_ag if not pd.isna(avg_acc_ag) else 1, 
            avg_eff_ag if not pd.isna(avg_eff_ag) else 1, 
            df_ag['score_satisfaction_num'].mean() if not pd.isna(df_ag['score_satisfaction_num'].mean()) else 1
        ]
        scores_nat = [
            df_clean['score_attente_num'].mean(), 
            df_clean['score_accueil_num'].mean(), 
            df_clean['score_effort_num'].mean(), 
            df_clean['score_satisfaction_num'].mean()
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=scores_ag, theta=dims, fill='toself', name=f'Agence : {agence_sel}', line_color='#D32F2F'))
        fig_radar.add_trace(go.Scatterpolar(r=scores_nat, theta=dims, fill='toself', name='Moyenne Nationale Réseau', line_color='#1A1A1A'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[1, 5])), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 💬 Analyse Qualitative des Verbatims Clients de l'Agence")
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("<b style='color:#D32F2F;'>Points d'Insatisfaction Soulevés :</b>", unsafe_allow_html=True)
            verbs_neg = df_ag['verbatim_negatif'].dropna().tolist()
            if verbs_neg:
                for vn in verbs_neg[:5]:
                    st.write(f"❌ *\"{vn}\"*")
            else:
                st.write("Aucun point négatif relevé cette semaine.")
        with v2:
            st.markdown("<b style='color:#1A1A1A;'>Pistes d'Amélioration Recommandées :</b>", unsafe_allow_html=True)
            verbs_am = df_ag['verbatim_amelioration'].dropna().tolist()
            if verbs_am:
                for va in verbs_am[:5]:
                    st.write(f"⚙️ *\"{va}\"*")
            else:
                st.write("Aucune suggestion d'amélioration enregistrée.")

    # --------------------------------------------------------------------------
    # PAGE 3 : MODÉLISATION & PRÉDICTIONS S+1
    # --------------------------------------------------------------------------
    elif page == "🔮 Modélisation & Prédictions S+1":
        st.markdown("<div class='main-title'>Moteur Prédictif d'Intelligence Artificielle</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Analyse prédictive de l'impact des dysfonctionnements sur les résultats de la semaine prochaine</div>", unsafe_allow_html=True)
        
        df_ml = df_clean[['score_attente_num', 'score_accueil_num', 'score_effort_num', 'nps_class']].dropna()
        
        if len(df_ml) > 3
