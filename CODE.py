# ==============================================================================
# 3. BARRE LATÉRALE (SIDEBAR) & NAVIGATION
# ==============================================================================
with st.sidebar:
    # --- ZONE LOGO AFRILAND FIRST BANK ---
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    logo_placeholder = "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/master/svgs/solid/building-columns.svg" 
    st.image(logo_placeholder, width=80)
    st.markdown("<h3 style='color: #D32F2F; margin-top:5px;'>AFRILAND FIRST BANK</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: gray; font-size:12px;'>BI Satisfaction v2.0</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # --- MODULE D'INJECTION DE DOCUMENT DYNAMIQUE ---
    st.subheader("📁 Alimentation des données")
    uploaded_file = st.file_uploader(
        "Injecter le dernier fichier d'enquête de satisfaction (.csv)", 
        type=["csv"], 
        help="Permet de mettre à jour le dashboard instantanément avec le fichier de la nouvelle semaine."
    )
    
    # Initialisation de la variable de contrôle des données
    df_clean = None
    
    # Vérification stricte de la présence d'un fichier injecté
    if uploaded_file is not None:
        try:
            df_clean = load_and_preprocess(uploaded_file)
            st.success("Nouveau fichier hebdomadaire chargé avec succès !")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
    else:
        # Tente de charger le fichier local s'il existe, sinon invite l'utilisateur à uploader
        import os
        fichier_par_defaut = "Exemple_donnée.xlsx - Feuil1.csv"
        if os.path.exists(fichier_par_defaut):
            df_clean = load_and_preprocess(fichier_par_defaut)
            st.info("Utilisation du fichier historique local.")
        else:
            st.warning("⚠️ En attente de l'injection d'un fichier de données pour afficher les analyses.")
        
    st.markdown("---")
    
    # --- NAVIGATION PRINCIPALE ---
    st.subheader("🗺️ Menu de Navigation")
    page = st.radio(
        "Aller vers :",
        ["🏢 Vue Globale (Réseau National)", "📍 Analyse par Agence & Parcours", "📈 Prédictions & Tendances de la Semaine Prochaine", "📝 Rapport Éditorial & Export"]
    )
    
    st.markdown("---")
    st.markdown("<p style='text-align:center; color:gray; font-size:11px;'>Conçu pour la Direction des Études et de l'Expérience Client</p>", unsafe_allow_html=True)

# ==============================================================================
# ENCAPSULATION DE L'AFFICHAGE POUR ÉVITER LES CRASHS SANS DONNÉES
# ==============================================================================
if df_clean is None:
    st.markdown("<div class='main-title'>Suivi de la Satisfaction Clientèle</div>", unsafe_allow_html=True)
    st.info("👋 Bienvenue sur le Dashboard BI d'Afriland First Bank. Veuillez charger un fichier de données `.csv` dans le menu latéral gauche pour activer les analyses automatiques et les modèles prédictifs.")
else:
    # Mettre ici tout le reste de votre code existant pour l'affichage des pages :
    # (if page == "🏢 Vue Globale (Réseau National)": ..., elif page == "📍 Analyse par Agence & Parcours": ... etc.)
