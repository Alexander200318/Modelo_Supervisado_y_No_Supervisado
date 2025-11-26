import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Configuración de la página
st.set_page_config(
    page_title="ML App - AdaBoost & Hierarchical Clustering",
    page_icon="🤖",
    layout="wide"
)

# Título principal
st.title("🤖 Aplicación de Machine Learning")
st.markdown("**AdaBoost** (Supervisado) y **Clustering Jerárquico** (No Supervisado)")
st.divider()

# Cargar datos
@st.cache_data
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['target_name'] = [target_names[i] for i in y]
    return X, y, feature_names, target_names, df

X, y, feature_names, target_names, df = load_data()

# Sidebar para selección de modo
st.sidebar.title("⚙️ Configuración")
mode = st.sidebar.radio(
    "Seleccione el modo:",
    ["🏠 Inicio", "📊 Modo Supervisado (AdaBoost)", "🔍 Modo No Supervisado (Clustering)", "💾 Zona de Exportación"]
)

# ==================== PÁGINA DE INICIO ====================
if mode == "🏠 Inicio":
    st.header("Bienvenido a la Aplicación de Machine Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Modelo Supervisado")
        st.markdown("""
        **AdaBoost Classifier**
        - Algoritmo de ensemble boosting
        - Combina múltiples clasificadores débiles
        - Ideal para clasificación multiclase
        """)
        
    with col2:
        st.subheader("🔍 Modelo No Supervisado")
        st.markdown("""
        **Clustering Jerárquico (Average Linkage)**
        - Agrupa datos sin etiquetas
        - Construye jerarquía de clusters
        - Método: Average Linkage
        """)
    
    st.divider()
    
    st.subheader("📁 Dataset: Iris")
    st.write(f"**Dimensiones**: {X.shape[0]} muestras, {X.shape[1]} características")
    st.write(f"**Clases**: {', '.join(target_names)}")
    
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("""
    ### 📌 Instrucciones:
    1. Use el menú lateral para navegar entre modos
    2. **Modo Supervisado**: Entrene AdaBoost y haga predicciones
    3. **Modo No Supervisado**: Aplique clustering jerárquico
    4. **Zona de Exportación**: Descargue modelos y resultados en JSON
    """)

# ==================== MODO SUPERVISADO ====================
elif mode == "📊 Modo Supervisado (AdaBoost)":
    st.header("📊 Modelo Supervisado: AdaBoost Classifier")
    
    # Parámetros del modelo
    st.sidebar.subheader("Parámetros del Modelo")
    n_estimators = st.sidebar.slider("Número de estimadores", 10, 200, 50, 10)
    learning_rate = st.sidebar.slider("Learning Rate", 0.1, 2.0, 1.0, 0.1)
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Entrenar modelo
    if st.button("🚀 Entrenar Modelo", type="primary"):
        with st.spinner("Entrenando modelo AdaBoost..."):
            # Crear y entrenar modelo
            base_estimator = DecisionTreeClassifier(max_depth=1)
            model = AdaBoostClassifier(
                estimator=base_estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                algorithm='SAMME'
            )
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Guardar en session_state
            st.session_state['supervised_model'] = model
            st.session_state['supervised_metrics'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        st.success("✅ Modelo entrenado exitosamente!")
    
    # Mostrar métricas si el modelo existe
    if 'supervised_model' in st.session_state:
        st.subheader("📈 Métricas de Evaluación")
        
        col1, col2, col3, col4 = st.columns(4)
        metrics = st.session_state['supervised_metrics']
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        
        st.divider()
        
        # Sección de predicción interactiva
        st.subheader("🎯 Predicción Interactiva")
        st.markdown("Ajuste los valores de las características para obtener una predicción:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.slider(
                "Sepal Length (cm)",
                float(X[:, 0].min()),
                float(X[:, 0].max()),
                float(X[:, 0].mean())
            )
            sepal_width = st.slider(
                "Sepal Width (cm)",
                float(X[:, 1].min()),
                float(X[:, 1].max()),
                float(X[:, 1].mean())
            )
        
        with col2:
            petal_length = st.slider(
                "Petal Length (cm)",
                float(X[:, 2].min()),
                float(X[:, 2].max()),
                float(X[:, 2].mean())
            )
            petal_width = st.slider(
                "Petal Width (cm)",
                float(X[:, 3].min()),
                float(X[:, 3].max()),
                float(X[:, 3].mean())
            )
        
        # Realizar predicción
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = st.session_state['supervised_model'].predict(input_data)[0]
        prediction_proba = st.session_state['supervised_model'].predict_proba(input_data)[0]
        
        st.divider()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎯 Resultado")
            st.success(f"**Clase Predicha**: {prediction}")
            st.info(f"**Nombre**: {target_names[prediction]}")
        
        with col2:
            st.markdown("### 📊 Probabilidades")
            proba_df = pd.DataFrame({
                'Clase': target_names,
                'Probabilidad': prediction_proba
            })
            st.dataframe(proba_df, use_container_width=True)
        
        # Guardar predicción actual
        st.session_state['current_prediction'] = {
            'input': input_data[0].tolist(),
            'output_class': int(prediction),
            'output_label': target_names[prediction],
            'probabilities': prediction_proba.tolist()
        }

# ==================== MODO NO SUPERVISADO ====================
elif mode == "🔍 Modo No Supervisado (Clustering)":
    st.header("🔍 Clustering Jerárquico (Average Linkage)")
    
    # Parámetros del modelo
    st.sidebar.subheader("Parámetros del Clustering")
    n_clusters = st.sidebar.slider("Número de Clusters", 2, 6, 3)
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if st.button("🚀 Aplicar Clustering", type="primary"):
        with st.spinner("Aplicando clustering jerárquico..."):
            # Crear y aplicar modelo
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='average'
            )
            cluster_labels = model.fit_predict(X_scaled)
            
            # Calcular métricas
            silhouette = silhouette_score(X_scaled, cluster_labels)
            davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
            
            # Guardar en session_state
            st.session_state['unsupervised_model'] = model
            st.session_state['cluster_labels'] = cluster_labels
            st.session_state['unsupervised_metrics'] = {
                'silhouette_score': silhouette,
                'davies_bouldin': davies_bouldin
            }
            st.session_state['n_clusters'] = n_clusters
            
        st.success("✅ Clustering aplicado exitosamente!")
    
    # Mostrar resultados si el modelo existe
    if 'unsupervised_model' in st.session_state:
        st.subheader("📈 Métricas de Calidad")
        
        col1, col2 = st.columns(2)
        metrics = st.session_state['unsupervised_metrics']
        
        with col1:
            st.metric(
                "Silhouette Score",
                f"{metrics['silhouette_score']:.4f}",
                help="Valores cercanos a 1 indican mejor separación"
            )
        with col2:
            st.metric(
                "Davies-Bouldin Score",
                f"{metrics['davies_bouldin']:.4f}",
                help="Valores más bajos indican mejor clustering"
            )
        
        st.divider()
        
        # Visualización de clusters
        st.subheader("📊 Visualización de Clusters")
        
        cluster_labels = st.session_state['cluster_labels']
        
        # Crear visualización 2D
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Sepal
        scatter1 = axes[0].scatter(
            X[:, 0], X[:, 1],
            c=cluster_labels,
            cmap='viridis',
            s=100,
            alpha=0.6,
            edgecolors='black'
        )
        axes[0].set_xlabel('Sepal Length (cm)')
        axes[0].set_ylabel('Sepal Width (cm)')
        axes[0].set_title('Clusters - Características Sepal')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Plot 2: Petal
        scatter2 = axes[1].scatter(
            X[:, 2], X[:, 3],
            c=cluster_labels,
            cmap='viridis',
            s=100,
            alpha=0.6,
            edgecolors='black'
        )
        axes[1].set_xlabel('Petal Length (cm)')
        axes[1].set_ylabel('Petal Width (cm)')
        axes[1].set_title('Clusters - Características Petal')
        plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Dendrograma
        st.subheader("🌳 Dendrograma")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        linkage_matrix = linkage(X_scaled, method='average')
        dendrogram(linkage_matrix, ax=ax)
        ax.set_title('Dendrograma - Clustering Jerárquico (Average Linkage)')
        ax.set_xlabel('Índice de Muestra')
        ax.set_ylabel('Distancia')
        st.pyplot(fig)
        
        # Distribución de clusters
        st.subheader("📊 Distribución de Clusters")
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            cluster_counts.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Número de Muestras')
            ax.set_title('Muestras por Cluster')
            plt.xticks(rotation=0)
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Conteo por Cluster:**")
            for cluster, count in cluster_counts.items():
                st.write(f"Cluster {cluster}: {count} muestras")

# ==================== ZONA DE EXPORTACIÓN ====================
elif mode == "💾 Zona de Exportación":
    st.header("💾 Zona de Exportación (Dev Tools)")
    st.markdown("Descargue los modelos entrenados y sus resultados en formato JSON y PKL.")
    
    st.divider()
    
    # Exportación Modelo Supervisado
    st.subheader("📊 Modelo Supervisado - AdaBoost")
    
    if 'supervised_model' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON
            json_data = {
                "model_type": "Supervised",
                "model_name": "AdaBoost",
                "parameters": {
                    "n_estimators": st.session_state.get('supervised_model').n_estimators,
                    "learning_rate": st.session_state.get('supervised_model').learning_rate,
                    "algorithm": "SAMME"
                },
                "metrics": st.session_state['supervised_metrics'],
                "current_prediction": st.session_state.get('current_prediction', {})
            }
            
            json_str = json.dumps(json_data, indent=2)
            
            st.download_button(
                label="📥 Descargar JSON",
                data=json_str,
                file_name="adaboost_results.json",
                mime="application/json"
            )
        
        with col2:
            # PKL
            model_pkl = pickle.dumps(st.session_state['supervised_model'])
            
            st.download_button(
                label="📥 Descargar Modelo (.pkl)",
                data=model_pkl,
                file_name="adaboost_model.pkl",
                mime="application/octet-stream"
            )
        
        st.success("✅ Modelo supervisado disponible para descarga")
        
        with st.expander("👁️ Preview del JSON"):
            st.json(json_data)
    else:
        st.warning("⚠️ Primero debe entrenar el modelo supervisado")
    
    st.divider()
    
    # Exportación Modelo No Supervisado
    st.subheader("🔍 Modelo No Supervisado - Clustering Jerárquico")
    
    if 'unsupervised_model' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON
            json_data = {
                "model_type": "Unsupervised",
                "algorithm": "Hierarchical Clustering (Average Linkage)",
                "parameters": {
                    "n_clusters": st.session_state['n_clusters'],
                    "linkage": "average"
                },
                "metrics": st.session_state['unsupervised_metrics'],
                "cluster_labels": st.session_state['cluster_labels'].tolist()
            }
            
            json_str = json.dumps(json_data, indent=2)
            
            st.download_button(
                label="📥 Descargar JSON",
                data=json_str,
                file_name="hierarchical_clustering_results.json",
                mime="application/json"
            )
        
        with col2:
            # PKL
            model_pkl = pickle.dumps(st.session_state['unsupervised_model'])
            
            st.download_button(
                label="📥 Descargar Modelo (.pkl)",
                data=model_pkl,
                file_name="hierarchical_clustering_model.pkl",
                mime="application/octet-stream"
            )
        
        st.success("✅ Modelo no supervisado disponible para descarga")
        
        with st.expander("👁️ Preview del JSON"):
            st.json(json_data)
    else:
        st.warning("⚠️ Primero debe aplicar el clustering")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🤖 Aplicación de Machine Learning | AdaBoost & Clustering Jerárquico</p>
    <p>Desarrollado con Streamlit 🎈</p>
</div>
""", unsafe_allow_html=True)