import streamlit as st
import pandas as pd
from utils import load_model, preprocess_data, predict_anomalies
st.set_page_config(page_icon="🤡", page_title="Trabajo Final")
st.image("logob_m_EPG.png", width=200)
st.title("Detección de anomalías para clientes en el sector financiero")
col1, col2 = st.columns([5, 5])

with col1:
    st.subheader("Algoritmo Isolation Forest para Detección de Fraude")
    st.info("""
    El algoritmo **Isolation Forest** es un método de aprendizaje automático utilizado para la detección de anomalías.
    Se basa en el principio de que los puntos de datos anómalos (en este caso, posibles casos de fraude) son más fáciles de 
    "aislar" o separar que los puntos normales. En el contexto financiero, Isolation Forest ayuda a identificar patrones 
    sospechosos en las cuentas de los clientes, lo que puede indicar actividades fraudulentas.
    
    Este modelo divide los datos en árboles de aislamiento y mide la facilidad con la que un punto se aísla. Si un punto es 
    aislado rápidamente, se clasifica como una posible anomalía, permitiendo así una detección temprana de posibles fraudes 
    en las cuentas de los clientes.
    """)
with col2:
    uploaded_model = st.file_uploader(
        "Sube un archivo .pkl para el modelo de análisis", type='pkl',
        key="1"
    )

    if uploaded_model is not None:

        model = load_model(uploaded_model)
        

        data_file = st.file_uploader("Sube un archivo .csv de datos de transacciones", type="csv", key="2")
        
        if data_file is not None:
    
            data = pd.read_csv(data_file)
            st.write("Datos cargados con éxito:", data.head())
            
    
            data_scaled = preprocess_data(data)
            anomalies = predict_anomalies(model, data_scaled)
            
    
            st.write("Resultados de la detección de fraude:")
            data['EsFraude'] = anomalies
            st.write(data[['EsFraude']])
            st.info(f"Total de transacciones anómalas detectadas: {sum(anomalies)}")
    else:
        st.info("Por favor, sube el modelo en formato .pkl.")
