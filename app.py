import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página (Título y Icono)
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# --- 1. CARGA DE MODELOS (Usando Joblib) ---
@st.cache_resource
def load_models():
    try:
        # Cargamos los archivos .joblib (Asegúrate de haberlos subido)
        model = joblib.load('ensemble_model.joblib')
        preprocessor = joblib.load('preprocessor.joblib')
        return model, preprocessor
    except FileNotFoundError as e:
        st.error(f"⚠️ Error Crítico: No se encuentran los archivos del modelo. \n\nDetalle: {e}")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Error al cargar los modelos: {e}")
        return None, None

# Título principal
st.title("🚢 Predicción de Supervivencia del Titanic")
st.markdown("---")
st.write("""
**Bienvenido.** Introduce los datos del pasajero en el menú de la izquierda para predecir si sobreviviría o no al naufragio.
Este sistema utiliza un modelo de *Machine Learning* optimizado (Ensemble).
""")

# Cargar modelos
model, preprocessor = load_models()

if model is not None and preprocessor is not None:

    # --- 2. BARRA LATERAL (Inputs del usuario) ---
    st.sidebar.header("📝 Datos del Pasajero")

    def user_input_features():
        # --- INPUTS EXISTENTES ---
        pclass = st.sidebar.selectbox("Clase", [1, 2, 3], format_func=lambda x: f"Clase {x}")
        sex = st.sidebar.selectbox("Sexo", ["male", "female"], format_func=lambda x: "Hombre" if x == "male" else "Mujer")
        age = st.sidebar.slider("Edad", 0, 80, 30)
        sibsp = st.sidebar.number_input("Hermanos/Cónyuges", 0, 8, 0)
        parch = st.sidebar.number_input("Padres/Hijos", 0, 6, 0)
        fare = st.sidebar.number_input("Tarifa ($)", 0.0, 512.0, 32.0)
        embarked = st.sidebar.selectbox("Embarque", ["S", "C", "Q"], 
                                      format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])

        data = {
            'Pclass': pclass, 
            'Sex': sex, 
            'Age': age, 
            'SibSp': sibsp, 
            'Parch': parch, 
            'Fare': fare, 
            'Embarked': embarked
        }
        
        features = pd.DataFrame(data, index=[0])

        # --- 🚨 CORRECCIÓN: FEATURE ENGINEERING ---
        # Calculamos las columnas que le faltan al modelo
        
        # 1. FamilySize: Suma de hermanos + padres + el pasajero (1)
        features['FamilySize'] = features['SibSp'] + features['Parch'] + 1
        
        # 2. IsAlone: 1 si FamilySize es 1, si no 0
        features['IsAlone'] = 1 # Asumimos que viaja solo
        features.loc[features['FamilySize'] > 1, 'IsAlone'] = 0 # Si tiene familia, ponemos 0
        
        return features

    # Obtener el DataFrame del input
    input_df = user_input_features()

    # Mostrar los datos ingresados al usuario
    st.subheader("📋 Resumen de Datos Ingresados")
    st.dataframe(input_df)

    # --- 3. PREDICCIÓN ---
    # Botón para ejecutar
    if st.button("🚀 Predecir Supervivencia", type="primary"):
        try:
            with st.spinner('Procesando datos...'):
                # A. Preprocesar los datos (Escalado, OneHotEncoding, etc.)
                processed_data = preprocessor.transform(input_df)
                
                # B. Realizar la predicción
                prediction = model.predict(processed_data)
                prediction_proba = model.predict_proba(processed_data)

                # C. Mostrar resultados
                st.markdown("---")
                st.subheader("📊 Resultado de la Predicción")
                
                survived = prediction[0] == 1
                prob_survival = prediction_proba[0][1]
                prob_death = prediction_proba[0][0]

                if survived:
                    st.success(f"🎉 **¡SOBREVIVE!**")
                    st.write(f"El modelo estima una probabilidad de supervivencia del **{prob_survival:.2%}**.")
                    st.balloons()
                else:
                    st.error(f"💀 **NO SOBREVIVE**")
                    st.write(f"El modelo estima una probabilidad de no sobrevivir del **{prob_death:.2%}**.")
        
        except Exception as e:
            st.error(f"❌ Ocurrió un error al procesar los datos: {e}")
            st.info("Nota técnica: Verifica que las columnas de entrada coincidan exactamente con las usadas en el entrenamiento.")

else:
    st.warning("⚠️ Esperando la carga de modelos... Por favor sube los archivos .joblib")