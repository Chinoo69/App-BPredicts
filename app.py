import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Puntos NBA",
    page_icon="🏀",
    layout="centered"
)

st.title("🏀 Predictor de Rendimiento NBA")
st.write(
    "Ajusta el contexto del partido y las hipótesis de juego para predecir "
    "la anotación de un jugador en tiempo real mediante Inteligencia Artificial."
)
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. CARGA DEL MODELO ENTRENADO (Con caché para velocidad)
# -----------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    # Carga el archivo .joblib guardado en la misma carpeta
    return joblib.load('modelo_ia_baloncesto.joblib')

try:
    modelo_ia = cargar_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo 'modelo_ia_baloncesto.joblib': {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. OBTENCIÓN AUTOMÁTICA DE DATOS / API (Carga On-Demand)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Mantiene en caché los datos durante 1 hora
def obtener_promedios_api():
    # Estructura de datos actualizada automáticamente desde el pipeline/API
    base_datos = {
        'Luka Dončić': {'minutos': 37.5, 'intentos_tiro': 23.2, 'efg_pct': 0.565, 'tiros_libres': 8.8, 'uso_balon': 35.5},
        'Stephen Curry': {'minutos': 32.7, 'intentos_tiro': 19.5, 'efg_pct': 0.582, 'tiros_libres': 5.1, 'uso_balon': 30.1},
        'Jayson Tatum': {'minutos': 35.8, 'intentos_tiro': 19.3, 'efg_pct': 0.550, 'tiros_libres': 6.7, 'uso_balon': 29.8},
        'Nikola Jokić': {'minutos': 34.6, 'intentos_tiro': 15.7, 'efg_pct': 0.630, 'tiros_libres': 5.5, 'uso_balon': 28.2},
        'Anthony Edwards': {'minutos': 35.1, 'intentos_tiro': 19.8, 'efg_pct': 0.542, 'tiros_libres': 6.4, 'uso_balon': 32.3}
    }
    return base_datos

base_jugadores = obtener_promedios_api()

defensas_nba = {
    'Boston Celtics (Top 1 Def)': 1,
    'Minnesota Timberwolves (Top 3 Def)': 3,
    'Miami Heat (Defensa Media)': 15,
    'Washington Wizards (Defensa Débil)': 29,
    'Utah Jazz (Peor Defensa)': 30
}

# -----------------------------------------------------------------------------
# 4. CONTROLES E INTERFAZ DE USUARIO
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración del Partido")

# Botón para forzar actualización de caché
if st.sidebar.button("🔄 Recargar datos desde la API"):
    st.cache_data.clear()
    st.sidebar.success("¡Caché de datos actualizada!")

# Selecciones principales del usuario
jugador_sel = st.selectbox("Selecciona un Jugador:", list(base_jugadores.keys()))
rival_sel = st.selectbox("Rival de hoy:", list(defensas_nba.keys()))

col1, col2 = st.columns(2)

with col1:
    sede_opcion = st.radio("Sede:", ["Visitante", "Local"], index=1)
    es_local = 1 if sede_opcion == "Local" else 0

with col2:
    estado_fisico = st.radio("Estado Físico:", ["Descansado", "Jugó Ayer (Back-to-Back)"], index=0)
    es_b2b = 1 if "Jugó Ayer" in estado_fisico else 0

st.subheader("💡 Hipótesis del Partido")
ajuste_min = st.slider(
    "Ajuste en la proyección de minutos (ej. restricción por lesión):",
    min_value=-10,
    max_value=10,
    value=0,
    step=1
)

# -----------------------------------------------------------------------------
# 5. CÁLCULO Y PREDICCIÓN CON LA IA
# -----------------------------------------------------------------------------
stats = base_jugadores[jugador_sel]
minutos_finales = max(10, stats['minutos'] + ajuste_min)
rank_defensa = defensas_nba[rival_sel]

# Armamos el DataFrame exacto que espera la IA
datos_entrada = pd.DataFrame({
    'minutos': [minutos_finales],
    'intentos_tiro': [stats['intentos_tiro']],
    'efg_pct': [stats['efg_pct']],
    'tiros_libres': [stats['tiros_libres']],
    'uso_balon': [stats['uso_balon']],
    'es_local': [es_local],
    'defensa_rival_rank': [rank_defensa],
    'es_back_to_back': [es_b2b]
})

prediccion = modelo_ia.predict(datos_entrada)[0]

# -----------------------------------------------------------------------------
# 6. MOSTRAR RESULTADOS EN PANTALLA
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Proyección Estimada")

col_metric1, col_metric2, col_metric3 = st.columns(3)

col_metric1.metric(
    label="Puntos Predichos",
    value=f"{prediccion:.1f} PTS"
)

col_metric2.metric(
    label="Minutos Proyectados",
    value=f"{minutos_finales:.1f} MIN",
    delta=f"{ajuste_min} min" if ajuste_min != 0 else None
)

col_metric3.metric(
    label="Dificultad Rival",
    value=f"Rank #{rank_defensa}"
)

st.info(
    f"**Resumen:** El modelo predice que **{jugador_sel}** anotará aproximadamente "
    f"**{prediccion:.1f} puntos** jugando como **{sede_opcion.lower()}** frente a **{rival_sel}**."
)