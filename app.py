import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BPredicts - NBA Analytics Platform",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 BPredicts — Live NBA Analytics Platform")
st.caption("Plataforma de predicción analítica de rendimiento e Inteligencia Artificial.")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. CARGA DEL MODELO ENTRENADO
# -----------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_ia_baloncesto.joblib')

try:
    modelo_ia = cargar_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo local: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. CARGA INSTANTÁNEA DE BASE DE DATOS DE JUGADORES
# -----------------------------------------------------------------------------
@st.cache_data
def cargar_jugadores():
    try:
        df = pd.read_csv('nba_players.csv')
        return df.sort_values(by='promedio_pts', ascending=False)
    except Exception as e:
        st.error(f"Error al cargar la base de datos de jugadores: {e}")
        st.stop()

df_jugadores = cargar_jugadores()

defensas_nba = {
    'Boston Celtics (Top 1 Def)': 1,
    'Minnesota Timberwolves (Top 2 Def)': 2,
    'Orlando Magic (Top 5 Def)': 5,
    'Miami Heat (Defensa Media)': 15,
    'Golden State Warriors (Defensa Media)': 16,
    'Washington Wizards (Defensa Débil)': 29,
    'Utah Jazz (Peor Defensa)': 30
}

# -----------------------------------------------------------------------------
# 4. BARRA LATERAL (BÚSQUEDA Y SELECCIÓN GLOBAL)
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Buscador de Jugadores")

lista_nombres = df_jugadores['full_name'].tolist()
jugador_seleccionado = st.sidebar.selectbox("Escribe o selecciona un jugador:", lista_nombres)

datos_jugador = df_jugadores[df_jugadores['full_name'] == jugador_seleccionado].iloc[0]

st.sidebar.markdown("---")
st.sidebar.header("📊 Contexto del Partido")

rival_sel = st.sidebar.selectbox("Rival de hoy:", list(defensas_nba.keys()))
rank_defensa = defensas_nba[rival_sel]

sede_opcion = st.sidebar.radio("Sede:", ["Local", "Visitante"])
es_local = 1 if sede_opcion == "Local" else 0

estado_fisico = st.sidebar.radio("Carga Física:", ["Descansado (1+ días)", "Back-to-Back (Jugó ayer)"])
es_b2b = 1 if "Back-to-Back" in estado_fisico else 0

st.sidebar.markdown("---")
st.sidebar.header("🎲 Mercado de Apuestas")
linea_casas = st.sidebar.number_input("Línea Over/Under Puntos:", value=float(round(datos_jugador['promedio_pts'])), step=0.5)

# -----------------------------------------------------------------------------
# 5. PERFIL Y FICHA TÉCNICA
# -----------------------------------------------------------------------------
col_foto, col_info = st.columns([1, 3])

with col_foto:
    url_foto = f"https://cdn.nba.com/headshots/nba/latest/260x190/{int(datos_jugador['id'])}.png"
    st.image(url_foto, width=180)

with col_info:
    st.subheader(f"{datos_jugador['full_name']} ({datos_jugador['TEAM_ABBREVIATION']})")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Promedio Puntos", f"{datos_jugador['promedio_pts']} PTS")
    col_m2.metric("Minutos/Juego", f"{datos_jugador['minutos']} MIN")
    col_m3.metric("% Tiro Efectivo (eFG)", f"{datos_jugador['efg_pct']*100:.1f}%")
    col_m4.metric("Uso de Balón (USG%)", f"{datos_jugador['uso_balon']}%")

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. SIMULADOR CONTEXTUAL & PREDICCIÓN
# -----------------------------------------------------------------------------
st.subheader("⚙️ Simulador Contextual (Ajustes de Juego)")

col_adj1, col_adj2, col_adj3 = st.columns(3)

with col_adj1:
    ajuste_min = st.slider("Ajuste de Minutos Proyectados:", -10, 10, 0)

with col_adj2:
    ajuste_usg = st.slider("Ajuste de % Uso de Balón:", -5.0, 8.0, 0.0, step=0.5)

with col_adj3:
    racha_reciente = st.select_slider("Racha de Tiro Reciente:", options=["Frío", "Normal", "Encendido"], value="Normal")

minutos_finales = max(10.0, datos_jugador['minutos'] + ajuste_min)
uso_balon_final = max(10.0, datos_jugador['uso_balon'] + ajuste_usg)

factor_racha = 1.05 if racha_reciente == "Encendido" else (0.95 if racha_reciente == "Frío" else 1.0)
intentos_tiro_final = datos_jugador['intentos_tiro'] * (uso_balon_final / max(1.0, datos_jugador['uso_balon'])) * factor_racha

entrada_modelo = pd.DataFrame({
    'minutos': [minutos_finales],
    'intentos_tiro': [intentos_tiro_final],
    'efg_pct': [datos_jugador['efg_pct']],
    'tiros_libres': [datos_jugador['tiros_libres']],
    'uso_balon': [uso_balon_final],
    'es_local': [es_local],
    'defensa_rival_rank': [rank_defensa],
    'es_back_to_back': [es_b2b]
})

pts_predichos = modelo_ia.predict(entrada_modelo)[0] * factor_racha

# -----------------------------------------------------------------------------
# 7. METRICAS Y PROBABILIDAD DE APUESTAS
# -----------------------------------------------------------------------------
col_res1, col_res2 = st.columns([2, 2])

with col_res1:
    st.subheader("📊 Proyección del Modelo")
    
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Puntos Predichos", f"{pts_predichos:.1f} PTS", delta=f"{pts_predichos - datos_jugador['promedio_pts']:.1f} vs Promedio")
    col_p2.metric("Línea Apuesta", f"{linea_casas:.1f} PTS")
    
    diferencia = pts_predichos - linea_casas
    if diferencia > 1.5:
        st.success(f"🔥 **Sugerencia:** OVER ({linea_casas} Puntos) — Ventaja proyectada: +{diferencia:.1f} PTS.")
    elif diferencia < -1.5:
        st.error(f"❄️ **Sugerencia:** UNDER ({linea_casas} Puntos) — Ventaja proyectada: {diferencia:.1f} PTS.")
    else:
        st.warning(f"⚖️ **Sugerencia:** Línea neutral (Margen de diferencia: {diferencia:.1f} PTS).")

with col_res2:
    st.subheader("📈 Distribución de Probabilidad")
    
    puntos_simulados = np.random.normal(pts_predichos, 5.5, 5000)
    prob_over = (puntos_simulados > linea_casas).mean() * 100
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=puntos_simulados,
        nbinsx=30,
        name="Simulación",
        marker_color='#1f77b4',
        opacity=0.75
    ))
    fig.add_vline(x=linea_casas, line_width=3, line_dash="dash", line_color="red", annotation_text=f"Línea ({linea_casas})")
    fig.add_vline(x=pts_predichos, line_width=3, line_color="green", annotation_text=f"Predicción ({pts_predichos:.1f})")
    fig.update_layout(title="Rango de Anotación Estimado", xaxis_title="Puntos", yaxis_title="Frecuencia", height=280, margin=dict(l=20, r=20, t=40, b=20))
    
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"**Probabilidad Estimada de OVER ({linea_casas}):** {prob_over:.1f}%")
