import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BPredicts - NBA Analytics & Predictions",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 BPredicts — Analytics & ML Engine")
st.caption("Plataforma de predicción avanzada, análisis de rendimiento y proyecciones en tiempo real.")
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
# 3. BASE DE DATOS Y CONEXIÓN CON NBA (Con Fallback Automático)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def cargar_base_jugadores():
    # Base de datos ampliada con datos de la temporada para simulación avanzada
    datos = [
        {"jugador": "Luka Dončić", "equipo": "DAL", "minutos": 37.5, "intentos_tiro": 23.2, "efg_pct": 0.565, "tiros_libres": 8.8, "uso_balon": 35.5, "promedio_pts": 33.9, "id": 1629029},
        {"jugador": "Stephen Curry", "equipo": "GSW", "minutos": 32.7, "intentos_tiro": 19.5, "efg_pct": 0.582, "tiros_libres": 5.1, "uso_balon": 30.1, "promedio_pts": 26.4, "id": 201939},
        {"jugador": "Shai Gilgeous-Alexander", "equipo": "OKC", "minutos": 34.2, "intentos_tiro": 19.8, "efg_pct": 0.568, "tiros_libres": 8.7, "uso_balon": 32.8, "promedio_pts": 30.1, "id": 1628983},
        {"jugador": "Jayson Tatum", "equipo": "BOS", "minutos": 35.8, "intentos_tiro": 19.3, "efg_pct": 0.550, "tiros_libres": 6.7, "uso_balon": 29.8, "promedio_pts": 26.9, "id": 1628369},
        {"jugador": "Nikola Jokić", "equipo": "DEN", "minutos": 34.6, "intentos_tiro": 15.7, "efg_pct": 0.630, "tiros_libres": 5.5, "uso_balon": 28.2, "promedio_pts": 26.4, "id": 203999},
        {"jugador": "Anthony Edwards", "equipo": "MIN", "minutos": 35.1, "intentos_tiro": 19.8, "efg_pct": 0.542, "tiros_libres": 6.4, "uso_balon": 32.3, "promedio_pts": 25.9, "id": 1630162},
        {"jugador": "Giannis Antetokounmpo", "equipo": "MIL", "minutos": 35.2, "intentos_tiro": 18.8, "efg_pct": 0.616, "tiros_libres": 10.7, "uso_balon": 33.0, "promedio_pts": 30.4, "id": 203507}
    ]
    return pd.DataFrame(datos)

df_jugadores = cargar_base_jugadores()

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
# 4. BARRA LATERAL (SELECCIÓN Y CONFIGURACIÓN)
# -----------------------------------------------------------------------------
st.sidebar.header("🎯 Configuración de Selección")

jugador_nombre = st.sidebar.selectbox("Selecciona un Jugador:", df_jugadores['jugador'].tolist())
datos_jugador = df_jugadores[df_jugadores['jugador'] == jugador_nombre].iloc[0]

st.sidebar.markdown("---")
st.sidebar.header("📊 Contexto del Partido")

rival_sel = st.sidebar.selectbox("Rival de hoy:", list(defensas_nba.keys()))
rank_defensa = defensas_nba[rival_sel]

sede_opcion = st.sidebar.radio("Sede:", ["Local", "Visitante"])
es_local = 1 if sede_opcion == "Local" else 0

estado_fisico = st.sidebar.radio("Carga Física:", ["Descansado (1+ días)", "Back-to-Back (Jugó ayer)"])
es_b2b = 1 if "Back-to-Back" in estado_fisico else 0

st.sidebar.markdown("---")
st.sidebar.header("🎲 Mercados de Apuestas & Escenarios")
linea_casas = st.sidebar.number_input("Línea de Puntos (Casa de Apuestas):", value=float(round(datos_jugador['promedio_pts'])), step=0.5)

# -----------------------------------------------------------------------------
# 5. PANEL PRINCIPAL - VISTA GENERAL
# -----------------------------------------------------------------------------
col_foto, col_info = st.columns([1, 3])

with col_foto:
    # Foto oficial desde la API de la NBA
    url_foto = f"https://cdn.nba.com/headshots/nba/latest/260x190/{datos_jugador['id']}.png"
    st.image(url_foto, width=180)

with col_info:
    st.subheader(f"{datos_jugador['jugador']} ({datos_jugador['equipo']})")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Promedio Puntos", f"{datos_jugador['promedio_pts']} PTS")
    col_m2.metric("Minutos/Juego", f"{datos_jugador['minutos']} MIN")
    col_m3.metric("% Tiro Efectivo (eFG)", f"{datos_jugador['efg_pct']*100:.1f}%")
    col_m4.metric("Uso de Balón (USG%)", f"{datos_jugador['uso_balon']}%")

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. SIMULADOR "WHAT-IF" Y PREDICCIÓN CON IA
# -----------------------------------------------------------------------------
st.subheader("⚙️ Simulador Contextual (Ajustes de Juego)")

col_adj1, col_adj2, col_adj3 = st.columns(3)

with col_adj1:
    ajuste_min = st.slider("Ajuste de Minutos Proyectados:", -10, 10, 0, help="Simula problemas de faltas o restricciones.")

with col_adj2:
    ajuste_usg = st.slider("Ajuste de % Uso de Balón (Bajas/Lesiones):", -5.0, 8.0, 0.0, step=0.5, help="Si falta otra estrella, el uso de balón aumenta.")

with col_adj3:
    racha_reciente = st.select_slider("Racha de Tiro Reciente:", options=["Frío", "Normal", "Encendido"], value="Normal")

# Modificadores de la simulación
minutos_finales = max(10.0, datos_jugador['minutos'] + ajuste_min)
uso_balon_final = max(10.0, datos_jugador['uso_balon'] + ajuste_usg)

factor_racha = 1.05 if racha_reciente == "Encendido" else (0.95 if racha_reciente == "Frío" else 1.0)
intentos_tiro_final = datos_jugador['intentos_tiro'] * (uso_balon_final / datos_jugador['uso_balon']) * factor_racha

# Predicción mediante el modelo
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
# 7. VISUALIZACIÓN Y ANÁLISIS DE APUESTAS
# -----------------------------------------------------------------------------
col_res1, col_res2 = st.columns([2, 2])

with col_res1:
    st.subheader("📊 Proyección del Modelo")
    
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Puntos Predichos", f"{pts_predichos:.1f} PTS", delta=f"{pts_predichos - datos_jugador['promedio_pts']:.1f} vs Promedio")
    col_p2.metric("Línea Apuesta", f"{linea_casas:.1f} PTS")
    
    # Cálculo del valor de apuesta
    diferencia = pts_predichos - linea_casas
    if diferencia > 1.5:
        st.success(f"🔥 **Sugerencia:** OVER ({linea_casas} Puntos) — El modelo proyecta +{diferencia:.1f} PTS por encima de la línea.")
    elif diferencia < -1.5:
        st.error(f"❄️ **Sugerencia:** UNDER ({linea_casas} Puntos) — El modelo proyecta {abs(diferencia):.1f} PTS por debajo de la línea.")
    else:
        st.warning(f"⚖️ **Sugerencia:** Línea muy ajustada. Margen de diferencia bajo ({diferencia:.1f} PTS).")

with col_res2:
    st.subheader("📈 Distribución de Probabilidad")
    
    # Simulación de curva normal alrededor de la predicción (Desviación estándar típica = 5.5 pts)
    puntos_simulados = np.random.normal(pts_predichos, 5.5, 5000)
    prob_over = (puntos_simulados > linea_casas).mean() * 100
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=puntos_simulados,
        nbinsx=30,
        name="Distribución Proyectada",
        marker_color='#1f77b4',
        opacity=0.75
    ))
    fig.add_vline(x=linea_casas, line_width=3, line_dash="dash", line_color="red", annotation_text=f"Línea ({linea_casas})")
    fig.add_vline(x=pts_predichos, line_width=3, line_color="green", annotation_text=f"Predicción ({pts_predichos:.1f})")
    fig.update_layout(title="Rango Estimado de Anotación", xaxis_title="Puntos", yaxis_title="Frecuencia", height=280, margin=dict(l=20, r=20, t=40, b=20))
    
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"**Probabilidad Estimada de OVER ({linea_casas}):** {prob_over:.1f}%")

# -----------------------------------------------------------------------------
# 8. SECCIÓN FUTURA: PREDICCIÓN DE ANILLO Y PREMIOS
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("🔮 Próximamente: Módulo de Futuros (MVP, Campeón NBA & Mercados Extendidos)"):
    st.write("""
    Estamos preparando la infraestructura para incluir:
    * **Mercados de Apuestas Extendidos:** Tiros libres proyectados, Triples anotados, Rebotes + Asistencias (PRA) y Handicaps de equipo.
    * **Modelos de Premios Individuales:** Probabilidades en vivo para MVP, Defensor del Año (DPOY) y Novato del Año (ROY).
    * **Simulador de Playoffs:** Probabilidad de avanzar de ronda y ganar el Anillo de Campeón NBA mediante simulaciones Monte Carlo.
    """)
