import streamlit as st
import pandas as pd

import ephem
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

# --- Helper functions ---
def get_moon_phase(date):
    obs_date = ephem.Date(date)
    moon = ephem.Moon(obs_date)
    phase = moon.phase  # 0-29.53
    if phase < 1.84566 or phase > 27.68493:
        phase_name = "New Moon"
    elif phase < 5.53699:
        phase_name = "Waxing Crescent"
    elif phase < 9.22831:
        phase_name = "First Quarter"
    elif phase < 12.91963:
        phase_name = "Waxing Gibbous"
    elif phase < 16.61096:
        phase_name = "Full Moon"
    elif phase < 20.30228:
        phase_name = "Waning Gibbous"
    elif phase < 23.99361:
        phase_name = "Last Quarter"
    else:
        phase_name = "Waning Crescent"
    return phase, phase_name

def get_moon_sign(date):
    obs_date = ephem.Date(date)
    moon = ephem.Moon(obs_date)
    lon = float(ephem.Ecliptic(moon).lon) * 180.0 / 3.141592653589793
    signs = [
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ]
    sign_index = int(lon // 30) % 12
    return signs[sign_index]

def get_venus_sign(date):
    obs = ephem.Observer()
    obs.date = date
    venus = ephem.Venus(obs)
    lon = float(ephem.Ecliptic(venus).lon) * 180.0 / 3.141592653589793
    signs = [
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ]
    sign_index = int(lon // 30) % 12
    return signs[sign_index]

def get_mercury_sign(date):
    obs = ephem.Observer()
    obs.date = date
    mercury = ephem.Mercury(obs)
    lon = float(ephem.Ecliptic(mercury).lon) * 180.0 / 3.141592653589793
    signs = [
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ]
    sign_index = int(lon // 30) % 12
    return signs[sign_index]

def get_planetary_positions(date):
    obs = ephem.Observer()
    obs.date = date
    planets = ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']
    positions = {}
    for planet in planets:
        body = getattr(ephem, planet)()
        body.compute(obs)
        positions[planet] = body.ra
    return positions

def get_planet_sign(lon):
    deg = float(lon) * 180.0 / 3.141592653589793
    signs = [
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ]
    sign_index = int(deg // 30) % 12
    return signs[sign_index]


def dashboard_page(df, df_daily):
    st.title('Panel Astro Ventas')
    st.write('Ventas con Fase Lunar, Signo Lunar, Signo de Venus y Signo de Mercurio (agrupadas por día):', df_daily[['date', 'sales', 'moon_phase', 'moon_phase_name', 'moon_sign', 'venus_sign', 'mercury_sign']])



    # Bar chart: sales by day of the week
    df_daily['weekday'] = df_daily['date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sales_by_weekday = df_daily.groupby('weekday')['sales'].mean().reindex(weekday_order)
    fig_weekday = px.bar(sales_by_weekday, x=sales_by_weekday.index, y=sales_by_weekday.values,
        labels={'x': 'Día de la semana', 'y': 'Ventas promedio'},
        title='Ventas promedio por día de la semana', color=sales_by_weekday.values, color_continuous_scale='Blues')
    st.plotly_chart(fig_weekday, use_container_width=True)

    # Bar chart: sales per 30-minute interval
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    # Bin minutes into 0 or 30
    df['minute_bin'] = (df['minute'] // 30) * 30
    df['time_bin'] = df['hour'].astype(str).str.zfill(2) + ':' + df['minute_bin'].astype(str).str.zfill(2)
    sales_by_timebin = df.groupby('time_bin')['sales'].mean()
    # Sort bins chronologically
    sales_by_timebin = sales_by_timebin.sort_index()
    fig_timebin = px.bar(sales_by_timebin, x=sales_by_timebin.index, y=sales_by_timebin.values,
        labels={'x': 'Intervalo de tiempo (cada 30 min)', 'y': 'Ventas promedio'},
        title='Ventas promedio por intervalo de 30 minutos', color=sales_by_timebin.values, color_continuous_scale='Oranges')
    st.plotly_chart(fig_timebin, use_container_width=True)

    # Bar chart: sales by moon sign
    sales_by_sign = df_daily.groupby('moon_sign')['sales'].mean().reindex([
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ])
    fig_sign = px.bar(sales_by_sign, x=sales_by_sign.index, y=sales_by_sign.values,
        labels={'x': 'Signo lunar', 'y': 'Ventas promedio'},
        title='Ventas promedio por signo lunar', color=sales_by_sign.values, color_continuous_scale='Viridis')
    st.plotly_chart(fig_sign, use_container_width=True)

    # Bar chart: sales by Venus sign
    sales_by_venus = df_daily.groupby('venus_sign')['sales'].mean().reindex([
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ])
    fig_venus = px.bar(sales_by_venus, x=sales_by_venus.index, y=sales_by_venus.values,
        labels={'x': 'Signo de Venus', 'y': 'Ventas promedio'},
        title='Ventas promedio por signo de Venus', color=sales_by_venus.values, color_continuous_scale='Plasma')
    st.plotly_chart(fig_venus, use_container_width=True)

    # Bar chart: sales by Mercury sign
    sales_by_mercury = df_daily.groupby('mercury_sign')['sales'].mean().reindex([
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ])
    fig_mercury = px.bar(sales_by_mercury, x=sales_by_mercury.index, y=sales_by_mercury.values,
        labels={'x': 'Signo de Mercurio', 'y': 'Ventas promedio'},
        title='Ventas promedio por signo de Mercurio', color=sales_by_mercury.values, color_continuous_scale='Cividis')
    st.plotly_chart(fig_mercury, use_container_width=True)

    # Interactive chart with hover info using Plotly
    fig = go.Figure()
    # Sales line in green
    fig.add_trace(go.Scatter(
        x=df_daily['date'],
        y=df_daily['sales'],
        mode='lines+markers',
        name='Sales',
        line=dict(color='green'),
        marker=dict(size=6)
    ))

    # Show planetary positions for high/low sales days
    high_sales = df_daily.sort_values('sales', ascending=False).head(3)
    low_sales = df_daily.sort_values('sales').head(3)
    st.subheader('Posiciones planetarias en días de mayores ventas')
    for _, row in high_sales.iterrows():
        positions = get_planetary_positions(row["date"])
        planet_signs = {}
        for planet in positions:
            obs = ephem.Observer()
            obs.date = row["date"]
            body = getattr(ephem, planet)(obs)
            lon = ephem.Ecliptic(body).lon
            sign = get_planet_sign(lon)
            planet_signs[planet] = f"{positions[planet]} ({sign})"
        st.write(f"{row['date'].date()}: {planet_signs}")
    st.subheader('Posiciones planetarias en días de menores ventas')
    for _, row in low_sales.iterrows():
        positions = get_planetary_positions(row["date"])
        planet_signs = {}
        for planet in positions:
            obs = ephem.Observer()
            obs.date = row["date"]
            body = getattr(ephem, planet)(obs)
            lon = ephem.Ecliptic(body).lon
            sign = get_planet_sign(lon)
            planet_signs[planet] = f"{positions[planet]} ({sign})"
        st.write(f"{row['date'].date()}: {planet_signs}")

def model_page(df, df_daily):
    st.title('Modelo de Predicción de Ventas')
    st.write('Entrena y prueba un modelo para predecir ventas usando características planetarias y astrológicas.')
    st.write('Características disponibles:', list(df_daily.columns))

    import astro_model
    if st.button('Train Model'):
        with st.spinner('Training model...'):
            model, X_test, y_test, y_pred = astro_model.train_sales_model(df_daily)
        st.success('¡Modelo entrenado!')
        st.write('Predicciones en el conjunto de prueba:')
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)
        st.dataframe(results)
        st.write('MAE (Error Absoluto Medio):', (abs(y_test - y_pred)).mean())
        st.write('RMSE (Raíz del Error Cuadrático Medio):', ((y_test - y_pred) ** 2).mean() ** 0.5)

        # Mostrar importancia de las características
        importances = model.feature_importances_
        feature_names = X_test.columns
        importances_df = pd.DataFrame({'Característica': feature_names, 'Importancia': importances})
        importances_df = importances_df.sort_values('Importancia', ascending=False)
        st.subheader('Importancia de las características')
        st.dataframe(importances_df)
        st.bar_chart(importances_df.set_index('Característica'))

def main():
    # 1. Read the database
    df = pd.read_excel('database.xlsx')
    # 2. Convert columns: Fecha to date, Total to sales
    if 'Fecha' in df.columns and 'Total' in df.columns:
        df = df.rename(columns={"Fecha": "date", "Total": "sales"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "sales"])
        # 3. Group by day (ignore time)
        df["date_only"] = df["date"].dt.date
        df_daily = df.groupby("date_only", as_index=False)["sales"].sum()
        df_daily["date"] = pd.to_datetime(df_daily["date_only"])
        # 4. Add moon phase columns
        df_daily[['moon_phase', 'moon_phase_name']] = df_daily['date'].apply(lambda d: pd.Series(get_moon_phase(d)))
        # Add moon sign column
        df_daily['moon_sign'] = df_daily['date'].apply(get_moon_sign)
        # Add Venus sign column
        df_daily['venus_sign'] = df_daily['date'].apply(get_venus_sign)
        # Add Mercury sign column
        df_daily['mercury_sign'] = df_daily['date'].apply(get_mercury_sign)

        # Sidebar navigation
        page = st.sidebar.selectbox('Selecciona página', ['Panel', 'Modelo'])
        if page == 'Panel':
            dashboard_page(df, df_daily)
        elif page == 'Modelo':
            model_page(df, df_daily)
    else:
        st.error('El archivo Excel debe tener las columnas: Fecha, Total')

if __name__ == '__main__':
    main()
