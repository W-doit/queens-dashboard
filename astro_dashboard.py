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
    st.title('Astro Ventas Dashboard')
    st.write('Sales with Moon Phase, Moon Sign, Venus Sign, and Mercury Sign (grouped by day):', df_daily[['date', 'sales', 'moon_phase', 'moon_phase_name', 'moon_sign', 'venus_sign', 'mercury_sign']])


    # Bar chart: sales by day of the week
    df_daily['weekday'] = df_daily['date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sales_by_weekday = df_daily.groupby('weekday')['sales'].mean().reindex(weekday_order)
    fig_weekday = px.bar(sales_by_weekday, x=sales_by_weekday.index, y=sales_by_weekday.values,
        labels={'x': 'Day of Week', 'y': 'Average Sales'},
        title='Average Sales by Day of the Week', color=sales_by_weekday.values, color_continuous_scale='Blues')
    st.plotly_chart(fig_weekday, use_container_width=True)

    # Bar chart: sales by moon sign
    sales_by_sign = df_daily.groupby('moon_sign')['sales'].mean().reindex([
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ])
    fig_sign = px.bar(sales_by_sign, x=sales_by_sign.index, y=sales_by_sign.values,
        labels={'x': 'Moon Sign', 'y': 'Average Sales'},
        title='Average Sales by Moon Sign', color=sales_by_sign.values, color_continuous_scale='Viridis')
    st.plotly_chart(fig_sign, use_container_width=True)

    # Bar chart: sales by Venus sign
    sales_by_venus = df_daily.groupby('venus_sign')['sales'].mean().reindex([
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ])
    fig_venus = px.bar(sales_by_venus, x=sales_by_venus.index, y=sales_by_venus.values,
        labels={'x': 'Venus Sign', 'y': 'Average Sales'},
        title='Average Sales by Venus Sign', color=sales_by_venus.values, color_continuous_scale='Plasma')
    st.plotly_chart(fig_venus, use_container_width=True)

    # Bar chart: sales by Mercury sign
    sales_by_mercury = df_daily.groupby('mercury_sign')['sales'].mean().reindex([
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ])
    fig_mercury = px.bar(sales_by_mercury, x=sales_by_mercury.index, y=sales_by_mercury.values,
        labels={'x': 'Mercury Sign', 'y': 'Average Sales'},
        title='Average Sales by Mercury Sign', color=sales_by_mercury.values, color_continuous_scale='Cividis')
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
    st.subheader('Planetary Positions on High Sales Days')
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
    st.subheader('Planetary Positions on Low Sales Days')
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
    st.title('Sales Prediction Model')
    st.write('Train and test a model to predict sales using planetary and astrological features.')
    st.write('Available features:', list(df_daily.columns))

    import astro_model
    if st.button('Train Model'):
        with st.spinner('Training model...'):
            model, X_test, y_test, y_pred = astro_model.train_sales_model(df_daily)
        st.success('Model trained!')
        st.write('Test set predictions:')
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)
        st.dataframe(results)
        st.write('MAE:', (abs(y_test - y_pred)).mean())
        st.write('RMSE:', ((y_test - y_pred) ** 2).mean() ** 0.5)

        # Show feature importances
        importances = model.feature_importances_
        feature_names = X_test.columns
        importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importances_df = importances_df.sort_values('Importance', ascending=False)
        st.subheader('Feature Importances')
        st.dataframe(importances_df)
        st.bar_chart(importances_df.set_index('Feature'))

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
        page = st.sidebar.selectbox('Select page', ['Dashboard', 'Model'])
        if page == 'Dashboard':
            dashboard_page(df, df_daily)
        elif page == 'Model':
            model_page(df, df_daily)
    else:
        st.error('The Excel file must have columns: Fecha, Total')

if __name__ == '__main__':
    main()
