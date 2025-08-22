# Astro Ventas Dashboard

A Streamlit dashboard that analyzes your sales data in relation to lunar phases and planetary alignments. The app reads your sales from an Excel file and automatically computes astronomical data for each sales date, helping you discover possible correlations between sales performance and celestial events.

## Features
- **Upload your sales data** in `database.xlsx` (with columns `Fecha` for date and `Total` for sales).
- **Automatic calculation** of the moon phase and planetary positions (Mercury, Venus, Mars, Jupiter, Saturn) for each sales date using the `ephem` library.
- **Visualize sales and moon phase** over time with interactive charts.
- **Correlation analysis** between sales and moon phase.
- **Display planetary positions** for the top 3 highest and lowest sales days.

## Requirements
* Python 3.8+
* All required packages are listed in `requirements.txt`.

## Installation
1. Clone or download this repository to your computer.
2. Place your sales Excel file as `database.xlsx` in the project folder. The file must have at least these columns:
   - `Fecha`: Date of the sale (any standard date format)
   - `Total`: Sales amount (numeric)
3. (Recommended) Create a virtual environment:
   ```sh
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Make sure your `database.xlsx` is in the project folder and formatted correctly.
2. Run the Streamlit app:
   ```sh
   streamlit run astro_dashboard.py
   ```
3. Open the local URL provided by Streamlit in your browser to view the dashboard.

## How it works
- The app reads your sales data, renames columns to standard names, and parses dates.
- For each date, it calculates the moon phase and planetary positions using the `ephem` library (no external API or astronomical database needed).
- It visualizes sales and moon phase, computes their correlation, and shows planetary alignments for the best and worst sales days.

## Troubleshooting
- If you see an import error for `streamlit`, make sure you have selected the correct Python interpreter in VS Code (the one where you installed the packages).
- If your Excel file is not recognized, check that the columns are named exactly `Fecha` and `Total`.

## License
MIT License

## Author
Your Name Here
