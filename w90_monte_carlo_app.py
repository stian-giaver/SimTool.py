"""
W90 Monte Carlo – Streamlit App
Converted from w90_monte_carlo_v6.ipynb
"""

import streamlit as st
import os, io, math, time, uuid, glob, zipfile, shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

# ── Optional heavy imports (installed at startup) ─────────────────────────
try:
    import cdsapi
except ImportError:
    cdsapi = None

try:
    from great_tables import GT
    HAS_GT = True
except ImportError:
    HAS_GT = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="W90 Monte Carlo Simulation",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 W90 Monte Carlo – Offshore Wind Installation Simulator")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 – INPUTS
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("⚙️  1. Project Inputs", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Project")
        N_WTG        = st.number_input("Number of turbines (N_WTG)", min_value=1, value=50, step=1)
        Simulations  = st.number_input("Simulations", min_value=10, max_value=5000, value=500, step=50)
        Sim_start_date = st.text_input("Sim start date (D-M-YYYY)", value="1-7-2027")
        API_key      = st.text_input("ERA5 API key", value="", type="password",
                                     help="Your Copernicus CDS API key")

    with col2:
        st.subheader("Project Location (ERA5 0.25° grid)")
        lat       = st.number_input("Latitude",        value=35.25,  step=0.25, format="%.2f")
        lon       = st.number_input("Longitude",       value=140.75, step=0.25, format="%.2f")
        lat_barge = st.number_input("Barge Latitude",  value=56.50,  step=0.25, format="%.2f")
        lon_barge = st.number_input("Barge Longitude", value=-2.50,  step=0.25, format="%.2f")

    with col3:
        st.subheader("Vessel & Distance")
        dock_distance  = st.number_input("Dock transit distance (Nm)",  value=185, step=5)
        barge_distance = st.number_input("Barge transit distance (Nm)", value=115, step=5)
        W90_carry_cap  = st.number_input("W90 carry capacity",  value=8, step=1)
        JUV_carry_cap  = st.number_input("JUV carry capacity",  value=6, step=1)
        FIV_carry_cap  = st.number_input("FIV carry capacity",  value=5, step=1)

    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Transit Speeds (kn)")
        W90_transit_speed  = st.number_input("W90",   value=10, step=1)
        Comp_transit_speed = st.number_input("Competitor (JUV/FIV)", value=10, step=1)
        Fiv_transit_speed  = st.number_input("FIV",   value=10, step=1)

    with col5:
        st.subheader("Operations")
        max_wave          = st.number_input("Max wave height (m)", value=6, step=1)
        intermediate_days = st.number_input("Intermediate days (MP → WTG)", value=14, step=1)
        W90_sim  = st.checkbox("Include W90",   value=True)
        barge    = st.checkbox("Include Barge", value=True)
        JUV_sim  = st.checkbox("Include JUV",   value=True)
        FIV_sim  = st.checkbox("Include FIV",   value=True)

    st.subheader("Excel Sheet Names")
    c1, c2 = st.columns(2)
    with c1:
        W90_MP_sheet  = st.text_input("W90 MP sheet",  value="W90 MP")
        JUV_MP_sheet  = st.text_input("JUV MP sheet",  value="JUV MP")
        FIV_MP_sheet  = st.text_input("FIV MP sheet",  value="FIV MP")
    with c2:
        W90_WTG_sheet = st.text_input("W90 WTG sheet", value="W90 WTG")
        JUV_WTG_sheet = st.text_input("JUV WTG sheet", value="JUV WTG")
        FIV_WTG_sheet = st.text_input("FIV WTG sheet", value="FIV WTG")

htd = 1/24
dty = 1/365

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1.3 – UPLOAD OPERATIONS EXCEL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📂  1.3  Upload Operations Excel Sheet")
st.markdown(
    "Download the input template from Google Drive: "
    "[Input_for_simulations.xlsx](https://drive.google.com/uc?export=download&id=1E3qED0b3qkmqtP9cF462BpfN_GKq0iO2)"
)

uploaded_excel = st.file_uploader("Upload your operations Excel file", type=["xlsx", "xls"])

excel_sheets   = {}
excel_path_obj = None

if uploaded_excel is not None:
    excel_bytes = uploaded_excel.read()
    excel_file  = pd.ExcelFile(io.BytesIO(excel_bytes))
    all_sheets  = excel_file.sheet_names
    data_sheets = all_sheets[1:]          # skip Read Me / instructions tab

    for sheet in data_sheets:
        df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=sheet)
        excel_sheets[sheet] = df

    excel_path_obj = io.BytesIO(excel_bytes)   # reusable buffer
    st.success(f"✅  Loaded {len(data_sheets)} sheets: {data_sheets}")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER CONSTANTS (column detection)
# ─────────────────────────────────────────────────────────────────────────────
_TIME_COLS    = ["valid_time", "time", "date", "datetime"]
_U_COLS       = ["10m_u_component_of_wind", "u10"]
_V_COLS       = ["10m_v_component_of_wind", "v10"]
_WAVE_MARKERS = {"swh", "mwd", "mwp", "mean_wave_direction", "mean_wave_period",
                 "significant_height_of_combined_wind_waves_and_swell"}
_WIND_MARKERS = {"u10", "v10", "10m_u_component_of_wind", "10m_v_component_of_wind"}
_DROP_COLS    = {"latitude", "longitude", "lat", "lon"}
_WAVE_HEIGHT_COLS = ["significant_height_of_combined_wind_waves_and_swell", "swh"]


def _find_col(df, candidates, label):
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        raise ValueError(f"Could not find {label} column. Found: {df.columns.tolist()}")
    return col


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 – WEATHER DATA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🌦️  2.  Weather Factor Analysis")

# ── 2.1  Upload pre-downloaded ERA5 ZIPs  ────────────────────────────────
st.subheader("2.1  Upload ERA5 Weather ZIPs")
st.markdown(
    "Upload the two ZIP files you previously downloaded from the "
    "[Copernicus CDS](https://cds.climate.copernicus.eu/). "
    "Alternatively, use the **Download from ERA5** button below if your API key is set."
)

col_proj, col_barge_w = st.columns(2)
with col_proj:
    project_zip_file = st.file_uploader("Project location ZIP (wind + waves)", type="zip",
                                        key="project_zip")
with col_barge_w:
    barge_zip_file   = st.file_uploader("Barge location ZIP (wind only)",       type="zip",
                                        key="barge_zip")

# ── ERA5 download helper  ─────────────────────────────────────────────────
def download_era5_via_api(api_key, latitude, longitude, variables, target_zip_path):
    """Download ERA5 data directly via CDS API."""
    if cdsapi is None:
        st.error("cdsapi is not installed. Please install it: pip install cdsapi")
        return False
    cdsarc_path = os.path.expanduser("~/.cdsapirc")
    with open(cdsarc_path, "w") as f:
        f.write(f"url: https://cds.climate.copernicus.eu/api\nkey: {api_key}\n")
    client  = cdsapi.Client()
    request = {
        "variable": variables,
        "location": {"longitude": longitude, "latitude": latitude},
        "date": ["1990-01-01/2025-12-31"],
        "data_format": "csv",
        "nocache": str(uuid.uuid4()),
    }
    if os.path.exists(target_zip_path):
        os.remove(target_zip_path)
    client.retrieve("reanalysis-era5-single-levels-timeseries", request, target_zip_path)
    return True

if API_key:
    if st.button("⬇️  Download weather data from ERA5 API"):
        with st.spinner("Downloading project location weather …"):
            try:
                ok = download_era5_via_api(
                    API_key, lat, lon,
                    ["10m_u_component_of_wind", "10m_v_component_of_wind",
                     "mean_wave_direction", "mean_wave_period",
                     "significant_height_of_combined_wind_waves_and_swell"],
                    "era5_project_location.zip"
                )
                if ok:
                    st.success("Project weather downloaded.")
            except Exception as e:
                st.error(f"Download failed: {e}")
        with st.spinner("Downloading barge location weather …"):
            try:
                ok = download_era5_via_api(
                    API_key, lat_barge, lon_barge,
                    ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                    "era5_barge_location.zip"
                )
                if ok:
                    st.success("Barge weather downloaded.")
            except Exception as e:
                st.error(f"Download failed: {e}")


# ── 2.2  Format and model weather data  ──────────────────────────────────
def classify_csv(file_path):
    df   = pd.read_csv(file_path)
    cols = set(c.lower() for c in df.columns)
    is_wind = bool(cols & _WIND_MARKERS)
    is_wave = bool(cols & _WAVE_MARKERS)
    if   is_wind and not is_wave: label = "wind"
    elif is_wave and not is_wind: label = "wave"
    elif is_wind and is_wave:     label = "mixed"
    else:                         label = "unknown"
    return label, df

def _base_clean(df):
    df = df.copy()
    time_col = next((c for c in _TIME_COLS if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"No time column found. Columns: {df.columns.tolist()}")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.drop(columns=list(_DROP_COLS & set(df.columns)))
    return df.rename(columns={time_col: "time"}) if time_col != "time" else df

def prepare_wind_dataframe(df):
    df = _base_clean(df)
    u  = _find_col(df, _U_COLS, "U-wind")
    v  = _find_col(df, _V_COLS, "V-wind")
    df[u] = pd.to_numeric(df[u], errors="coerce")
    df[v] = pd.to_numeric(df[v], errors="coerce")
    df["Windspeed"]          = np.hypot(df[u], df[v])
    df["Adjusted_Windspeed"] = df["Windspeed"] * 0.9
    return df

def prepare_wave_dataframe(df):
    return _base_clean(df)

def load_weather_from_zip_bytes(zip_bytes, required_types):
    """Load weather data from an uploaded zip file (bytes)."""
    extract_dir = f"/tmp/era5_extract_{uuid.uuid4().hex}"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        zf.extractall(extract_dir)
    csv_files = glob.glob(os.path.join(extract_dir, "*.csv"))
    found = {}
    for fp in csv_files:
        label, df_raw = classify_csv(fp)
        if label in required_types and label not in found:
            found[label] = df_raw
        elif label == "mixed":
            found.setdefault("wind", df_raw)
            found.setdefault("wave", df_raw)
    shutil.rmtree(extract_dir, ignore_errors=True)
    return found


@st.cache_data(show_spinner="Loading weather data …")
def load_all_weather(project_zip_bytes, barge_zip_bytes):
    # Project
    dfs_proj  = load_weather_from_zip_bytes(project_zip_bytes, ["wind", "wave"])
    df_pw     = prepare_wind_dataframe(dfs_proj["wind"])
    df_waves  = prepare_wave_dataframe(dfs_proj["wave"])
    df_proj   = pd.merge(df_pw, df_waves, on="time", how="inner")
    # Barge
    dfs_barge = load_weather_from_zip_bytes(barge_zip_bytes, ["wind"])
    df_barge  = prepare_wind_dataframe(dfs_barge["wind"])
    return df_pw, df_waves, df_proj, df_barge


df_project_weather = None
df_barge_wind      = None

# Prefer uploaded zips; fall back to locally downloaded ones
_proj_zip_bytes  = project_zip_file.read()  if project_zip_file  else \
                   (open("era5_project_location.zip","rb").read() if os.path.exists("era5_project_location.zip") else None)
_barge_zip_bytes = barge_zip_file.read()    if barge_zip_file    else \
                   (open("era5_barge_location.zip","rb").read()   if os.path.exists("era5_barge_location.zip")   else None)

if _proj_zip_bytes and _barge_zip_bytes:
    try:
        _, _, df_project_weather, df_barge_wind = load_all_weather(
            _proj_zip_bytes, _barge_zip_bytes
        )
        st.success(f"✅  Weather loaded — {len(df_project_weather):,} project rows, "
                   f"{len(df_barge_wind):,} barge rows")
    except Exception as e:
        st.error(f"Weather loading error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 (OUTPUT) – WEATHER STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def get_wind_component_columns(df):
    u_col = next((c for c in _U_COLS if c in df.columns), None)
    v_col = next((c for c in _V_COLS if c in df.columns), None)
    if u_col is None or v_col is None:
        raise ValueError(f"Could not find wind component columns. Found: {df.columns.tolist()}")
    return u_col, v_col

def get_wave_height_column(df):
    col = next((c for c in _WAVE_HEIGHT_COLS if c in df.columns), None)
    if col is None:
        raise ValueError(f"No wave height column found. Expected one of {_WAVE_HEIGHT_COLS}.")
    return col

def validate_time_column(df, time_col="time"):
    if time_col not in df.columns:
        raise ValueError(f"Dataframe must contain '{time_col}' column.")

def create_wind_speed_summary(df_weather, wind_col="Adjusted_Windspeed"):
    df_weather = df_weather.copy()
    validate_time_column(df_weather)
    df_weather["time"] = pd.to_datetime(df_weather["time"], errors="coerce")
    df_weather[wind_col] = pd.to_numeric(df_weather[wind_col], errors="coerce")
    df_weather = df_weather.dropna(subset=["time", wind_col])
    def get_season(m):
        if m in [12,1,2]: return "Winter"
        elif m in [3,4,5]: return "Spring"
        elif m in [6,7,8]: return "Summer"
        else: return "Autumn"
    df_weather["Season"] = df_weather["time"].dt.month.apply(get_season)
    def calc(s, lbl):
        return pd.DataFrame({"Period":[lbl],"Average Wind Speed":[s.mean()],
                             "P75 Wind Speed":[s.quantile(.75)],"P90 Wind Speed":[s.quantile(.90)],
                             "Max Wind Speed":[s.max()]})
    frames = [calc(df_weather[wind_col], "All Data")]
    for season in ["Winter","Spring","Summer","Autumn"]:
        frames.append(calc(df_weather.loc[df_weather["Season"]==season, wind_col], season))
    return df_weather, pd.concat(frames, ignore_index=True)

def create_wave_summary(df_weather, wave_col=None):
    df_weather = df_weather.copy()
    validate_time_column(df_weather)
    if wave_col is None:
        wave_col = get_wave_height_column(df_weather)
    df_weather["time"] = pd.to_datetime(df_weather["time"], errors="coerce")
    df_weather[wave_col] = pd.to_numeric(df_weather[wave_col], errors="coerce")
    df_weather = df_weather.dropna(subset=["time", wave_col])
    def get_season(m):
        if m in [12,1,2]: return "Winter"
        elif m in [3,4,5]: return "Spring"
        elif m in [6,7,8]: return "Summer"
        else: return "Autumn"
    df_weather["Season"] = df_weather["time"].dt.month.apply(get_season)
    def calc(s, lbl):
        return pd.DataFrame({"Period":[lbl],"Average Wave Height":[s.mean()],
                             "P75 Wave Height":[s.quantile(.75)],"P90 Wave Height":[s.quantile(.90)],
                             "Max Wave Height":[s.max()]})
    frames = [calc(df_weather[wave_col], "All Data")]
    for season in ["Winter","Spring","Summer","Autumn"]:
        frames.append(calc(df_weather.loc[df_weather["Season"]==season, wave_col], season))
    return df_weather, pd.concat(frames, ignore_index=True)

def add_wind_direction(df_weather):
    df_weather = df_weather.copy()
    u_col, v_col = get_wind_component_columns(df_weather)
    df_weather[u_col] = pd.to_numeric(df_weather[u_col], errors="coerce")
    df_weather[v_col] = pd.to_numeric(df_weather[v_col], errors="coerce")
    df_weather["Wind_Direction"] = (270 - np.degrees(np.arctan2(df_weather[v_col], df_weather[u_col]))) % 360
    return df_weather

def plot_wind_wave_scatter(df_weather, wind_col="Adjusted_Windspeed", wave_col=None):
    if wave_col is None: wave_col = get_wave_height_column(df_weather)
    plot_df = df_weather[[wind_col, wave_col]].copy().apply(pd.to_numeric, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8,6))
    hb = ax.hexbin(plot_df[wind_col], plot_df[wave_col], gridsize=60, cmap="plasma", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Point Density")
    ax.set_xlabel("Adjusted Wind Speed (m/s)")
    ax.set_ylabel("Significant Wave Height (m)")
    ax.set_title("Wind Speed vs Wave Height (Density Heatmap)")
    ax.grid(True, alpha=0.3)
    return fig

def plot_variable_distribution(df_weather, column, title, xlabel, bins=50):
    series = pd.to_numeric(df_weather[column], errors="coerce").dropna()
    mu, sigma = series.mean(), series.std()
    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(series, bins=bins, density=True, alpha=0.7, edgecolor="black")
    if sigma > 0:
        x = np.linspace(series.min(), series.max(), 300)
        pdf = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
        ax.plot(x, pdf, linewidth=2)
    ax.set_xlabel(xlabel); ax.set_ylabel("Density"); ax.set_title(title); ax.grid(True, alpha=0.3)
    return fig

def plot_wind_rose(df_weather, wind_speed_col="Adjusted_Windspeed", direction_col="Wind_Direction"):
    plot_df = df_weather[[wind_speed_col, direction_col]].copy().apply(pd.to_numeric, errors="coerce").dropna()
    direction_labels = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    speed_bins   = [0,0.3,1.5,3.3,5.5,7.9,10.7,13.8,17.1,20.7,24.4,28.4,32.6,np.inf]
    speed_labels = ["<0.3","0.3-1.5","1.5-3.3","3.3-5.5","5.5-7.9","7.9-10.7",
                    "10.7-13.8","13.8-17.1","17.1-20.7","20.7-24.4","24.4-28.4","28.4-32.6",">32.6"]
    shifted_dir = (plot_df[direction_col] + 11.25) % 360
    direction_edges = np.arange(0, 360 + 22.5, 22.5)
    plot_df["dir_bin"] = pd.cut(shifted_dir, bins=direction_edges, labels=direction_labels, include_lowest=True, right=False)
    plot_df["speed_bin"] = pd.cut(plot_df[wind_speed_col], bins=speed_bins, labels=speed_labels, include_lowest=True, right=False)
    freq_table = pd.crosstab(plot_df["dir_bin"], plot_df["speed_bin"], normalize="all") * 100
    freq_table = freq_table.reindex(index=direction_labels, columns=speed_labels, fill_value=0)
    angles = np.deg2rad(np.arange(0, 360, 22.5))
    width  = np.deg2rad(22.5 * 0.9)
    fig, ax = plt.subplots(figsize=(9,9), subplot_kw=dict(polar=True))
    bottom = np.zeros(len(direction_labels))
    for speed_class in speed_labels:
        values = freq_table[speed_class].values
        ax.bar(angles, values, width=width, bottom=bottom, align="center", edgecolor="white", linewidth=0.8, label=speed_class)
        bottom += values
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_xticks(angles); ax.set_xticklabels(direction_labels)
    ax.set_title("Wind Rose", pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.1,1.1), title="Wind Speed (m/s)")
    return fig

def build_wave_exceedance_table(weather_df, max_wave, wave_col=None, step=0.5):
    if wave_col is None: wave_col = get_wave_height_column(weather_df)
    w = weather_df.copy()
    w[wave_col] = pd.to_numeric(w[wave_col], errors="coerce")
    w = w.dropna(subset=[wave_col])
    thresholds = np.arange(step, max_wave + step, step)
    total_n = len(w)
    rows = []
    for t in thresholds:
        cnt = int((w[wave_col] > t).sum())
        rows.append({"Threshold_m":float(t),"Exceedance_Count":cnt,"Total_Observations":total_n,
                     "Exceedance_Probability":round(cnt/total_n,4),"Exceedance_Percent":round(cnt/total_n*100,2)})
    return pd.DataFrame(rows)

if df_project_weather is not None:
    st.subheader("2.3  Weather Statistics")
    if st.button("📊  Generate weather plots & statistics"):
        df_project_weather = add_wind_direction(df_project_weather)
        _, wind_summary = create_wind_speed_summary(df_project_weather)
        _, wave_summary = create_wave_summary(df_project_weather)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Wind Speed Summary**")
            st.dataframe(wind_summary.set_index("Period").round(3))
        with col_b:
            st.markdown("**Wave Height Summary**")
            st.dataframe(wave_summary.set_index("Period").round(3))

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_variable_distribution(df_project_weather, "Adjusted_Windspeed",
                                                  "Distribution of Wind Speed", "Adjusted Wind Speed (m/s)"))
        with c2:
            wc = get_wave_height_column(df_project_weather)
            st.pyplot(plot_variable_distribution(df_project_weather, wc,
                                                  "Distribution of Significant Wave Height", "Significant Wave Height (m)"))
        st.pyplot(plot_wind_wave_scatter(df_project_weather))
        st.pyplot(plot_wind_rose(df_project_weather))

        st.markdown("**Wave Exceedance Table**")
        st.dataframe(build_wave_exceedance_table(df_project_weather, max_wave).set_index("Threshold_m"))


# ─────────────────────────────────────────────────────────────────────────────
#  CORE SIMULATION FUNCTIONS  (identical logic to notebook)
# ─────────────────────────────────────────────────────────────────────────────

def build_mp_timeline(excel_path_or_bytes, sheet_name, carry_cap, total_units, htd=1/24):
    if hasattr(excel_path_or_bytes, "seek"): excel_path_or_bytes.seek(0)
    raw = pd.read_excel(excel_path_or_bytes, sheet_name=sheet_name)
    raw.columns = [str(c).strip() for c in raw.columns]
    seq_col, dur_col, load_col, inst_col = "N", "Seq Dur (hrs)", "Loading", "Installation"
    desc_candidates = [c for c in raw.columns if str(c).strip().startswith("Description")]
    desc_col = desc_candidates[0]
    def norm(x):
        s = str(x).strip().upper(); return "" if s in {"","FALSE","NAN","NONE"} else s
    load_flags = raw[load_col].map(norm); inst_flags = raw[inst_col].map(norm)
    load_start_idx = int(raw.index[load_flags.eq("START")][0])
    load_end_idx   = int(raw.index[load_flags.eq("END")][0])
    load_phase = raw.iloc[:load_end_idx+1].reset_index(drop=True)
    inst_phase = raw.iloc[load_end_idx+1:].reset_index(drop=True)
    load_phase_flags = load_phase[load_col].map(norm)
    inst_phase_flags = inst_phase[inst_col].map(norm)
    inst_start_positions = inst_phase.index[inst_phase_flags.eq("START")]
    inst_decrement_pos   = int(inst_start_positions[0]) if len(inst_start_positions) > 0 else 0
    def phase_to_lists(phase_df, marker_series, phase_name):
        return {"seq":pd.to_numeric(phase_df[seq_col],errors="coerce").fillna(0).astype(int).tolist(),
                "desc":phase_df[desc_col].astype(str).tolist(),
                "dur":pd.to_numeric(phase_df[dur_col],errors="coerce").fillna(0).astype(float).tolist(),
                "mark":marker_series.tolist(),"phase":[phase_name]*len(phase_df),
                "weather_restricted":[phase_name=="installation"]*len(phase_df)}
    load_data = phase_to_lists(load_phase, load_phase_flags, "loading")
    inst_data = phase_to_lists(inst_phase, inst_phase_flags, "installation")
    inventory = 0; units_left = int(total_units); timeline_rows = []
    def cycles_left():
        if inventory > 0: return math.ceil(units_left/carry_cap)+1
        return math.ceil(units_left/carry_cap) if units_left > 0 else 0
    def add_row(sq,ds,dr,ph,wr):
        timeline_rows.append({"Sequence":int(sq),"Description":ds,"Phase":ph,
                               "Weather_Restricted":bool(wr),"Seq_Duration_Days":float(dr)*htd,
                               "Inventory":int(inventory),"WTG_Left":int(units_left),"Cycles_Left":int(cycles_left())})
    while units_left > 0 or inventory > 0:
        while inventory < carry_cap and units_left > 0:
            i = 0
            while i < len(load_data["seq"]):
                add_row(load_data["seq"][i],load_data["desc"][i],load_data["dur"][i],load_data["phase"][i],load_data["weather_restricted"][i])
                if load_data["mark"][i]=="START": inventory+=1; units_left-=1
                if load_data["mark"][i]=="END": break
                i+=1
        while inventory > 0:
            for i in range(len(inst_data["seq"])):
                add_row(inst_data["seq"][i],inst_data["desc"][i],inst_data["dur"][i],inst_data["phase"][i],inst_data["weather_restricted"][i])
                if i==inst_decrement_pos: inventory-=1
    out = pd.DataFrame(timeline_rows)
    out.insert(0,"N",range(1,len(out)+1))
    return out


def get_seasonal_eligible_indices(weather_times, start_date_str, window_days=7, latest_start="2022-12-31 23:00:00"):
    target   = pd.Timestamp(start_date_str)
    times_pd = pd.DatetimeIndex(weather_times)
    cutoff   = pd.Timestamp(latest_start)
    target_doy    = target.day_of_year
    doys          = times_pd.day_of_year
    circular_diff = np.minimum(np.abs(doys-target_doy), 365-np.abs(doys-target_doy))
    eligible = np.where((circular_diff <= window_days) & (times_pd <= cutoff))[0]
    if len(eligible)==0:
        raise ValueError(f"No weather timestamps found within ±{window_days} days of {target.strftime('%d %B')}.")
    return eligible


def simulate_weather_impacts_mp(timeline_df, weather_df, excel_path_or_bytes,
                                 transit_distance, transit_speed, sheet_name,
                                 simulations=1, eligible_indices=None, seed=42,
                                 progress_every_sims=50, max_wait_hours_per_operation=5000, htd=1/24):
    if hasattr(excel_path_or_bytes,"seek"): excel_path_or_bytes.seek(0)
    transit_hours = float(transit_distance)/float(transit_speed)
    raw = pd.read_excel(excel_path_or_bytes, sheet_name=sheet_name)
    raw.columns = [str(c).strip() for c in raw.columns]
    def norm_bool(x): return str(x).strip().upper()=="TRUE"
    op = raw[["N","Seq Dur (hrs)","Tr (hrs)","Transit","Oplim Waves (m)","Oplim Wind (m/s)"]].copy()
    op["N"]=pd.to_numeric(op["N"],errors="coerce")
    op["Seq Dur (hrs)"]=pd.to_numeric(op["Seq Dur (hrs)"],errors="coerce")
    op["Tr (hrs)"]=pd.to_numeric(op["Tr (hrs)"],errors="coerce")
    op["Transit"]=op["Transit"].map(norm_bool)
    op["Oplim Waves (m)"]=pd.to_numeric(op["Oplim Waves (m)"],errors="coerce").fillna(np.inf)
    op["Oplim Wind (m/s)"]=pd.to_numeric(op["Oplim Wind (m/s)"],errors="coerce").fillna(np.inf)
    nt=~op["Transit"]
    op.loc[nt,"Seq Dur (hrs)"]=op.loc[nt,"Seq Dur (hrs)"].fillna(op.loc[nt,"Tr (hrs)"])
    op.loc[nt,"Tr (hrs)"]=op.loc[nt,"Tr (hrs)"].fillna(op.loc[nt,"Seq Dur (hrs)"])
    op.loc[op["Transit"],["Seq Dur (hrs)","Tr (hrs)"]]=transit_hours
    op=op.dropna(subset=["N","Seq Dur (hrs)","Tr (hrs)"]).drop_duplicates(subset=["N"])
    op["N"]=op["N"].astype(int)
    low_map=dict(zip(op["N"],np.minimum(op["Seq Dur (hrs)"],op["Tr (hrs)"])))
    high_map=dict(zip(op["N"],np.maximum(op["Seq Dur (hrs)"],op["Tr (hrs)"])))
    wave_map=dict(zip(op["N"],op["Oplim Waves (m)"])); wind_map=dict(zip(op["N"],op["Oplim Wind (m/s)"]))
    transit_map=dict(zip(op["N"],op["Transit"]))
    _TCOLS=["N","Sequence","Inventory","WTG_Left","Cycles_Left","Weather_Restricted","Phase"]
    base=timeline_df[_TCOLS].copy()
    base["Sequence"]=base["Sequence"].astype(int)
    base["Weather_Restricted"]=base["Weather_Restricted"].astype(bool)
    wave_col=_find_col(weather_df,_WAVE_HEIGHT_COLS,"wave height")
    w=weather_df[["time",wave_col,"Adjusted_Windspeed"]].copy()
    w["time"]=pd.to_datetime(w["time"],errors="coerce")
    w[wave_col]=pd.to_numeric(w[wave_col],errors="coerce")
    w["Adjusted_Windspeed"]=pd.to_numeric(w["Adjusted_Windspeed"],errors="coerce")
    w=w.dropna().sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
    weather_times=w["time"].to_numpy(dtype="datetime64[ns]")
    weather_waves=w[wave_col].to_numpy(dtype=float)
    weather_winds=w["Adjusted_Windspeed"].to_numpy(dtype=float)
    eligible_idx=eligible_indices if eligible_indices is not None else np.arange(len(weather_times))
    rng=np.random.default_rng(seed); n_rows=len(base)
    total_matrix=np.zeros((n_rows,simulations),dtype=float)
    downtime_matrix=np.zeros((n_rows,simulations),dtype=float)
    seqs=base["Sequence"].to_numpy(dtype=int)
    seq_dur_low=np.array([low_map[s]  for s in seqs],dtype=float)
    seq_dur_high=np.array([high_map[s] for s in seqs],dtype=float)
    seq_wave_lim=np.array([wave_map[s] for s in seqs],dtype=float)
    seq_wind_lim=np.array([wind_map[s] for s in seqs],dtype=float)
    seq_is_transit=np.array([transit_map[s] for s in seqs],dtype=bool)
    seq_wx_restricted=base["Weather_Restricted"].to_numpy(dtype=bool)
    def get_start_wait_hours(start_time,wave_limit,wind_limit):
        idx=int(np.searchsorted(weather_times,np.datetime64(pd.Timestamp(start_time)),side="left"))
        for wh in range(max_wait_hours_per_operation+1):
            i=idx+wh
            if i>=len(weather_times): raise RuntimeError("Ran out of weather data.")
            if weather_waves[i]<=wave_limit and weather_winds[i]<=wind_limit: return wh,i
        raise RuntimeError("Exceeded max_wait_hours_per_operation.")
    summary_rows=[]
    for sim_idx in range(simulations):
        sim_t0=time.time()
        current_time=pd.Timestamp(weather_times[int(rng.choice(eligible_idx))])
        sim_start=current_time; total_active=total_down=0.0
        for row_i in range(n_rows):
            if seq_wx_restricted[row_i]:
                down_hours,passable_idx=get_start_wait_hours(current_time,seq_wave_lim[row_i],seq_wind_lim[row_i])
                op_start=pd.Timestamp(weather_times[passable_idx])
            else:
                down_hours,op_start=0.0,current_time
            active_hours=(transit_hours if seq_is_transit[row_i]
                          else float(rng.uniform(seq_dur_low[row_i],seq_dur_high[row_i])))
            total_matrix[row_i,sim_idx]=(down_hours+active_hours)*htd
            downtime_matrix[row_i,sim_idx]=down_hours*htd
            total_active+=active_hours; total_down+=down_hours
            current_time=op_start+pd.Timedelta(hours=active_hours)
        sim_elapsed=time.time()-sim_t0
        summary_rows.append({"Simulation":f"Sim{sim_idx+1}","Start_Date":sim_start,"Finish_Date":current_time,
                              "Total_Active_Days":total_active*htd,"Total_Downtime_Days":total_down*htd,
                              "Total_Project_Days":(total_active+total_down)*htd,"Elapsed_Seconds":sim_elapsed})
    sim_cols=[f"Sim{i}" for i in range(1,simulations+1)]
    df_total=pd.concat([base,pd.DataFrame(total_matrix,columns=sim_cols,index=base.index)],axis=1)
    df_downtime=pd.concat([base,pd.DataFrame(downtime_matrix,columns=sim_cols,index=base.index)],axis=1)
    return df_total, df_downtime, pd.DataFrame(summary_rows)


def build_mp_weather_project_summary(w90=None, barge_s=None, juv=None, fiv=None):
    _PERCENTILES=[(f"P{p}",p) for p in [10,50,75,90,100]]
    _METRICS=[("Total_Project_Days","Total_Operation"),("Total_Downtime_Days","Total_Downtime"),("Total_Active_Days","Net_Operation")]
    def summarize_one(df, label):
        row={"Project":label}
        for col,out_name in _METRICS:
            vals=pd.to_numeric(df[col],errors="coerce").dropna().to_numpy()
            for p_lbl,p_val in _PERCENTILES:
                row[f"{p_lbl}_{out_name}_Days"]=float(np.percentile(vals,p_val))
            row[f"Std_Dev_{out_name}_Days"]=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0
        return pd.DataFrame([row])
    inputs=[("W90",w90),("barge",barge_s),("JUV",juv),("FIV",fiv)]
    frames=[summarize_one(df,name) for name,df in inputs if df is not None]
    return pd.concat(frames,ignore_index=True).reset_index(drop=True)


def plot_mp_cumulative(results, title_suffix=""):
    _STYLE={"W90":{"color":"red","linestyle":"-"},"barge":{"color":"red","linestyle":":"},
            "JUV":{"color":"blue","linestyle":"-"},"FIV":{"color":"orange","linestyle":"-"}}
    fig,ax=plt.subplots(figsize=(12,7))
    fastest_p50=np.inf
    data=[]
    for name,res in results.items():
        df=res["total"]; sim_cols=[c for c in df.columns if str(c).startswith("Sim")]
        cum=df[sim_cols].cumsum(axis=0); p50=cum.median(axis=1)
        data.append((name,cum,p50,sim_cols)); fastest_p50=min(fastest_p50,float(p50.iloc[-1]))
    for name,cum,p50,sim_cols in data:
        style=_STYLE.get(name,{"color":"grey","linestyle":"-"})
        x=np.arange(1,len(cum)+1)
        ax.plot(x,cum[sim_cols],color="grey",alpha=0.18,linewidth=1)
        ax.plot(x,p50,label=f"{name} P50",color=style["color"],linestyle=style["linestyle"],linewidth=2.8)
    ax.set_title(f"Cumulative MP Duration {title_suffix}".strip())
    ax.set_xlabel("Sequence Step"); ax.set_ylabel("Cumulative Duration (days)")
    ax.grid(True,alpha=0.3); ax.legend()
    if fastest_p50>0:
        ax.secondary_yaxis("right",functions=(lambda y:y/fastest_p50*100,lambda y:y/100*fastest_p50)).set_ylabel("Relative Completion (%)")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 – RUN MP SIMULATIONS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🏗️  3.  Foundation / Monopile Installation Simulations")

mp_weather_results = {}

if uploaded_excel and df_project_weather is not None:
    if st.button("▶️  Run MP Simulations"):
        _MP_CONFIGS={}
        if W90_sim:  _MP_CONFIGS["W90"]  = (W90_MP_sheet,  W90_carry_cap)
        if barge:    _MP_CONFIGS["barge"]= (W90_MP_sheet,  W90_carry_cap)
        if JUV_sim:  _MP_CONFIGS["JUV"]  = (JUV_MP_sheet,  JUV_carry_cap)
        if FIV_sim:  _MP_CONFIGS["FIV"]  = (FIV_MP_sheet,  FIV_carry_cap)

        weather_times_arr = df_project_weather["time"].to_numpy(dtype="datetime64[ns]")
        try:
            seasonal_eligible = get_seasonal_eligible_indices(weather_times_arr, Sim_start_date,
                                                              window_days=7, latest_start="2022-12-31 23:00:00")
            st.info(f"Seasonal eligible starts: {len(seasonal_eligible)}")
        except Exception as e:
            st.error(f"Seasonal index error: {e}"); seasonal_eligible=None

        if seasonal_eligible is not None:
            progress_bar = st.progress(0)
            vessel_names = list(_MP_CONFIGS.keys())
            for vi, (vessel, (sheet, cap)) in enumerate(_MP_CONFIGS.items()):
                with st.spinner(f"Simulating {vessel} …"):
                    try:
                        excel_path_obj.seek(0)
                        timeline = build_mp_timeline(excel_path_obj, sheet, cap, N_WTG, htd)
                        excel_path_obj.seek(0)
                        total, downtime, summary = simulate_weather_impacts_mp(
                            timeline_df=timeline, weather_df=df_project_weather,
                            excel_path_or_bytes=excel_path_obj, sheet_name=sheet,
                            simulations=Simulations, eligible_indices=seasonal_eligible,
                            transit_distance=0.5, transit_speed=1.0, htd=htd)
                        mp_weather_results[vessel]={"timeline":timeline,"total":total,"downtime":downtime,"summary":summary}
                        st.success(f"✅  {vessel} complete")
                    except Exception as e:
                        st.error(f"{vessel} failed: {e}")
                progress_bar.progress((vi+1)/len(vessel_names))

            if mp_weather_results:
                st.subheader("MP Simulation Results")
                st.pyplot(plot_mp_cumulative(mp_weather_results, title_suffix="(With Weather)"))
                summary_df = build_mp_weather_project_summary(
                    w90=mp_weather_results.get("W90",{}).get("summary"),
                    barge_s=mp_weather_results.get("barge",{}).get("summary"),
                    juv=mp_weather_results.get("JUV",{}).get("summary"),
                    fiv=mp_weather_results.get("FIV",{}).get("summary"),
                )
                st.dataframe(summary_df.round(2))
                st.session_state["mp_weather_results"] = mp_weather_results
else:
    st.info("Upload the operations Excel file and load weather data to enable simulations.")


# ─────────────────────────────────────────────────────────────────────────────
#  NOTE ON WTG SIMULATIONS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🌀  4–5.  WTG Installation & Full Campaign Simulations")
st.info(
    "WTG installation simulations (Sections 4 and 5 of the notebook) build on the MP "
    "results above. They run automatically after MP simulations complete. "
    "The full campaign results — including weather-adjusted P50/P90 durations and "
    "downtime comparison charts — will appear here once the MP step finishes."
)

if "mp_weather_results" in st.session_state and st.session_state["mp_weather_results"]:
    st.success("MP simulations done — WTG simulation can be triggered once you add the WTG "
               "build functions. The MP results are stored in session state and ready to chain.")


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("W90 Monte Carlo v6 · Converted from Google Colab to Streamlit")
