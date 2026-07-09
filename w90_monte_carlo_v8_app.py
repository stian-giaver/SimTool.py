"""
W90 Monte Carlo – Streamlit App (v8)

New in v8
─────────
• THREE weather locations are now downloaded / loaded:
      – Project site : wind + waves   (installation, offshore loading)
      – Barge site   : wind + waves   (nearshore loading)   ← was wind-only in v7
      – Yard site    : wind only      (yard loading)        ← NEW location
  All three are downloaded from ERA5 (or uploaded as ZIPs) and reformatted the
  same way (wind dataframes get Windspeed / Adjusted_Windspeed; wave dataframes
  are merged on time).

• Per-phase weather routing (deterministic):
      – Loading @ Yard      → YARD weather, WIND ONLY (no wave downtime)
      – Loading @ Nearshore → BARGE weather (wind + waves)
      – Loading @ Offshore  → PROJECT weather (wind + waves)
      – ALL installation     → PROJECT weather (wind + waves)
      – Transit rows         → PROJECT weather (both transit-to-site and
                               transit-back-to-loading are gated by project wx)

• Loading-mode gating: a loading mode can only be selected once the weather data
  for its corresponding location is available (Offshore→project, Nearshore→barge,
  Yard→yard).

• New W90 WTG loading model (sheet 'W90 WTG '):
      – Loading is now done one turbine at a time.  The parallel-loading block
        (Parallell 1 + Parallell 2 rows) repeats carry_cap times before the
        vessel transits away.
      – Parallell 1 and Parallell 2 are treated as taking the same time, so the
        per-turbine loading duration uses the Parallell-1 hourly durations.
      – In the Excel export each loading step pairs the Parallell-1 description
        with its corresponding Parallell-2 description on one row, using the
        Parallell-1 duration.
      – The pre-loading row (before Loading START) runs once per loading cycle.

New in v7
─────────
• Foundation type selection:  Monopile  or  Jacket  (or no foundation phase at all)
• Per-vessel loading-mode selection.  For each vessel (W90 / JUV / FIV) the user
  can tick any combination of:
      – Loading Offshore   (transit fixed at 0.5 hours)
      – Loading Nearshore  (transit distance = barge distance)
      – Loading at Yard    (transit distance = dock distance)
  Every (vessel × loading-mode) combination becomes a separately simulated case
  that is fully comparable on durations & downtime against every other case.
• Jacket installation timeline builder for sheets that have no Loading phase
  markers (jackets are barge-fed at the project site).

Backwards-compatible:  all v6/v7 simulation functions are kept and re-used.
"""

import streamlit as st
import os, io, math, time, uuid, glob, zipfile, shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import folium
    from streamlit_folium import st_folium
except ImportError:
    folium = None
    st_folium = None

try:
    import cdsapi
except ImportError:
    cdsapi = None

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="W90 Monte Carlo v8", page_icon="🌊", layout="wide")
st.title("🌊 W90 Monte Carlo – Offshore Wind Installation Simulator  ·  v8")
st.markdown("---")




# ─────────────────────────────────────────────────────────────────────────────
#  MAP-BASED COORDINATE PICKER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _round_to_era5_grid(value):
    """Round latitude/longitude to the nearest ERA5 0.25° grid point."""
    return round(float(value) * 4) / 4


def _safe_rerun():
    """Rerun Streamlit across older/newer Streamlit versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def coordinate_picker(label, default_lat, default_lon, key):
    """Manual coordinate inputs plus an optional click-to-select Folium map.

    This version avoids writing directly to a widget's own session_state key
    after that widget has been created. That pattern can raise Streamlit errors
    and can also prevent number_input boxes from visually updating.

    Instead:
      1) Stable state keys hold the actual latitude/longitude values.
      2) number_input widgets use versioned keys.
      3) When a map click is applied, the stable values are updated and the
         widget key version is incremented, forcing the boxes to refresh.
    """
    lat_value_key = f"{key}_lat_value"
    lon_value_key = f"{key}_lon_value"
    show_map_key = f"{key}_show_map"
    selected_lat_key = f"{key}_selected_lat"
    selected_lon_key = f"{key}_selected_lon"
    input_version_key = f"{key}_input_version"

    st.session_state.setdefault(lat_value_key, float(default_lat))
    st.session_state.setdefault(lon_value_key, float(default_lon))
    st.session_state.setdefault(show_map_key, False)
    st.session_state.setdefault(input_version_key, 0)

    version = int(st.session_state[input_version_key])

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        lat_manual = st.number_input(
            f"{label} Latitude",
            value=float(st.session_state[lat_value_key]),
            step=0.25,
            format="%.2f",
            key=f"{key}_lat_input_v{version}",
        )
    with c2:
        lon_manual = st.number_input(
            f"{label} Longitude",
            value=float(st.session_state[lon_value_key]),
            step=0.25,
            format="%.2f",
            key=f"{key}_lon_input_v{version}",
        )
    with c3:
        st.write("")
        if st.button("🗺️ Select coordinates", key=f"{key}_select_btn"):
            st.session_state[show_map_key] = not st.session_state[show_map_key]

    # Keep stable values synchronized with manual edits.
    st.session_state[lat_value_key] = float(lat_manual)
    st.session_state[lon_value_key] = float(lon_manual)

    if st.session_state[show_map_key]:
        if folium is None or st_folium is None:
            st.error(
                "Map selector requires `folium` and `streamlit-folium`. "
                "Install them, then restart the app."
            )
        else:
            current_lat = float(st.session_state[lat_value_key])
            current_lon = float(st.session_state[lon_value_key])
            is_zero_sentinel = abs(current_lat) < 1e-9 and abs(current_lon) < 1e-9

            m = folium.Map(
                location=[current_lat, current_lon],
                zoom_start=2 if is_zero_sentinel else 5,
            )
            if not is_zero_sentinel:
                folium.Marker(
                    [current_lat, current_lon],
                    tooltip=f"Current {label} location",
                ).add_to(m)

            st.caption(
                "Click the map to choose a point, then press Apply selected point. "
                "The selected coordinate is rounded to the nearest ERA5 0.25° grid point."
            )

            result = st_folium(
                m,
                height=350,
                use_container_width=True,
                key=f"{key}_map",
                returned_objects=["last_clicked"],
            )
            clicked = result.get("last_clicked") if result else None

            if clicked:
                raw_lat = float(clicked["lat"])
                raw_lon = float(clicked["lng"])
                st.session_state[selected_lat_key] = _round_to_era5_grid(raw_lat)
                st.session_state[selected_lon_key] = _round_to_era5_grid(raw_lon)

            if selected_lat_key in st.session_state and selected_lon_key in st.session_state:
                selected_lat = float(st.session_state[selected_lat_key])
                selected_lon = float(st.session_state[selected_lon_key])
                st.info(
                    f"Selected {label} point: {selected_lat:.2f}, {selected_lon:.2f} "
                    "(rounded to ERA5 grid)."
                )
                if st.button(f"✅ Apply selected {label} point", key=f"{key}_apply_selected"):
                    st.session_state[lat_value_key] = selected_lat
                    st.session_state[lon_value_key] = selected_lon
                    st.session_state[input_version_key] = version + 1
                    del st.session_state[selected_lat_key]
                    del st.session_state[selected_lon_key]
                    _safe_rerun()
            else:
                st.caption("No map click registered yet for this location.")

    return float(st.session_state[lat_value_key]), float(st.session_state[lon_value_key])


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – INPUTS
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("⚙️  1. Project Inputs", expanded=True):

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Project")
        N_WTG          = st.number_input("Number of turbines (N_WTG)", min_value=1, value=50, step=1)
        Simulations    = st.number_input("Simulations", min_value=10, max_value=5000, value=500, step=50)
        Sim_start_date = st.text_input("Sim start date (D-M-YYYY)", value="1-7-2027")
        API_key        = st.text_input("ERA5 API key", value="", type="password")
    with col2:
        st.subheader("Locations (ERA5 0.25° grid)")
        lat, lon = coordinate_picker("Project", 35.25, 140.75, "project")
        lat_barge, lon_barge = coordinate_picker("Barge", 56.50, -2.50, "barge")
        lat_yard, lon_yard = coordinate_picker("Yard", 0.0, 0.0, "yard")
        st.caption(
            "Project & Barge now both download **wind + waves**.  "
            "Yard downloads **wind only**.  Leave Yard at 0.00 / 0.00 if you "
            "are not loading at the yard."
        )
    with col3:
        st.subheader("Vessel & Distance")
        dock_distance  = st.number_input("Dock (Yard) transit distance (Nm)",  value=185, step=5)
        barge_distance = st.number_input("Barge (Nearshore) transit distance (Nm)", value=115, step=5)
        W90_carry_cap  = st.number_input("W90 carry capacity",  value=8, step=1)
        JUV_carry_cap  = st.number_input("JUV carry capacity",  value=6, step=1)
        FIV_carry_cap  = st.number_input("FIV carry capacity",  value=5, step=1)

    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Transit Speeds (kn)")
        W90_transit_speed  = st.number_input("W90",                  value=10, step=1)
        Comp_transit_speed = st.number_input("Competitor (JUV)",     value=10, step=1)
        Fiv_transit_speed  = st.number_input("FIV",                  value=10, step=1)
    with col5:
        st.subheader("Global Operations")
        max_wave          = st.number_input("Max wave height (m)",          value=6,  step=1)
        intermediate_days = st.number_input("Intermediate days (Foundation → WTG)", value=14, step=1)

    # ─────────────────────────────────────────────────────────────────────
    # 1.1   Foundation phase selection
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏗️  1.1  Foundation Phase")

    simulate_foundations = st.checkbox("Simulate foundation installation?", value=True)
    if simulate_foundations:
        foundation_type = st.radio(
            "Foundation type",
            options=["Monopile", "Jacket"],
            horizontal=True,
            help="Monopile uses loading-yard cycles (Loading START/END markers). "
                 "Jacket uses barge-fed installation cycles at the project site."
        )
    else:
        foundation_type = None
        st.info("Foundation phase will be skipped — only WTG installation will be simulated.")

    # ─────────────────────────────────────────────────────────────────────
    # 1.2   Vessel selection
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🚢  1.2  Vessels to Simulate")

    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        W90_sim = st.checkbox("Include W90", value=True)
    with vc2:
        JUV_sim = st.checkbox("Include JUV", value=True)
    with vc3:
        FIV_sim = st.checkbox("Include FIV", value=True)

    # ─────────────────────────────────────────────────────────────────────
    # 1.3   Per-vessel loading modes
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📦  1.3  Loading Modes per Vessel")

    # Determine which loading locations have usable coordinates.
    # A coordinate pair counts as "filled out" if it is not the (0.0, 0.0) sentinel.
    def _coords_filled(latitude, longitude):
        return not (abs(float(latitude)) < 1e-9 and abs(float(longitude)) < 1e-9)

    _mode_available = {
        "Offshore":  _coords_filled(lat,       lon),        # loaded at project site
        "Nearshore": _coords_filled(lat_barge, lon_barge),  # loaded at barge site
        "Yard":      _coords_filled(lat_yard,  lon_yard),   # loaded at yard site
    }
    _unavailable_modes = [m for m, ok in _mode_available.items() if not ok]
    if _unavailable_modes:
        st.warning(
            "These loading modes are **disabled** until their coordinates are filled "
            "in (Project Inputs → Locations): " + ", ".join(_unavailable_modes) + ".  "
            "Offshore needs Project coords, Nearshore needs Barge coords, "
            "Yard needs Yard coords."
        )

    st.caption(
        "Choose loading modes independently for **foundation** and **WTG** installation. "
        "This allows, for example, foundations to be loaded nearshore while WTGs are loaded at the yard.\n\n"
        "• **Offshore**  — loaded at the project site (PROJECT weather), transit fixed at **0.5 hours**.\n"
        "• **Nearshore** — loaded at the barge location (BARGE weather), transit distance = **barge distance**.\n"
        "• **Yard**      — loaded at the yard (YARD weather, wind only), transit distance = **dock distance**.\n\n"
        "A mode cannot be selected until its location's coordinates have been filled out."
    )

    def _loading_mode_picker(vessel, enabled, default_modes, phase_key):
        """Render a row of three checkboxes for one vessel/phase; returns selected modes.

        Each loading mode is additionally disabled if its corresponding location's
        coordinates have not been filled out (see _mode_available above)."""
        st.markdown(f"**{vessel}**" + ("" if enabled else "  *(vessel disabled above)*"))
        c1, c2, c3 = st.columns(3)
        off_enabled  = enabled and _mode_available["Offshore"]
        near_enabled = enabled and _mode_available["Nearshore"]
        yard_enabled = enabled and _mode_available["Yard"]
        with c1:
            m_off  = st.checkbox("Offshore",  value=("Offshore"  in default_modes) and off_enabled,
                                 key=f"{phase_key}_{vessel}_off",  disabled=not off_enabled)
        with c2:
            m_near = st.checkbox("Nearshore", value=("Nearshore" in default_modes) and near_enabled,
                                 key=f"{phase_key}_{vessel}_near", disabled=not near_enabled)
        with c3:
            m_yard = st.checkbox("Yard",      value=("Yard"      in default_modes) and yard_enabled,
                                 key=f"{phase_key}_{vessel}_yard", disabled=not yard_enabled)
        selected = []
        if enabled:
            if m_off  and off_enabled:  selected.append("Offshore")
            if m_near and near_enabled: selected.append("Nearshore")
            if m_yard and yard_enabled: selected.append("Yard")
        return selected

    lm_col1, lm_col2 = st.columns(2)
    with lm_col1:
        st.markdown("#### Foundation loading cases")
        fdn_picker_enabled = simulate_foundations
        W90_foundation_modes = _loading_mode_picker("W90", W90_sim and fdn_picker_enabled, default_modes=["Yard", "Nearshore"], phase_key="fdn")
        JUV_foundation_modes = _loading_mode_picker("JUV", JUV_sim and fdn_picker_enabled, default_modes=["Yard"], phase_key="fdn")
        FIV_foundation_modes = _loading_mode_picker("FIV", FIV_sim and fdn_picker_enabled, default_modes=["Yard"], phase_key="fdn")
        if not simulate_foundations:
            st.info("Foundation loading cases are disabled because the foundation phase is skipped.")
    with lm_col2:
        st.markdown("#### WTG loading cases")
        W90_wtg_modes = _loading_mode_picker("W90", W90_sim, default_modes=["Yard", "Nearshore"], phase_key="wtg")
        JUV_wtg_modes = _loading_mode_picker("JUV", JUV_sim, default_modes=["Yard"], phase_key="wtg")
        FIV_wtg_modes = _loading_mode_picker("FIV", FIV_sim, default_modes=["Yard"], phase_key="wtg")

    # ─────────────────────────────────────────────────────────────────────
    # 1.4   Excel sheet names
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📑  1.4  Excel Sheet Names")
    c1, c2, c3 = st.columns(3)
    with c1:
        W90_MP_sheet     = st.text_input("W90 MP sheet",     value="W90 MP")
        W90_Jacket_sheet = st.text_input("W90 Jacket sheet", value="W90 Jacket")
        W90_WTG_sheet    = st.text_input("W90 WTG sheet",    value="W90 WTG ")
    with c2:
        JUV_MP_sheet     = st.text_input("JUV MP sheet",     value="JUV MP")
        JUV_Jacket_sheet = st.text_input("JUV Jacket sheet", value="JUV Jacket")
        JUV_WTG_sheet    = st.text_input("JUV WTG sheet",    value="JUV WTG")
    with c3:
        FIV_MP_sheet     = st.text_input("FIV MP sheet",     value="FIV MP")
        FIV_Jacket_sheet = st.text_input("FIV Jacket sheet", value="FIV Jacket")
        FIV_WTG_sheet    = st.text_input("FIV WTG sheet",    value="FIV WTG")


htd = 1 / 24   # hours-to-days


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers to resolve loading-mode → transit distance
# ─────────────────────────────────────────────────────────────────────────────
def _mode_transit_distance(mode, transit_speed, dock_distance, barge_distance):
    """Return the transit *distance* (Nm) that corresponds to a loading mode.
    Offshore is defined by a fixed 0.5-hour transit, so we back-calculate the
    equivalent distance from the vessel's transit speed."""
    if mode == "Offshore":
        return 0.5 * float(transit_speed)
    if mode == "Nearshore":
        return float(barge_distance)
    if mode == "Yard":
        return float(dock_distance)
    raise ValueError(f"Unknown loading mode: {mode}")


# Sheet & cap lookup tables
_VESSEL_INFO = {}    # populated below once sheet inputs exist


def _populate_vessel_info():
    """Build the master vessel info dict used by the run-pipeline."""
    return {
        "W90": {
            "selected"     : W90_sim,
            "foundation_modes": W90_foundation_modes,
            "wtg_modes"       : W90_wtg_modes,
            "mp_sheet"     : W90_MP_sheet,
            "jacket_sheet" : W90_Jacket_sheet,
            "wtg_sheet"    : W90_WTG_sheet,
            "carry_cap"    : int(W90_carry_cap),
            "transit_speed": float(W90_transit_speed),
        },
        "JUV": {
            "selected"     : JUV_sim,
            "foundation_modes": JUV_foundation_modes,
            "wtg_modes"       : JUV_wtg_modes,
            "mp_sheet"     : JUV_MP_sheet,
            "jacket_sheet" : JUV_Jacket_sheet,
            "wtg_sheet"    : JUV_WTG_sheet,
            "carry_cap"    : int(JUV_carry_cap),
            "transit_speed": float(Comp_transit_speed),
        },
        "FIV": {
            "selected"     : FIV_sim,
            "foundation_modes": FIV_foundation_modes,
            "wtg_modes"       : FIV_wtg_modes,
            "mp_sheet"     : FIV_MP_sheet,
            "jacket_sheet" : FIV_Jacket_sheet,
            "wtg_sheet"    : FIV_WTG_sheet,
            "carry_cap"    : int(FIV_carry_cap),
            "transit_speed": float(Fiv_transit_speed),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1.5 – UPLOAD OPERATIONS EXCEL
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("📂  1.5  Upload Operations Excel Sheet")
st.markdown(
    "Download the input template: "
    "[Input_for_simulations.xlsx](https://github.com/Frigstad-Engineering/Sim-Tool/blob/main/Input_for_simulations.xlsx)"
)

uploaded_excel = st.file_uploader("Upload your operations Excel file", type=["xlsx", "xls"])
excel_bytes    = None

if uploaded_excel is not None:
    excel_bytes    = uploaded_excel.read()
    excel_file_obj = pd.ExcelFile(io.BytesIO(excel_bytes))
    data_sheets    = excel_file_obj.sheet_names[1:]
    st.success(f"✅  Loaded {len(data_sheets)} data sheets: {data_sheets}")


# ═════════════════════════════════════════════════════════════════════════════
#  COLUMN-DETECTION CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
_TIME_COLS        = ["valid_time", "time", "date", "datetime"]
_U_COLS           = ["10m_u_component_of_wind", "u10"]
_V_COLS           = ["10m_v_component_of_wind", "v10"]
_WAVE_MARKERS     = {"swh", "mwd", "mwp", "mean_wave_direction", "mean_wave_period",
                     "significant_height_of_combined_wind_waves_and_swell"}
_WIND_MARKERS     = {"u10", "v10", "10m_u_component_of_wind", "10m_v_component_of_wind"}
_DROP_COLS        = {"latitude", "longitude", "lat", "lon"}
_WAVE_HEIGHT_COLS = ["significant_height_of_combined_wind_waves_and_swell", "swh"]


def _find_col(df, candidates, label):
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        raise ValueError(f"Could not find {label} column. Found: {df.columns.tolist()}")
    return col


def _norm_marker(x):
    s = str(x).strip().upper()
    return "" if s in {"", "FALSE", "NAN", "NONE"} else s


def _norm_bool(x):
    return str(x).strip().upper() == "TRUE"


def _excel_read(src, sheet_name):
    if hasattr(src, "seek"):
        src.seek(0)
    return pd.read_excel(src, sheet_name=sheet_name)


# ═════════════════════════════════════════════════════════════════════════════
#  WEATHER HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def classify_csv(fp):
    df   = pd.read_csv(fp)
    cols = set(c.lower() for c in df.columns)
    is_wind = bool(cols & _WIND_MARKERS)
    is_wave = bool(cols & _WAVE_MARKERS)
    if   is_wind and not is_wave: label = "wind"
    elif is_wave and not is_wind: label = "wave"
    elif is_wind and is_wave:     label = "mixed"
    else:                         label = "unknown"
    return label, df


def _base_clean(df):
    df       = df.copy()
    time_col = next((c for c in _TIME_COLS if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"No time column. Columns: {df.columns.tolist()}")
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
    extract_dir = f"/tmp/era5_{uuid.uuid4().hex}"
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


def _build_merged_site(zip_bytes):
    """Load a wind+wave site from a ZIP and return the merged dataframe."""
    dfs = load_weather_from_zip_bytes(zip_bytes, ["wind", "wave"])
    df_wind = prepare_wind_dataframe(dfs["wind"])
    df_wave = prepare_wave_dataframe(dfs["wave"])
    return pd.merge(df_wind, df_wave, on="time", how="inner")


@st.cache_data(show_spinner="Loading weather data …")
def load_all_weather(project_zip_bytes, barge_zip_bytes, yard_zip_bytes=None):
    """Load all weather sites.

    Project and Barge are both wind + waves (merged).  Yard is wind only
    (optional — may be None when the yard loading mode is not used).

    Returns: (df_project_weather, df_barge_weather, df_yard_wind)
    """
    df_proj  = _build_merged_site(project_zip_bytes)
    df_barge = _build_merged_site(barge_zip_bytes)
    df_yard  = None
    if yard_zip_bytes is not None:
        dfs_yard = load_weather_from_zip_bytes(yard_zip_bytes, ["wind"])
        df_yard  = prepare_wind_dataframe(dfs_yard["wind"])
    return df_proj, df_barge, df_yard


def add_wind_direction(df):
    df = df.copy()
    u_col = next((c for c in _U_COLS if c in df.columns), None)
    v_col = next((c for c in _V_COLS if c in df.columns), None)
    if u_col and v_col:
        df[u_col] = pd.to_numeric(df[u_col], errors="coerce")
        df[v_col] = pd.to_numeric(df[v_col], errors="coerce")
        df["Wind_Direction"] = (270 - np.degrees(np.arctan2(df[v_col], df[u_col]))) % 360
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – WEATHER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🌦️  2.  Weather Factor Analysis")

st.markdown(
    "Provide weather data using **one** of the two methods below. "
    "The ERA5 API is the primary method — enter your API key in the inputs above and click Download. "
    "Alternatively, upload ZIPs you have already downloaded from Copernicus CDS."
)

# ── Method A: ERA5 API download ───────────────────────────────────────────────
st.subheader("2.1a  Download from ERA5 API  *(primary method)*")

_PROJ_ZIP  = "/tmp/era5_project_location.zip"
_BARGE_ZIP = "/tmp/era5_barge_location.zip"
_YARD_ZIP  = "/tmp/era5_yard_location.zip"

# Variable sets for the two site types.
_WIND_WAVE_VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind",
                   "mean_wave_direction", "mean_wave_period",
                   "significant_height_of_combined_wind_waves_and_swell"]
_WIND_ONLY_VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind"]


def _run_era5_download(api_key, latitude, longitude, variables, target_zip):
    """Download ERA5 timeseries data via CDS API and return the ZIP as bytes."""
    if cdsapi is None:
        raise RuntimeError("cdsapi is not installed. Add 'cdsapi' to requirements.txt.")
    cdsarc = os.path.expanduser("~/.cdsapirc")
    with open(cdsarc, "w") as f:
        f.write(f"url: https://cds.climate.copernicus.eu/api\nkey: {api_key}\n")
    client = cdsapi.Client()
    request = {
        "variable": variables,
        "location": {"longitude": longitude, "latitude": latitude},
        "date": ["1990-01-01/2025-12-31"],
        "data_format": "csv",
        "nocache": str(uuid.uuid4()),
    }
    if os.path.exists(target_zip):
        os.remove(target_zip)
    client.retrieve("reanalysis-era5-single-levels-timeseries", request, target_zip)
    with open(target_zip, "rb") as f:
        return f.read()


_yard_coords_filled = not (abs(float(lat_yard)) < 1e-9 and abs(float(lon_yard)) < 1e-9)

if API_key:
    if st.button("⬇️  Download weather data via ERA5 API"):
        proj_bytes_api  = None
        barge_bytes_api = None
        yard_bytes_api  = None
        with st.spinner("Downloading project location weather (wind + waves) …"):
            try:
                proj_bytes_api = _run_era5_download(
                    API_key, float(lat), float(lon), _WIND_WAVE_VARS, _PROJ_ZIP)
                st.success("✅  Project location weather downloaded (wind + waves).")
            except Exception as e:
                st.error(f"Project weather download failed: {e}")
        with st.spinner("Downloading barge location weather (wind + waves) …"):
            try:
                barge_bytes_api = _run_era5_download(
                    API_key, float(lat_barge), float(lon_barge), _WIND_WAVE_VARS, _BARGE_ZIP)
                st.success("✅  Barge location weather downloaded (wind + waves).")
            except Exception as e:
                st.error(f"Barge weather download failed: {e}")
        if _yard_coords_filled:
            with st.spinner("Downloading yard location weather (wind only) …"):
                try:
                    yard_bytes_api = _run_era5_download(
                        API_key, float(lat_yard), float(lon_yard), _WIND_ONLY_VARS, _YARD_ZIP)
                    st.success("✅  Yard location weather downloaded (wind only).")
                except Exception as e:
                    st.error(f"Yard weather download failed: {e}")
        else:
            st.info("Yard coordinates not filled in — skipping yard download.")
        if proj_bytes_api and barge_bytes_api:
            st.session_state["proj_zip_bytes"]  = proj_bytes_api
            st.session_state["barge_zip_bytes"] = barge_bytes_api
            if yard_bytes_api:
                st.session_state["yard_zip_bytes"] = yard_bytes_api
            st.success("✅  ERA5 downloads complete — weather data is ready.")
else:
    st.info("Enter your ERA5 API key in the Project Inputs panel above, then click Download.")

# ── Method B: Manual ZIP upload ───────────────────────────────────────────────
st.subheader("2.1b  Upload ERA5 ZIPs manually  *(alternative)*")
col_pw, col_bw, col_yw = st.columns(3)
with col_pw:
    project_zip_file = st.file_uploader("Project location ZIP (wind + waves)", type="zip", key="proj_zip")
with col_bw:
    barge_zip_file   = st.file_uploader("Barge location ZIP (wind + waves)",   type="zip", key="barge_zip")
with col_yw:
    yard_zip_file    = st.file_uploader("Yard location ZIP (wind only)",        type="zip", key="yard_zip")

if project_zip_file:
    st.session_state["proj_zip_bytes"]  = project_zip_file.read()
    st.success("✅  Project ZIP uploaded.")
if barge_zip_file:
    st.session_state["barge_zip_bytes"] = barge_zip_file.read()
    st.success("✅  Barge ZIP uploaded.")
if yard_zip_file:
    st.session_state["yard_zip_bytes"]  = yard_zip_file.read()
    st.success("✅  Yard ZIP uploaded.")

# ── Load weather from whichever source provided bytes ────────────────────────
df_project_weather = None
df_barge_weather   = None
df_yard_wind       = None

_proj_bytes  = st.session_state.get("proj_zip_bytes")
_barge_bytes = st.session_state.get("barge_zip_bytes")
_yard_bytes  = st.session_state.get("yard_zip_bytes")

if _proj_bytes and _barge_bytes:
    try:
        df_project_weather, df_barge_weather, df_yard_wind = load_all_weather(
            _proj_bytes, _barge_bytes, _yard_bytes)
        msg = (f"✅  Weather loaded — {len(df_project_weather):,} project rows, "
               f"{len(df_barge_weather):,} barge rows")
        if df_yard_wind is not None:
            msg += f", {len(df_yard_wind):,} yard rows"
        st.success(msg)
    except Exception as e:
        st.error(f"Weather loading error: {e}")

# ── 2.3 Weather plots ─────────────────────────────────────────────────────────
if df_project_weather is not None:
    if st.button("📊  Generate weather statistics & plots"):
        df_project_weather = add_wind_direction(df_project_weather)
        dfw = df_project_weather.copy()
        def season_of(m):
            return ("Winter" if m in [12, 1, 2]
                    else "Spring" if m in [3, 4, 5]
                    else "Summer" if m in [6, 7, 8]
                    else "Autumn")
        dfw["Season"] = dfw["time"].dt.month.apply(season_of)

        def calc_wind(s, lbl):
            return {"Period": lbl, "Avg Wind (m/s)": round(s.mean(), 2),
                    "P75": round(s.quantile(.75), 2),
                    "P90": round(s.quantile(.90), 2),
                    "Max": round(s.max(), 2)}
        ws = [calc_wind(dfw["Adjusted_Windspeed"], "All Data")]
        for ssn in ["Winter", "Spring", "Summer", "Autumn"]:
            ws.append(calc_wind(dfw.loc[dfw["Season"] == ssn, "Adjusted_Windspeed"], ssn))
        st.subheader("Wind Speed Summary"); st.dataframe(pd.DataFrame(ws).set_index("Period"))

        wc = next((c for c in _WAVE_HEIGHT_COLS if c in dfw.columns), None)
        if wc:
            def calc_wave(s, lbl):
                return {"Period": lbl, "Avg Hs (m)": round(s.mean(), 2),
                        "P75": round(s.quantile(.75), 2),
                        "P90": round(s.quantile(.90), 2),
                        "Max": round(s.max(), 2)}
            wvs = [calc_wave(dfw[wc], "All Data")]
            for ssn in ["Winter", "Spring", "Summer", "Autumn"]:
                wvs.append(calc_wave(dfw.loc[dfw["Season"] == ssn, wc], ssn))
            st.subheader("Wave Height Summary"); st.dataframe(pd.DataFrame(wvs).set_index("Period"))

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 5))
            s = dfw["Adjusted_Windspeed"].dropna()
            ax.hist(s, bins=50, density=True, alpha=0.7, edgecolor="black")
            mu, sg = s.mean(), s.std()
            x = np.linspace(s.min(), s.max(), 300)
            if sg > 0:
                ax.plot(x, (1 / (sg * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sg) ** 2), lw=2)
            ax.set(title="Wind Speed Distribution", xlabel="Adjusted Wind Speed (m/s)", ylabel="Density")
            ax.grid(True, alpha=0.3); st.pyplot(fig)
        with c2:
            if wc:
                fig, ax = plt.subplots(figsize=(7, 5))
                s = pd.to_numeric(dfw[wc], errors="coerce").dropna()
                ax.hist(s, bins=50, density=True, alpha=0.7, edgecolor="black")
                mu, sg = s.mean(), s.std()
                x = np.linspace(s.min(), s.max(), 300)
                if sg > 0:
                    ax.plot(x, (1 / (sg * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sg) ** 2), lw=2)
                ax.set(title="Wave Height Distribution",
                       xlabel="Significant Wave Height (m)", ylabel="Density")
                ax.grid(True, alpha=0.3); st.pyplot(fig)

        if wc:
            fig, ax = plt.subplots(figsize=(8, 6))
            hb = ax.hexbin(dfw["Adjusted_Windspeed"], pd.to_numeric(dfw[wc], errors="coerce"),
                           gridsize=60, cmap="plasma", mincnt=1)
            fig.colorbar(hb, ax=ax, label="Point Density")
            ax.set(title="Wind Speed vs Wave Height",
                   xlabel="Adjusted Wind Speed (m/s)", ylabel="Significant Wave Height (m)")
            ax.grid(True, alpha=0.3); st.pyplot(fig)

        if "Wind_Direction" in dfw.columns:
            st.subheader("Wind Rose")
            direction_labels = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            speed_bins   = [0, 0.3, 1.5, 3.3, 5.5, 7.9, 10.7, 13.8,
                            17.1, 20.7, 24.4, 28.4, 32.6, np.inf]
            speed_labels = ["<0.3", "0.3-1.5", "1.5-3.3", "3.3-5.5", "5.5-7.9", "7.9-10.7",
                            "10.7-13.8", "13.8-17.1", "17.1-20.7", "20.7-24.4",
                            "24.4-28.4", "28.4-32.6", ">32.6"]
            wr_df = dfw[["Adjusted_Windspeed", "Wind_Direction"]].copy() \
                       .apply(pd.to_numeric, errors="coerce").dropna()
            shifted_dir = (wr_df["Wind_Direction"] + 11.25) % 360
            dir_edges   = np.arange(0, 382.5, 22.5)
            wr_df["dir_bin"]   = pd.cut(shifted_dir, bins=dir_edges, labels=direction_labels,
                                        include_lowest=True, right=False)
            wr_df["speed_bin"] = pd.cut(wr_df["Adjusted_Windspeed"], bins=speed_bins,
                                        labels=speed_labels, include_lowest=True, right=False)
            freq = pd.crosstab(wr_df["dir_bin"], wr_df["speed_bin"], normalize="all") * 100
            freq = freq.reindex(index=direction_labels, columns=speed_labels, fill_value=0)
            angles = np.deg2rad(np.arange(0, 360, 22.5))
            wr_width = np.deg2rad(22.5 * 0.9)
            fig_wr, ax_wr = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
            wr_bottom = np.zeros(len(direction_labels))
            for sc_lbl in speed_labels:
                vals = freq[sc_lbl].values
                ax_wr.bar(angles, vals, width=wr_width, bottom=wr_bottom, align="center",
                          edgecolor="white", linewidth=0.8, label=sc_lbl)
                wr_bottom += vals
            ax_wr.set_theta_zero_location("N"); ax_wr.set_theta_direction(-1)
            ax_wr.set_xticks(angles); ax_wr.set_xticklabels(direction_labels)
            ax_wr.set_title("Wind Rose", pad=20)
            ax_wr.legend(loc="upper left", bbox_to_anchor=(1.1, 1.1), title="Wind Speed (m/s)")
            st.pyplot(fig_wr)

        if wc:
            st.subheader("Wave Exceedance Table")
            thresholds = np.arange(0.5, float(max_wave) + 0.5, 0.5)
            total_n = len(dfw[wc].dropna())
            exc_rows = [{"Threshold (m)": round(t, 1),
                         "Exceedance %": round((pd.to_numeric(dfw[wc], errors="coerce") > t).sum()
                                               / total_n * 100, 2)} for t in thresholds]
            st.dataframe(pd.DataFrame(exc_rows).set_index("Threshold (m)"))


# ═════════════════════════════════════════════════════════════════════════════
#  SIMULATION FUNCTIONS  (Monopile + Jacket + WTG)
# ═════════════════════════════════════════════════════════════════════════════

# ── Helper: detect whether a sheet uses Loading START/END markers ────────────
def _sheet_has_loading_phase(excel_src, sheet_name):
    raw = _excel_read(excel_src, sheet_name)
    raw.columns = [str(c).strip() for c in raw.columns]
    if "Loading" not in raw.columns:
        return False
    flags = raw["Loading"].map(_norm_marker)
    return ("START" in flags.values) and ("END" in flags.values)


# ── Monopile timeline builder (with Loading + Installation phases) ───────────
def build_mp_timeline(excel_src, sheet_name, carry_cap, total_units, htd=1/24):
    raw = _excel_read(excel_src, sheet_name)
    raw.columns = [str(c).strip() for c in raw.columns]
    desc_col = next(c for c in raw.columns if str(c).strip().startswith("Description"))
    load_flags = raw["Loading"].map(_norm_marker)
    inst_flags = raw["Installation"].map(_norm_marker)
    load_end_idx = int(raw.index[load_flags.eq("END")][0])
    load_phase   = raw.iloc[:load_end_idx + 1].reset_index(drop=True)
    inst_phase   = raw.iloc[load_end_idx + 1:].reset_index(drop=True)
    load_pf = load_phase["Loading"].map(_norm_marker)
    inst_pf = inst_phase["Installation"].map(_norm_marker)
    inst_dec = (int(inst_phase.index[inst_pf.eq("START")][0])
                if len(inst_phase.index[inst_pf.eq("START")]) > 0 else 0)

    def plists(pdf, mser, pname):
        return {"seq" : pd.to_numeric(pdf["N"], errors="coerce").fillna(0).astype(int).tolist(),
                "desc": pdf[desc_col].astype(str).tolist(),
                "dur" : pd.to_numeric(pdf["Seq Dur (hrs)"], errors="coerce").fillna(0).astype(float).tolist(),
                "mark": mser.tolist(),
                "phase": [pname] * len(pdf),
                "wxr": [pname == "installation"] * len(pdf)}

    ld  = plists(load_phase, load_pf, "loading")
    id_ = plists(inst_phase, inst_pf, "installation")
    inventory = 0; units_left = int(total_units); rows = []

    def cl():
        if inventory > 0:
            return math.ceil(units_left / carry_cap) + 1
        return math.ceil(units_left / carry_cap) if units_left > 0 else 0

    def add(sq, ds, dr, ph, wr):
        rows.append({"Sequence": int(sq), "Description": ds, "Phase": ph,
                     "Weather_Restricted": bool(wr),
                     "Seq_Duration_Days": float(dr) * htd,
                     "Inventory": int(inventory),
                     "WTG_Left": int(units_left), "Cycles_Left": int(cl())})

    while units_left > 0 or inventory > 0:
        while inventory < carry_cap and units_left > 0:
            i = 0
            while i < len(ld["seq"]):
                add(ld["seq"][i], ld["desc"][i], ld["dur"][i], ld["phase"][i], ld["wxr"][i])
                if ld["mark"][i] == "START": inventory += 1; units_left -= 1
                if ld["mark"][i] == "END":   break
                i += 1
        while inventory > 0:
            for i in range(len(id_["seq"])):
                add(id_["seq"][i], id_["desc"][i], id_["dur"][i], id_["phase"][i], id_["wxr"][i])
                if i == inst_dec: inventory -= 1
    out = pd.DataFrame(rows); out.insert(0, "N", range(1, len(out) + 1))
    return out


# ── Jacket timeline builder (no loading phase — barge-fed at project site) ───
def build_jacket_timeline(excel_src, sheet_name, total_units, htd=1/24):
    """Jackets are delivered on barges to the project site, so there is no
    loading-yard cycle.  The sheet's Installation column carries START/END
    markers around the per-jacket install cycle.  Rows OUTSIDE the START..END
    block (typically a leading transit row, and a trailing reposition row)
    are executed once per jacket and treated as installation-phase, weather-
    restricted operations.
    """
    raw = _excel_read(excel_src, sheet_name)
    raw.columns = [str(c).strip() for c in raw.columns]
    desc_col = next(c for c in raw.columns if str(c).strip().startswith("Description"))

    inst_flags = raw["Installation"].map(_norm_marker)
    starts = raw.index[inst_flags.eq("START")]
    ends   = raw.index[inst_flags.eq("END")]
    if len(starts) == 0 or len(ends) == 0:
        raise ValueError(f"Jacket sheet '{sheet_name}' must have Installation START and END markers.")
    start_idx = int(starts[0])
    end_idx   = int(ends[0])

    # The whole sheet is the per-jacket cycle (transit-in → install → reposition).
    # We repeat it total_units times.
    cycle = raw.reset_index(drop=True)

    seqs  = pd.to_numeric(cycle["N"], errors="coerce").fillna(0).astype(int).tolist()
    descs = cycle[desc_col].astype(str).tolist()
    durs  = pd.to_numeric(cycle["Seq Dur (hrs)"], errors="coerce").fillna(0).astype(float).tolist()

    inventory = 0; units_left = int(total_units); rows = []

    def cl():
        return units_left if units_left > 0 else 0

    def add(sq, ds, dr):
        rows.append({"Sequence": int(sq), "Description": ds, "Phase": "installation",
                     "Weather_Restricted": True,
                     "Seq_Duration_Days": float(dr) * htd,
                     "Inventory": int(inventory),
                     "WTG_Left": int(units_left), "Cycles_Left": int(cl())})

    for _ in range(int(total_units)):
        inventory = 1
        for i in range(len(cycle)):
            add(seqs[i], descs[i], durs[i])
            if i == start_idx: inventory = 1   # picked up jacket from barge
            if i == end_idx:   inventory = 0   # jacket installed
        units_left -= 1

    out = pd.DataFrame(rows); out.insert(0, "N", range(1, len(out) + 1))
    return out


# ── Foundation weather impacts (works for both MP and Jacket sheets) ─────────
def simulate_weather_impacts_foundation(timeline_df, weather_df, excel_src, sheet_name,
                                         transit_distance, transit_speed, simulations=1,
                                         eligible_indices=None, seed=42,
                                         max_wait_hours=5000, htd=1/24):
    """Project-site weather sim for foundation installation (MP or Jacket).
    Same logic as the original simulate_weather_impacts_mp — both sheet styles
    share the same column shape (N / Seq Dur / Tr / Transit / Oplim Wave / Oplim Wind)."""
    transit_hours = float(transit_distance) / float(transit_speed)
    raw = _excel_read(excel_src, sheet_name)
    raw.columns = [str(c).strip() for c in raw.columns]
    op = raw[["N", "Seq Dur (hrs)", "Tr (hrs)", "Transit",
              "Oplim Waves (m)", "Oplim Wind (m/s)"]].copy()
    for c in ["N", "Seq Dur (hrs)", "Tr (hrs)"]:
        op[c] = pd.to_numeric(op[c], errors="coerce")
    op["Transit"]          = op["Transit"].map(_norm_bool)
    op["Oplim Waves (m)"]  = pd.to_numeric(op["Oplim Waves (m)"], errors="coerce").fillna(np.inf)
    op["Oplim Wind (m/s)"] = pd.to_numeric(op["Oplim Wind (m/s)"], errors="coerce").fillna(np.inf)
    nt = ~op["Transit"]
    op.loc[nt, "Seq Dur (hrs)"] = op.loc[nt, "Seq Dur (hrs)"].fillna(op.loc[nt, "Tr (hrs)"])
    op.loc[nt, "Tr (hrs)"]      = op.loc[nt, "Tr (hrs)"].fillna(op.loc[nt, "Seq Dur (hrs)"])
    op.loc[op["Transit"], ["Seq Dur (hrs)", "Tr (hrs)"]] = transit_hours
    op = op.dropna(subset=["N", "Seq Dur (hrs)", "Tr (hrs)"]).drop_duplicates(subset=["N"])
    op["N"] = op["N"].astype(int)
    lm  = dict(zip(op["N"], np.minimum(op["Seq Dur (hrs)"], op["Tr (hrs)"])))
    hm  = dict(zip(op["N"], np.maximum(op["Seq Dur (hrs)"], op["Tr (hrs)"])))
    wvm = dict(zip(op["N"], op["Oplim Waves (m)"]))
    wdm = dict(zip(op["N"], op["Oplim Wind (m/s)"]))
    tm  = dict(zip(op["N"], op["Transit"]))
    base = timeline_df[["N", "Sequence", "Inventory", "WTG_Left", "Cycles_Left",
                        "Weather_Restricted", "Phase"]].copy()
    base["Sequence"] = base["Sequence"].astype(int)
    base["Weather_Restricted"] = base["Weather_Restricted"].astype(bool)

    wave_col = _find_col(weather_df, _WAVE_HEIGHT_COLS, "wave height")
    w = weather_df[["time", wave_col, "Adjusted_Windspeed"]].copy()
    w["time"] = pd.to_datetime(w["time"], errors="coerce")
    for c in [wave_col, "Adjusted_Windspeed"]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.dropna().sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
    wt  = w["time"].to_numpy(dtype="datetime64[ns]")
    ww  = w[wave_col].to_numpy(dtype=float)
    wnd = w["Adjusted_Windspeed"].to_numpy(dtype=float)
    eli = eligible_indices if eligible_indices is not None else np.arange(len(wt))
    rng = np.random.default_rng(seed); n = len(base)
    tot_m = np.zeros((n, simulations), dtype=float)
    dwn_m = np.zeros((n, simulations), dtype=float)
    seqs = base["Sequence"].to_numpy(dtype=int)
    sl  = np.array([lm[s]  for s in seqs], dtype=float)
    sh  = np.array([hm[s]  for s in seqs], dtype=float)
    swl = np.array([wvm[s] for s in seqs], dtype=float)
    swd = np.array([wdm[s] for s in seqs], dtype=float)
    sit = np.array([tm[s]  for s in seqs], dtype=bool)
    swx = base["Weather_Restricted"].to_numpy(dtype=bool)

    def wait(start, wlim, wndlim):
        idx = int(np.searchsorted(wt, np.datetime64(pd.Timestamp(start)), side="left"))
        for h in range(max_wait_hours + 1):
            i = idx + h
            if i >= len(wt): raise RuntimeError("Ran out of weather data.")
            if ww[i] <= wlim and wnd[i] <= wndlim: return h, i
        raise RuntimeError("Exceeded max_wait_hours.")

    summ = []
    for si in range(simulations):
        ct = pd.Timestamp(wt[int(rng.choice(eli))])
        ss = ct; ta = td = 0.0
        for ri in range(n):
            if swx[ri]:
                dh, pi = wait(ct, swl[ri], swd[ri])
                op_s = pd.Timestamp(wt[pi])
            else:
                dh, op_s = 0.0, ct
            ah = transit_hours if sit[ri] else float(rng.uniform(sl[ri], sh[ri]))
            tot_m[ri, si] = (dh + ah) * htd
            dwn_m[ri, si] = dh * htd
            ta += ah; td += dh
            ct = op_s + pd.Timedelta(hours=ah)
        summ.append({"Simulation": f"Sim{si+1}", "Start_Date": ss, "Finish_Date": ct,
                     "Total_Active_Days":   ta * htd,
                     "Total_Downtime_Days": td * htd,
                     "Total_Project_Days":  (ta + td) * htd})
    sc = [f"Sim{i}" for i in range(1, simulations + 1)]
    return (pd.concat([base, pd.DataFrame(tot_m, columns=sc, index=base.index)], axis=1),
            pd.concat([base, pd.DataFrame(dwn_m, columns=sc, index=base.index)], axis=1),
            pd.DataFrame(summ))


def get_seasonal_eligible_indices(weather_times, start_date_str,
                                   window_days=7, latest_start="2022-12-31 23:00:00"):
    target = pd.Timestamp(start_date_str)
    times_pd = pd.DatetimeIndex(weather_times); cutoff = pd.Timestamp(latest_start)
    doys = times_pd.day_of_year; tdoy = target.day_of_year
    circ = np.minimum(np.abs(doys - tdoy), 365 - np.abs(doys - tdoy))
    eligible = np.where((circ <= window_days) & (times_pd <= cutoff))[0]
    if len(eligible) == 0:
        raise ValueError(f"No eligible starts near {target.strftime('%d %B')}.")
    return eligible


# ── W90 WTG timeline (new per-turbine parallel-loading model) ────────────────
def build_w90_timeline(excel_src, sheet_name, carry_cap, total_wtg, htd=1/24,
                       dock_distance=None, transit_speed=None):
    """Build the W90 WTG installation timeline under the v8 loading model.

    Loading model (per latest W90 WTG sheet):
      • There may be pre-loading rows before the Loading START marker — these
        run ONCE per loading cycle (not per turbine).
      • The parallel-loading block is the set of rows (between Loading START and
        Loading END) flagged Parallell 1 and/or Parallell 2.  Loading is now done
        one turbine at a time: the parallel block is repeated once per turbine
        loaded (i.e. min(carry_cap, wtg_left) times) before the vessel transits.
      • Parallell 1 and Parallell 2 are treated as taking the same time, so the
        per-turbine loading duration uses the Parallell-1 hourly durations.  Each
        Parallell-1 row is paired 1:1 (by order) with a Parallell-2 row; the two
        descriptions are combined and the Parallell-1 duration is used.
      • Any non-parallel rows inside the loading block (rare) run once per cycle.

    Installation is unchanged: the per-turbine install cycle (Installation
    START..END) repeats once for each turbine in inventory; transit rows use
    dock_distance / transit_speed.
    """
    raw = _excel_read(excel_src, sheet_name); raw.columns = [str(c).strip() for c in raw.columns]
    lf  = raw["Loading"].map(_norm_marker); if_ = raw["Installation"].map(_norm_marker)
    lsi = int(lf[lf.eq("START")].index[0])
    lei = int(lf[lf.eq("END")].index[0])
    isi = int(if_[if_.eq("START")].index[0])
    iei = int(if_[if_.eq("END")].index[0])

    # Loading-cycle preamble = rows BEFORE Loading START (run once per cycle).
    pre = raw.iloc[:lsi].reset_index(drop=True)
    # Loading block = Loading START .. END (the parallel-loading section).
    lp  = raw.iloc[lsi:lei + 1].reset_index(drop=True)
    # Install block = Installation START .. END.
    ip  = raw.iloc[isi:iei + 1].reset_index(drop=True)
    ipf = ip["Installation"].map(_norm_marker)
    isp = int(ipf[ipf.eq("START")].index[0])

    def _row_dur(pdf, i):
        ts = _norm_bool(pdf.iloc[i]["Transit"])
        if ts:
            return dock_distance / transit_speed
        return float(pd.to_numeric(pdf.iloc[i]["Seq Dur (hrs)"], errors="coerce") or 0.0)

    def plists(pdf, mser):
        return {"seq" : pd.to_numeric(pdf["N"], errors="coerce").fillna(0).astype(int).tolist(),
                "desc": pdf["Description"].astype(str).tolist(),
                "dur" : [_row_dur(pdf, i) for i in range(len(pdf))],
                "mark": mser.tolist(),
                "p1"  : pdf["Parallell  1"].map(_norm_bool).tolist(),
                "p2"  : pdf["Parallell  2"].map(_norm_bool).tolist()}

    pre_l = plists(pre, pre["Loading"].map(_norm_marker)) if len(pre) else None
    ld    = plists(lp, lp["Loading"].map(_norm_marker))
    id_   = plists(ip, ipf)

    # Build the loading-step template (one collapsed row per parallel pair).
    # Parallell-1 rows are paired by order with Parallell-2 rows; non-parallel
    # rows inside the loading block become standalone once-per-cycle rows.
    p1_idx = [i for i in range(len(ld["seq"])) if ld["p1"][i]]
    p2_idx = [i for i in range(len(ld["seq"])) if ld["p2"][i]]

    def _clean_desc(d):
        d = str(d).strip()
        return "" if d.lower() in {"nan", "none"} else d

    # Per-turbine loading steps (paired P1/P2; duration from P1).
    per_turbine_steps = []
    n_pairs = max(len(p1_idx), len(p2_idx))
    for k in range(n_pairs):
        i1 = p1_idx[k] if k < len(p1_idx) else None
        i2 = p2_idx[k] if k < len(p2_idx) else None
        dur = ld["dur"][i1] if i1 is not None else ld["dur"][i2]      # P1 duration
        d1  = _clean_desc(ld["desc"][i1]) if i1 is not None else ""
        d2  = _clean_desc(ld["desc"][i2]) if i2 is not None else ""
        if d1 and d2:
            desc = f"{d1}  ||  {d2}"
        else:
            desc = d1 or d2 or "Parallel loading"
        seq = ld["seq"][i1] if i1 is not None else ld["seq"][i2]
        per_turbine_steps.append({
            "seq": int(seq), "desc": desc, "dur": float(dur),
            "p1_seq": int(ld["seq"][i1]) if i1 is not None else None,
            "p2_seq": int(ld["seq"][i2]) if i2 is not None else None,
            "p1_dur": float(ld["dur"][i1]) if i1 is not None else 0.0,
            "p2_dur": float(ld["dur"][i2]) if i2 is not None else 0.0,
        })

    # Non-parallel rows inside the loading block (once per cycle).
    once_loading_steps = [
        {"seq": int(ld["seq"][i]), "desc": _clean_desc(ld["desc"][i]) or "Loading step",
         "dur": float(ld["dur"][i])}
        for i in range(len(ld["seq"])) if not (ld["p1"][i] or ld["p2"][i])
    ]

    inventory = 0; wtg_left = int(total_wtg); rows = []

    def cl():
        if inventory > 0: return math.ceil(wtg_left / carry_cap) + 1
        return math.ceil(wtg_left / carry_cap) if wtg_left > 0 else 0

    def add(sq, ds, dr, is_par, p1_seq=None, p2_seq=None, p1_dur=0.0, p2_dur=0.0):
        rows.append({"Sequence": int(sq), "Description": ds,
                     "Seq_Duration_Hours": float(dr) * htd,
                     "Inventory": int(inventory),
                     "WTG_Left": int(wtg_left), "Cycles_Left": int(cl()),
                     "Parallel_Block": bool(is_par),
                     "Parallel_Branch_1_Hours": float(p1_dur) * htd,
                     "Parallel_Branch_2_Hours": float(p2_dur) * htd,
                     "Source_Sequences": (", ".join(str(s) for s in [p1_seq, p2_seq] if s is not None)
                                          if is_par else str(int(sq)))})

    while wtg_left > 0 or inventory > 0:
        if inventory == 0 and wtg_left > 0:
            # 1) Pre-loading preamble — once per cycle.
            if pre_l is not None:
                for i in range(len(pre_l["seq"])):
                    add(pre_l["seq"][i], _clean_desc(pre_l["desc"][i]) or "Pre-loading",
                        pre_l["dur"][i], False)
            # 2) Load one turbine at a time (parallel block repeats per turbine).
            to_load = min(carry_cap, wtg_left)
            for _t in range(to_load):
                for step in per_turbine_steps:
                    add(step["seq"], step["desc"], step["dur"], True,
                        p1_seq=step["p1_seq"], p2_seq=step["p2_seq"],
                        p1_dur=step["p1_dur"], p2_dur=step["p2_dur"])
                inventory += 1; wtg_left -= 1
            # 3) Any once-per-cycle non-parallel loading rows.
            for step in once_loading_steps:
                add(step["seq"], step["desc"], step["dur"], False)
        # 4) Install loop — per turbine in inventory.
        while inventory > 0:
            i = 0
            while i < len(id_["seq"]):
                add(id_["seq"][i], _clean_desc(id_["desc"][i]) or "Install step",
                    id_["dur"][i], False)
                if id_["mark"][i] == "START": inventory -= 1
                if id_["mark"][i] == "END":
                    if inventory > 0: i = isp; continue
                    else: break
                i += 1

    out = pd.DataFrame(rows); out.insert(0, "N", range(1, len(out) + 1))
    return out


def add_duration_simulations_w90(timeline_df, excel_src, sheet_name, simulations=1, seed=42,
                                  htd=1/24, transit_distance=None, transit_speed=None):
    """No-weather duration simulator for W90 WTG (v8 paired-loading model).

    Each timeline row is a single operation.  For a parallel-loading pair the
    timeline row's Sequence is the Parallell-1 sequence and its duration is drawn
    from that Parallell-1 row's Seq Dur range (P1 and P2 are treated as equal
    time, so the P1 duration represents the parallel step).  Transit rows use
    transit_distance / transit_speed."""
    raw = _excel_read(excel_src, sheet_name); raw.columns = [str(c).strip() for c in raw.columns]
    dur = raw[["N", "Seq Dur (hrs)", "Tr (hrs)", "Transit"]].copy()
    dur["N"]             = pd.to_numeric(dur["N"], errors="coerce")
    dur["Seq Dur (hrs)"] = pd.to_numeric(dur["Seq Dur (hrs)"], errors="coerce")
    dur["Tr (hrs)"]      = pd.to_numeric(dur["Tr (hrs)"], errors="coerce")
    dur["Transit"]       = dur["Transit"].map(_norm_bool)
    dur = dur.dropna(subset=["N"]).drop_duplicates(subset=["N"])
    dur["N"] = dur["N"].astype(int)
    tm  = dict(zip(dur["N"], dur["Transit"]))
    bl  = dict(zip(dur["N"], dur[["Seq Dur (hrs)", "Tr (hrs)"]].min(axis=1)))
    bh  = dict(zip(dur["N"], dur[["Seq Dur (hrs)", "Tr (hrs)"]].max(axis=1)))
    df = timeline_df.copy()
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Sim")])

    def _row_seq(row):
        # For a parallel pair, the first source sequence is the Parallell-1 row,
        # whose duration governs the (equal-time) parallel step.
        if bool(row["Parallel_Block"]):
            parts = [s.strip() for s in str(row["Source_Sequences"]).split(",") if s.strip()]
            return int(parts[0]) if parts else int(row["Sequence"])
        return int(row["Sequence"])

    seqs = [_row_seq(row) for _, row in df.iterrows()]
    lm  = {s: (transit_distance / transit_speed if tm.get(s) else float(bl.get(s, 0.0))) for s in set(seqs)}
    hm_ = {s: (transit_distance / transit_speed if tm.get(s) else float(bh.get(s, 0.0))) for s in set(seqs)}
    rng = np.random.default_rng(seed)
    mat = np.empty((len(df), simulations), dtype=float)
    for ri, s in enumerate(seqs):
        lo, hi = lm[s], hm_[s]
        if hi <= lo:
            mat[ri, :] = lo * htd
        else:
            mat[ri, :] = rng.uniform(lo, hi, size=simulations) * htd
    return pd.concat([df.copy(),
                      pd.DataFrame(mat, columns=[f"Sim{i}" for i in range(1, simulations + 1)],
                                   index=df.index)], axis=1)


def build_sequential_wtg_timeline(excel_src, sheet_name, carry_cap, total_wtg, htd=1/24,
                                   dock_distance=None, transit_speed=None):
    raw = _excel_read(excel_src, sheet_name); raw.columns = [str(c).strip() for c in raw.columns]
    lf  = raw["Loading"].map(_norm_marker); if_ = raw["Installation"].map(_norm_marker)
    isi = int(if_[if_.eq("START")].index[0])
    iec = if_[if_.eq("END")].index
    iei = int(iec[0]) if len(iec) > 0 else int(raw.index[-1])
    lp  = raw.iloc[:isi].reset_index(drop=True)
    ip  = raw.iloc[isi:iei + 1].reset_index(drop=True)
    lpf = lp["Loading"].map(_norm_marker); ipf = ip["Installation"].map(_norm_marker)

    def plists(pdf, mser):
        ts = pdf["Transit"].map(_norm_bool)
        durs = [dock_distance / transit_speed if ts.iloc[i]
                else float(pd.to_numeric(pdf.iloc[i]["Seq Dur (hrs)"], errors="coerce") or 0.0)
                for i in range(len(pdf))]
        return {"seq" : pd.to_numeric(pdf["N"], errors="coerce").fillna(0).astype(int).tolist(),
                "desc": pdf["Description"].astype(str).tolist(),
                "dur" : durs,
                "mark": mser.tolist()}

    ld = plists(lp, lpf); id_ = plists(ip, ipf)
    inventory = 0; wtg_left = int(total_wtg); rows = []

    def cl():
        if inventory > 0: return math.ceil(wtg_left / carry_cap) + 1
        return math.ceil(wtg_left / carry_cap) if wtg_left > 0 else 0

    def add(sq, ds, dr):
        rows.append({"Sequence": int(sq), "Description": ds,
                     "Seq_Duration_Hours": float(dr) * htd,
                     "Inventory": int(inventory),
                     "WTG_Left": int(wtg_left), "Cycles_Left": int(cl())})

    while wtg_left > 0 or inventory > 0:
        while inventory < carry_cap and wtg_left > 0:
            i = 0
            while i < len(ld["seq"]):
                add(ld["seq"][i], ld["desc"][i], ld["dur"][i])
                if ld["mark"][i] == "START": inventory += 1; wtg_left -= 1
                if ld["mark"][i] == "END": break
                i += 1
        while inventory > 0:
            i = 0
            while i < len(id_["seq"]):
                add(id_["seq"][i], id_["desc"][i], id_["dur"][i])
                if id_["mark"][i] == "START": inventory -= 1
                if id_["mark"][i] == "END": break
                i += 1
    out = pd.DataFrame(rows); out.insert(0, "N", range(1, len(out) + 1))
    return out


def add_duration_simulations_sequential(timeline_df, excel_src, sheet_name, simulations=1,
                                         dock_distance=0, transit_speed=1, seed=42, htd=1/24):
    raw = _excel_read(excel_src, sheet_name); raw.columns = [str(c).strip() for c in raw.columns]
    dur = raw[["N", "Seq Dur (hrs)", "Tr (hrs)", "Transit"]].copy()
    dur["N"]             = pd.to_numeric(dur["N"], errors="coerce")
    dur["Seq Dur (hrs)"] = pd.to_numeric(dur["Seq Dur (hrs)"], errors="coerce")
    dur["Tr (hrs)"]      = pd.to_numeric(dur["Tr (hrs)"], errors="coerce")
    dur["Transit"]       = dur["Transit"].map(_norm_bool)
    dur = dur.dropna(subset=["N"]).drop_duplicates(subset=["N"])
    dur["N"] = dur["N"].astype(int)
    td_days = (dock_distance / transit_speed) * htd
    lm = {}; hm = {}; fm = {}
    for _, row in dur.iterrows():
        s = int(row["N"])
        if row["Transit"]: fm[s] = float(td_days)
        else:
            lm[s] = float(min(row["Seq Dur (hrs)"], row["Tr (hrs)"])) * htd
            hm[s] = float(max(row["Seq Dur (hrs)"], row["Tr (hrs)"])) * htd
    df = timeline_df.copy()
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Sim")])
    df["_l"] = df["Sequence"].map(lm)
    df["_h"] = df["Sequence"].map(hm)
    df["_f"] = df["Sequence"].map(fm)
    lows = df["_l"].to_numpy(dtype=float)
    highs = df["_h"].to_numpy(dtype=float)
    fixeds = df["_f"].to_numpy(dtype=float)
    tmask = ~np.isnan(fixeds); ntmask = ~tmask
    rng = np.random.default_rng(seed)
    mat = np.empty((len(df), simulations), dtype=float)
    mat[tmask, :]  = fixeds[tmask, None]
    mat[ntmask, :] = rng.uniform(lows[ntmask, None], highs[ntmask, None],
                                  size=(ntmask.sum(), simulations))
    return pd.concat([df.drop(columns=["_l", "_h", "_f"]),
                      pd.DataFrame(mat, columns=[f"Sim{i}" for i in range(1, simulations + 1)],
                                   index=df.index)], axis=1)


def simulate_weather_impacts_w90(timeline_df, project_weather_df, barge_weather_df,
                                  yard_wind_df, excel_src, sheet_name,
                                  transit_distance, transit_speed,
                                  simulations=1, eligible_indices=None, seed=42,
                                  max_wait_hours=5000, htd=1/24, mp_offset_days=None,
                                  loading_mode="Yard"):
    """Weather sim for W90 WTG (v8 deterministic per-phase routing).

    Weather routing:
      • Loading rows (Loading START..END):
          – loading_mode == 'Yard'      → YARD weather, WIND ONLY (no waves)
          – loading_mode == 'Nearshore' → BARGE weather (wind + waves)
          – loading_mode == 'Offshore'  → PROJECT weather (wind + waves)
      • Installation rows → PROJECT weather (wind + waves)
      • Transit rows      → PROJECT weather (wind + waves)

    The timeline_df is the v8 paired-loading timeline: each row is a single
    operation, and a parallel-loading pair carries its Parallell-1 sequence first
    in Source_Sequences (its duration / oplims govern the equal-time step).
    """
    transit_hours = float(transit_distance) / float(transit_speed)
    raw = _excel_read(excel_src, sheet_name); raw.columns = [str(c).strip() for c in raw.columns]
    op = raw[["N", "Seq Dur (hrs)", "Tr (hrs)", "Transit", "Loading", "Installation",
              "Oplim Waves (m)", "Oplim Wind (m/s)"]].copy()
    op["N"] = pd.to_numeric(op["N"], errors="coerce")
    for c in ["Seq Dur (hrs)", "Tr (hrs)"]:
        op[c] = pd.to_numeric(op[c], errors="coerce")
    op["Transit"]      = op["Transit"].map(_norm_bool)
    op["Loading"]      = op["Loading"].map(_norm_marker)
    op["Installation"] = op["Installation"].map(_norm_marker)
    op["Oplim Waves (m)"]  = pd.to_numeric(op["Oplim Waves (m)"], errors="coerce").fillna(np.inf)
    op["Oplim Wind (m/s)"] = pd.to_numeric(op["Oplim Wind (m/s)"], errors="coerce").fillna(np.inf)
    nt = ~op["Transit"]
    op.loc[nt, "Seq Dur (hrs)"] = op.loc[nt, "Seq Dur (hrs)"].fillna(op.loc[nt, "Tr (hrs)"])
    op.loc[nt, "Tr (hrs)"]      = op.loc[nt, "Tr (hrs)"].fillna(op.loc[nt, "Seq Dur (hrs)"])
    op.loc[op["Transit"], ["Seq Dur (hrs)", "Tr (hrs)"]] = transit_hours
    op = op.dropna(subset=["N", "Seq Dur (hrs)", "Tr (hrs)"]).drop_duplicates(subset="N")
    op["N"] = op["N"].astype(int)

    # Determine the set of sequences that belong to the loading phase
    # (Loading START .. Loading END inclusive).
    ls = op.index[op["Loading"].eq("START")]
    le = op.index[op["Loading"].eq("END")]
    if len(ls) > 0 and len(le) > 0:
        loading_seq_set = set(op.loc[int(ls[0]):int(le[0]), "N"].tolist())
    else:
        loading_seq_set = set()

    lm  = dict(zip(op["N"], np.minimum(op["Seq Dur (hrs)"], op["Tr (hrs)"])))
    hm  = dict(zip(op["N"], np.maximum(op["Seq Dur (hrs)"], op["Tr (hrs)"])))
    wvm = dict(zip(op["N"], op["Oplim Waves (m)"]))
    wdm = dict(zip(op["N"], op["Oplim Wind (m/s)"]))
    tm  = dict(zip(op["N"], op["Transit"]))

    base = timeline_df[["N", "Sequence", "Inventory", "WTG_Left", "Cycles_Left",
                        "Parallel_Block", "Source_Sequences"]].copy()
    base["Sequence"] = pd.to_numeric(base["Sequence"], errors="coerce").astype(int)

    # Classify each timeline row's weather phase.
    #   loading  → uses the selected loading-location weather
    #   project  → installation + transit rows → project weather
    rspecs = []
    for _, row in base.iterrows():
        is_par = bool(row["Parallel_Block"])
        seqs = ([int(s.strip()) for s in str(row["Source_Sequences"]).split(",") if s.strip()]
                if is_par else [int(row["Sequence"])])
        gov = seqs[0]  # governing sequence (Parallell-1 for pairs)
        is_transit = bool(tm.get(gov, False))
        is_loading = (gov in loading_seq_set) and not is_transit
        phase_type = "loading" if is_loading else "project"
        rspecs.append({"is_parallel": is_par, "gov": gov, "is_transit": is_transit,
                       "wave_lim": float(wvm.get(gov, np.inf)),
                       "wind_lim": float(wdm.get(gov, np.inf)),
                       "phase_type": phase_type})

    wave_col = _find_col(project_weather_df, _WAVE_HEIGHT_COLS, "wave height")

    def prep(df, cols):
        df = df[list(cols)].copy(); df["time"] = pd.to_datetime(df["time"], errors="coerce")
        for c in cols:
            if c != "time": df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna().sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)

    # Project weather (installation + transit, and offshore loading).
    wp = prep(project_weather_df, {"time", wave_col, "Adjusted_Windspeed"})
    pt  = wp["time"].to_numpy(dtype="datetime64[ns]")
    pw  = wp[wave_col].to_numpy(dtype=float)
    pnd = wp["Adjusted_Windspeed"].to_numpy(dtype=float)

    # Loading-location weather dataset depends on loading_mode.
    if loading_mode == "Offshore":
        # Loaded at the project site → wind + waves (reuse project arrays).
        lt, lnd, lw_waves, loading_check_waves = pt, pnd, pw, True
    elif loading_mode == "Nearshore":
        # Loaded at the barge site → wind + waves.
        if barge_weather_df is None:
            raise ValueError("Nearshore loading selected but no barge weather is available.")
        bwave_col = _find_col(barge_weather_df, _WAVE_HEIGHT_COLS, "wave height")
        wb = prep(barge_weather_df, {"time", bwave_col, "Adjusted_Windspeed"})
        lt       = wb["time"].to_numpy(dtype="datetime64[ns]")
        lnd      = wb["Adjusted_Windspeed"].to_numpy(dtype=float)
        lw_waves = wb[bwave_col].to_numpy(dtype=float)
        loading_check_waves = True
    else:  # "Yard"
        # Loaded at the yard → WIND ONLY.
        if yard_wind_df is None:
            raise ValueError("Yard loading selected but no yard weather is available.")
        wy = prep(yard_wind_df, {"time", "Adjusted_Windspeed"})
        lt       = wy["time"].to_numpy(dtype="datetime64[ns]")
        lnd      = wy["Adjusted_Windspeed"].to_numpy(dtype=float)
        lw_waves = None
        loading_check_waves = False

    eli = eligible_indices if eligible_indices is not None else np.arange(len(pt))
    rng = np.random.default_rng(seed); n = len(base)
    tot_m = np.zeros((n, simulations), dtype=float)
    dwn_m = np.zeros((n, simulations), dtype=float)

    def _wait(times, winds, waves, start, wlim, wndlim, check_w):
        idx = int(np.searchsorted(times, np.datetime64(pd.Timestamp(start)), side="left"))
        for h in range(max_wait_hours + 1):
            i = idx + h
            if i >= len(times): raise RuntimeError("Ran out of weather data.")
            if winds[i] <= wndlim and (not check_w or waves[i] <= wlim):
                return h, i
        raise RuntimeError("Exceeded max_wait_hours.")

    def _draw(spec):
        s = spec["gov"]
        return transit_hours if tm.get(s) else float(rng.uniform(lm[s], hm[s]))

    summ = []
    for si in range(simulations):
        sidx = int(rng.choice(eli)); bs = pd.Timestamp(pt[sidx])
        offset = float(mp_offset_days[si]) if mp_offset_days is not None else 0.0
        ct = bs + pd.Timedelta(days=offset); ss = ct; ta = td = 0.0
        for ri, spec in enumerate(rspecs):
            if spec["phase_type"] == "loading":
                dh, pi = _wait(lt, lnd, lw_waves, ct,
                               spec["wave_lim"], spec["wind_lim"], loading_check_waves)
                op_s = pd.Timestamp(lt[pi])
            else:  # installation + transit → project weather
                dh, pi = _wait(pt, pnd, pw, ct,
                               spec["wave_lim"], spec["wind_lim"], True)
                op_s = pd.Timestamp(pt[pi])
            ah = _draw(spec)
            tot_m[ri, si] = (dh + ah) * htd
            dwn_m[ri, si] = dh * htd
            ta += ah; td += dh
            ct = op_s + pd.Timedelta(hours=ah)
        summ.append({"Simulation": f"Sim{si+1}", "WTG_Start_Date": ss,
                     "Finish_Date": ct, "MP_Offset_Days": offset,
                     "Total_Active_Days":   ta * htd,
                     "Total_Downtime_Days": td * htd,
                     "Total_Project_Days":  (ta + td) * htd})
    sc = [f"Sim{i}" for i in range(1, simulations + 1)]
    df_out = base[["N", "Sequence", "Inventory", "WTG_Left", "Cycles_Left"]].copy()
    return (pd.concat([df_out, pd.DataFrame(tot_m, columns=sc, index=df_out.index)], axis=1),
            pd.concat([df_out, pd.DataFrame(dwn_m, columns=sc, index=df_out.index)], axis=1),
            pd.DataFrame(summ))


def simulate_sequential_wtg_weather(timeline_df, project_weather_df, barge_weather_df,
                                     yard_wind_df, excel_src, sheet_name,
                                     transit_distance, transit_speed, simulations=1,
                                     eligible_indices=None, seed=42, max_wait_hours=5000,
                                     htd=1/24, mp_offset_days=None, loading_mode="Yard"):
    """Sequential WTG weather sim (JUV / FIV) with v8 per-phase weather routing.

    Weather routing (same rule as the W90 sim):
      • Loading rows (Loading START..END):
          – 'Yard'      → YARD weather, WIND ONLY
          – 'Nearshore' → BARGE weather (wind + waves)
          – 'Offshore'  → PROJECT weather (wind + waves)
      • Installation rows → PROJECT weather (wind + waves)
      • Transit rows      → PROJECT weather (wind + waves)
    """
    transit_hours = float(transit_distance) / float(transit_speed)
    raw = _excel_read(excel_src, sheet_name); raw.columns = [str(c).strip() for c in raw.columns]
    op = raw[["N", "Seq Dur (hrs)", "Tr (hrs)", "Transit", "Loading", "Installation",
              "Oplim Waves (m)", "Oplim Wind (m/s)"]].copy()
    op["N"] = pd.to_numeric(op["N"], errors="coerce")
    for c in ["Seq Dur (hrs)", "Tr (hrs)"]:
        op[c] = pd.to_numeric(op[c], errors="coerce")
    op["Transit"]      = op["Transit"].map(_norm_bool)
    op["Loading"]      = op["Loading"].map(_norm_marker)
    op["Installation"] = op["Installation"].map(_norm_marker)
    op["Oplim Waves (m)"]  = pd.to_numeric(op["Oplim Waves (m)"], errors="coerce").fillna(np.inf)
    op["Oplim Wind (m/s)"] = pd.to_numeric(op["Oplim Wind (m/s)"], errors="coerce").fillna(np.inf)
    nt = ~op["Transit"]
    op.loc[nt, "Seq Dur (hrs)"] = op.loc[nt, "Seq Dur (hrs)"].fillna(op.loc[nt, "Tr (hrs)"])
    op.loc[nt, "Tr (hrs)"]      = op.loc[nt, "Tr (hrs)"].fillna(op.loc[nt, "Seq Dur (hrs)"])
    op.loc[op["Transit"], ["Seq Dur (hrs)", "Tr (hrs)"]] = transit_hours
    op = op.dropna(subset=["N", "Seq Dur (hrs)", "Tr (hrs)"]).drop_duplicates(subset="N")
    op["N"] = op["N"].astype(int)

    # Sequences belonging to the loading phase (Loading START..END inclusive).
    ls = op.index[op["Loading"].eq("START")]
    le = op.index[op["Loading"].eq("END")]
    if len(ls) > 0 and len(le) > 0:
        loading_seq_set = set(op.loc[int(ls[0]):int(le[0]), "N"].tolist())
    else:
        loading_seq_set = set()

    lm  = dict(zip(op["N"], np.minimum(op["Seq Dur (hrs)"], op["Tr (hrs)"])))
    hm  = dict(zip(op["N"], np.maximum(op["Seq Dur (hrs)"], op["Tr (hrs)"])))
    wvm = dict(zip(op["N"], op["Oplim Waves (m)"]))
    wdm = dict(zip(op["N"], op["Oplim Wind (m/s)"]))
    tm  = dict(zip(op["N"], op["Transit"]))
    base = timeline_df[["N", "Sequence", "Inventory", "WTG_Left", "Cycles_Left"]].copy()
    base["Sequence"] = pd.to_numeric(base["Sequence"], errors="coerce").astype(int)

    def prep_ww(df):
        wc = _find_col(df, _WAVE_HEIGHT_COLS, "wave height")
        w = df[["time", wc, "Adjusted_Windspeed"]].copy()
        w["time"] = pd.to_datetime(w["time"], errors="coerce")
        for c in [wc, "Adjusted_Windspeed"]:
            w[c] = pd.to_numeric(w[c], errors="coerce")
        w = w.dropna().sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
        return (w["time"].to_numpy(dtype="datetime64[ns]"),
                w["Adjusted_Windspeed"].to_numpy(dtype=float),
                w[wc].to_numpy(dtype=float))

    def prep_wind(df):
        w = df[["time", "Adjusted_Windspeed"]].copy()
        w["time"] = pd.to_datetime(w["time"], errors="coerce")
        w["Adjusted_Windspeed"] = pd.to_numeric(w["Adjusted_Windspeed"], errors="coerce")
        w = w.dropna().sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
        return (w["time"].to_numpy(dtype="datetime64[ns]"),
                w["Adjusted_Windspeed"].to_numpy(dtype=float), None)

    # Project arrays (installation + transit).
    pt, pnd, pw = prep_ww(project_weather_df)

    # Loading-location arrays per mode.
    if loading_mode == "Offshore":
        lt, lnd, lw, loading_check_waves = pt, pnd, pw, True
    elif loading_mode == "Nearshore":
        if barge_weather_df is None:
            raise ValueError("Nearshore loading selected but no barge weather is available.")
        lt, lnd, lw = prep_ww(barge_weather_df); loading_check_waves = True
    else:  # Yard → wind only
        if yard_wind_df is None:
            raise ValueError("Yard loading selected but no yard weather is available.")
        lt, lnd, lw = prep_wind(yard_wind_df); loading_check_waves = False

    eli = eligible_indices if eligible_indices is not None else np.arange(len(pt))
    rng = np.random.default_rng(seed); n = len(base)
    tot_m = np.zeros((n, simulations), dtype=float)
    dwn_m = np.zeros((n, simulations), dtype=float)
    seqs = base["Sequence"].to_numpy(dtype=int)
    sl  = np.array([lm[s]  for s in seqs], dtype=float)
    sh  = np.array([hm[s]  for s in seqs], dtype=float)
    swl = np.array([wvm[s] for s in seqs], dtype=float)
    swd = np.array([wdm[s] for s in seqs], dtype=float)
    sit = np.array([tm[s]  for s in seqs], dtype=bool)
    # Per-row weather phase: loading rows (non-transit, in the loading block) use
    # the loading-location dataset; everything else uses project weather.
    s_is_loading = np.array([(s in loading_seq_set) and (not tm.get(s, False)) for s in seqs],
                            dtype=bool)

    def _wait(times, winds, waves, start, wlim, wndlim, check_w):
        idx = int(np.searchsorted(times, np.datetime64(pd.Timestamp(start)), side="left"))
        for h in range(max_wait_hours + 1):
            i = idx + h
            if i >= len(times): raise RuntimeError("Ran out of weather data.")
            if winds[i] <= wndlim and (not check_w or waves[i] <= wlim):
                return h, i
        raise RuntimeError("Exceeded max_wait_hours.")

    summ = []
    for si in range(simulations):
        sidx = int(rng.choice(eli)); bs = pd.Timestamp(pt[sidx])
        offset = float(mp_offset_days[si]) if mp_offset_days is not None else 0.0
        ct = bs + pd.Timedelta(days=offset); ss = ct; ta = td = 0.0
        for ri in range(n):
            if s_is_loading[ri]:
                dh, pi = _wait(lt, lnd, lw, ct, swl[ri], swd[ri], loading_check_waves)
                op_s = pd.Timestamp(lt[pi])
            else:
                dh, pi = _wait(pt, pnd, pw, ct, swl[ri], swd[ri], True)
                op_s = pd.Timestamp(pt[pi])
            ah = transit_hours if sit[ri] else float(rng.uniform(sl[ri], sh[ri]))
            tot_m[ri, si] = (dh + ah) * htd
            dwn_m[ri, si] = dh * htd
            ta += ah; td += dh
            ct = op_s + pd.Timedelta(hours=ah)
        summ.append({"Simulation": f"Sim{si+1}", "WTG_Start_Date": ss,
                     "Finish_Date": ct, "MP_Offset_Days": offset,
                     "Total_Active_Days":   ta * htd,
                     "Total_Downtime_Days": td * htd,
                     "Total_Project_Days":  (ta + td) * htd})
    sc = [f"Sim{i}" for i in range(1, simulations + 1)]
    return (pd.concat([base, pd.DataFrame(tot_m, columns=sc, index=base.index)], axis=1),
            pd.concat([base, pd.DataFrame(dwn_m, columns=sc, index=base.index)], axis=1),
            pd.DataFrame(summ))


# ── Output / plotting helpers ────────────────────────────────────────────────
def _sim_totals(df):
    sc = [c for c in df.columns if str(c).startswith("Sim")]
    return df[sc].apply(pd.to_numeric, errors="coerce").sum(axis=0).astype(float)


def build_project_summary(no_wx_df, wx_total_df, wx_down_df, label):
    nwt = _sim_totals(no_wx_df); wxt = _sim_totals(wx_total_df); wxd = _sim_totals(wx_down_df)
    net = wxt - wxd
    p0_perfect = float(nwt.min())

    def row(series, category, p0_override=None):
        v = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        r = {"Project": label, "Category": category,
             "Average": round(float(np.mean(v)), 2),
             "P0": round(p0_override if p0_override is not None
                         else float(np.percentile(v, 0)), 2)}
        for p in [10, 25, 50, 75, 90, 100]:
            r[f"P{p}"] = round(float(np.percentile(v, p)), 2)
        return r
    return pd.DataFrame([row(wxt, "Operation Days", p0_perfect),
                         row(wxd, "Downtime Days", 0.0),
                         row(net, "Net Operation", p0_perfect)])


# ── Excel export helpers ─────────────────────────────────────────────────────
def _sim_columns(df):
    """Return simulation columns (Sim1, Sim2, …) in their dataframe order."""
    if df is None or not isinstance(df, pd.DataFrame):
        return []
    return [c for c in df.columns if str(c).startswith("Sim")]


def _safe_sheet_name(name, used=None, max_len=31):
    """Create a unique Excel-safe sheet name."""
    used = used if used is not None else set()
    bad = '[]:*?/\\'
    clean = "".join("-" if ch in bad else ch for ch in str(name)).strip() or "Sheet"
    clean = clean[:max_len]
    candidate = clean
    i = 1
    while candidate in used:
        suffix = f"_{i}"
        candidate = clean[:max_len - len(suffix)] + suffix
        i += 1
    used.add(candidate)
    return candidate


def _step_export_summary(case_label, result_dict, phase):
    """Build one row per simulated timeline step with key duration/downtime stats.

    All duration and downtime values are in days because the simulation matrices
    are already stored in days throughout the app.
    """
    total_df = result_dict.get("total")
    downtime_df = result_dict.get("downtime")
    no_wx_df = result_dict.get("no_wx")
    if total_df is None or downtime_df is None:
        return pd.DataFrame()

    sim_cols = _sim_columns(total_df)
    down_cols = [c for c in sim_cols if c in downtime_df.columns]

    # Prefer the no-weather dataframe for explanatory text because some weather
    # simulation outputs intentionally keep only compact metadata columns.
    meta_source = no_wx_df if isinstance(no_wx_df, pd.DataFrame) else total_df
    meta_cols = [c for c in [
        "N", "Sequence", "Description", "Phase", "Inventory", "WTG_Left",
        "Cycles_Left", "Parallel_Block", "Source_Sequences",
        "Parallel_Branch_1_Hours", "Parallel_Branch_2_Hours"
    ] if c in meta_source.columns]
    meta = meta_source[meta_cols].copy() if meta_cols else pd.DataFrame(index=total_df.index)

    # Fill any missing columns from the total dataframe.
    for c in ["N", "Sequence", "Inventory", "WTG_Left", "Cycles_Left"]:
        if c not in meta.columns and c in total_df.columns:
            meta[c] = total_df[c].values
    if "Description" not in meta.columns:
        meta["Description"] = ""

    total_values = total_df[sim_cols].apply(pd.to_numeric, errors="coerce") if sim_cols else pd.DataFrame(index=total_df.index)
    down_values = downtime_df[down_cols].apply(pd.to_numeric, errors="coerce") if down_cols else pd.DataFrame(index=total_df.index)

    out = pd.DataFrame({
        "Case": case_label,
        "Phase": phase,
        "Step_Row": np.arange(1, len(total_df) + 1),
    })
    for c in ["N", "Sequence", "Description", "Phase", "Inventory", "WTG_Left", "Cycles_Left",
              "Parallel_Block", "Source_Sequences",
              "Parallel_Branch_1_Hours", "Parallel_Branch_2_Hours"]:
        if c in meta.columns:
            export_name = "Source_Phase" if c == "Phase" else c
            out[export_name] = meta[c].values

    # Per-step duration distribution across simulations.
    if len(sim_cols) > 0:
        out["Expected_Duration_Days"] = total_values.mean(axis=1)
        out["Minimum_Duration_Days"] = total_values.min(axis=1)
        out["P90_Duration_Days"] = total_values.quantile(0.90, axis=1)
        out["Max_Duration_Days"] = total_values.max(axis=1)
        out["Duration_Variance_Days2"] = total_values.var(axis=1, ddof=1)
    else:
        for c in ["Expected_Duration_Days", "Minimum_Duration_Days", "P90_Duration_Days",
                  "Max_Duration_Days", "Duration_Variance_Days2"]:
            out[c] = np.nan

    # Per-step weather downtime distribution across simulations.
    if len(down_cols) > 0:
        out["Expected_Downtime_Days"] = down_values.mean(axis=1)
        out["Downtime_Variance_Days2"] = down_values.var(axis=1, ddof=1)
        out["P90_Downtime_Days"] = down_values.quantile(0.90, axis=1)
        out["Max_Downtime_Days"] = down_values.max(axis=1)
    else:
        for c in ["Expected_Downtime_Days", "Downtime_Variance_Days2",
                  "P90_Downtime_Days", "Max_Downtime_Days"]:
            out[c] = np.nan

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(6)
    return out


def _write_case_frames(writer, case_label, result_dict, phase, used_sheets):
    """Write all dataframes for one simulation case to the Excel workbook."""
    step_summary = _step_export_summary(case_label, result_dict, phase)
    if not step_summary.empty:
        step_summary.to_excel(
            writer,
            sheet_name=_safe_sheet_name(f"{phase}_{case_label}_steps", used_sheets),
            index=False,
        )

    frame_map = {
        "summary": result_dict.get("summary"),
        "total_by_step": result_dict.get("total"),
        "downtime_by_step": result_dict.get("downtime"),
        "no_weather_by_step": result_dict.get("no_wx"),
    }
    for suffix, df in frame_map.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_excel(
                writer,
                sheet_name=_safe_sheet_name(f"{phase}_{case_label}_{suffix}", used_sheets),
                index=False,
            )


def build_simulation_excel_export(foundation_results, wtg_weather_results,
                                  simulate_foundations=False, intermediate_days=0):
    """Create an in-memory Excel workbook with all simulation dataframes."""
    output = io.BytesIO()
    used_sheets = set()
    all_step_summaries = []
    all_project_summaries = []

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for phase, results in [("Foundation", foundation_results), ("WTG", wtg_weather_results)]:
            if not isinstance(results, dict):
                continue
            for case_label, result_dict in results.items():
                step_summary = _step_export_summary(case_label, result_dict, phase)
                if not step_summary.empty:
                    all_step_summaries.append(step_summary)
                try:
                    all_project_summaries.append(
                        build_project_summary(result_dict["no_wx"], result_dict["total"],
                                              result_dict["downtime"], f"{phase}: {case_label}")
                    )
                except Exception:
                    pass
                _write_case_frames(writer, case_label, result_dict, phase, used_sheets)

        if all_step_summaries:
            pd.concat(all_step_summaries, ignore_index=True).to_excel(
                writer, sheet_name=_safe_sheet_name("All step summaries", used_sheets), index=False
            )
        if all_project_summaries:
            pd.concat(all_project_summaries, ignore_index=True).to_excel(
                writer, sheet_name=_safe_sheet_name("Project summaries", used_sheets), index=False
            )

        export_notes = pd.DataFrame([
            {"Field": "Expected_Duration_Days", "Meaning": "Average per-step total duration across simulations, including weather downtime."},
            {"Field": "Minimum_Duration_Days", "Meaning": "Minimum per-step total duration across simulations."},
            {"Field": "P90_Duration_Days", "Meaning": "90th percentile per-step total duration across simulations."},
            {"Field": "Max_Duration_Days", "Meaning": "Maximum per-step total duration across simulations."},
            {"Field": "Expected_Downtime_Days", "Meaning": "Average per-step weather downtime across simulations."},
            {"Field": "Downtime_Variance_Days2", "Meaning": "Sample variance of per-step weather downtime across simulations."},
            {"Field": "Duration_Variance_Days2", "Meaning": "Sample variance of per-step total duration across simulations."},
            {"Field": "Intermediate_Days", "Meaning": float(intermediate_days) if simulate_foundations else 0.0},
        ])
        export_notes.to_excel(writer, sheet_name=_safe_sheet_name("Export notes", used_sheets), index=False)

        # Basic formatting for readability.
        for sheet in writer.sheets.values():
            sheet.freeze_panes(1, 0)
            sheet.autofilter(0, 0, 0, 0)

    output.seek(0)
    return output.getvalue()


# A simple, stable color mapping per vessel family + linestyle per loading mode
_VESSEL_COLORS = {"W90": "red", "JUV": "blue", "FIV": "orange"}
_MODE_STYLES   = {"Yard": "-", "Nearshore": ":", "Offshore": "--"}


def _case_style(case_label):
    """case_label like 'W90 (Yard)' → (color, linestyle)."""
    color = "grey"; ls = "-"
    for v, c in _VESSEL_COLORS.items():
        if case_label.startswith(v):
            color = c; break
    for m, s in _MODE_STYLES.items():
        if f"({m})" in case_label:
            ls = s; break
    return color, ls


def plot_cumulative(results_dict, title):
    fig, ax = plt.subplots(figsize=(12, 7)); p50_finals = []
    for name, df in results_dict.items():
        sc = [c for c in df.columns if str(c).startswith("Sim")]
        if not sc: continue
        cum = df[sc].cumsum(axis=0); p50 = cum.median(axis=1)
        color, ls = _case_style(name)
        ax.plot(np.arange(1, len(df) + 1), cum, color="grey", alpha=0.15, linewidth=1)
        ax.plot(np.arange(1, len(df) + 1), p50, label=f"{name} P50",
                color=color, linestyle=ls, linewidth=2.5)
        p50_finals.append(float(p50.iloc[-1]))
    if p50_finals and min(p50_finals) > 0:
        ref = min(p50_finals)
        ax.secondary_yaxis("right",
                           functions=(lambda y: y / ref * 100,
                                      lambda y: y / 100 * ref)
                           ).set_ylabel("Relative Completion (%)")
    ax.set_title(title); ax.set_xlabel("Sequence Step")
    ax.set_ylabel("Cumulative Duration (days)")
    ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
    return fig




# ── Advanced Weather Intelligence helpers ───────────────────────────────────
def _get_wave_col(df):
    if df is None:
        return None
    return next((c for c in _WAVE_HEIGHT_COLS if c in df.columns), None)


def _read_operation_limits(excel_src, sheet_name):
    """Read per-operation metocean limits from the uploaded Excel sheet."""
    raw = _excel_read(excel_src, sheet_name)
    raw.columns = [str(c).strip() for c in raw.columns]
    needed = ["N", "Description", "Transit", "Loading", "Installation",
              "Oplim Waves (m)", "Oplim Wind (m/s)"]
    for c in needed:
        if c not in raw.columns:
            raw[c] = np.nan
    out = raw[needed].copy()
    out["N"] = pd.to_numeric(out["N"], errors="coerce")
    out = out.dropna(subset=["N"]).drop_duplicates(subset=["N"])
    out["N"] = out["N"].astype(int)
    out["Transit"] = out["Transit"].map(_norm_bool)
    out["Loading"] = out["Loading"].map(_norm_marker)
    out["Installation"] = out["Installation"].map(_norm_marker)
    out["Oplim Waves (m)"] = pd.to_numeric(out["Oplim Waves (m)"], errors="coerce").fillna(np.inf)
    out["Oplim Wind (m/s)"] = pd.to_numeric(out["Oplim Wind (m/s)"], errors="coerce").fillna(np.inf)
    return out


def _site_weather_for_mode(mode, project_weather, barge_weather, yard_wind):
    if mode == "Yard" and yard_wind is not None:
        return yard_wind, False, "Yard weather (wind only)"
    if mode == "Nearshore" and barge_weather is not None:
        return barge_weather, True, "Barge / nearshore weather (wind + waves)"
    return project_weather, True, "Project / offshore weather (wind + waves)"


def _weather_operability_heatmap(weather_df, op_limits_df, check_waves=True):
    """Return month x hour table with mean operability across operation rows.

    Each weather timestamp is scored as the share of operation limits that are
    workable at that exact hour. Aggregating by month/hour gives an intuitive
    operability heatmap while respecting individual Excel operation limits.
    """
    if weather_df is None or op_limits_df is None or op_limits_df.empty:
        return pd.DataFrame()

    w = weather_df.copy()
    if "time" not in w.columns or "Adjusted_Windspeed" not in w.columns:
        return pd.DataFrame()
    w["time"] = pd.to_datetime(w["time"], errors="coerce")
    w["Adjusted_Windspeed"] = pd.to_numeric(w["Adjusted_Windspeed"], errors="coerce")
    wave_col = _get_wave_col(w)
    if check_waves and wave_col:
        w[wave_col] = pd.to_numeric(w[wave_col], errors="coerce")
    w = w.dropna(subset=["time", "Adjusted_Windspeed"])
    if check_waves and wave_col:
        w = w.dropna(subset=[wave_col])
    if w.empty:
        return pd.DataFrame()

    wind_limits = op_limits_df["Oplim Wind (m/s)"].to_numpy(dtype=float)
    wave_limits = op_limits_df["Oplim Waves (m)"].to_numpy(dtype=float)
    winds = w["Adjusted_Windspeed"].to_numpy(dtype=float)[:, None]
    ok = winds <= wind_limits[None, :]
    if check_waves and wave_col:
        waves = w[wave_col].to_numpy(dtype=float)[:, None]
        ok = ok & (waves <= wave_limits[None, :])

    w["Operability_%"] = ok.mean(axis=1) * 100.0
    w["Month"] = w["time"].dt.month
    w["Hour"] = w["time"].dt.hour
    heat = w.pivot_table(index="Month", columns="Hour", values="Operability_%", aggfunc="mean")
    return heat.reindex(index=range(1, 13), columns=range(0, 24))


def _plot_heatmap_table(heat, title):
    fig, ax = plt.subplots(figsize=(13, 5.8))
    if heat is None or heat.empty:
        ax.text(0.5, 0.5, "No heatmap data available", ha="center", va="center")
        ax.axis("off")
        return fig
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", origin="upper", vmin=0, vmax=100)
    ax.set_title(title)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Month")
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels([str(h) for h in range(24)])
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average operability across Excel operation limits (%)")
    plt.tight_layout()
    return fig


def _sim_column_for_percentile(summary_df, percentile=0.5):
    if summary_df is None or summary_df.empty or "Simulation" not in summary_df.columns:
        return None
    vals = pd.to_numeric(summary_df["Total_Project_Days"], errors="coerce")
    target = float(vals.quantile(percentile))
    idx = (vals - target).abs().idxmin()
    return str(summary_df.loc[idx, "Simulation"]), target


def _build_replay_table(result_dict, percentile=0.5):
    """Build cumulative replay table for the simulation closest to P50/P90."""
    if not result_dict:
        return pd.DataFrame(), None, None
    sim_info = _sim_column_for_percentile(result_dict.get("summary"), percentile)
    if sim_info is None:
        return pd.DataFrame(), None, None
    sim_col, target = sim_info
    total = result_dict.get("total")
    down = result_dict.get("downtime")
    meta = result_dict.get("no_wx") if isinstance(result_dict.get("no_wx"), pd.DataFrame) else total
    if total is None or sim_col not in total.columns:
        return pd.DataFrame(), sim_col, target

    df = pd.DataFrame({
        "Step": np.arange(1, len(total) + 1),
        "Duration_Days": pd.to_numeric(total[sim_col], errors="coerce").fillna(0.0),
        "Downtime_Days": pd.to_numeric(down[sim_col], errors="coerce").fillna(0.0) if down is not None and sim_col in down.columns else 0.0,
    })
    for c in ["Description", "Sequence", "Phase", "Inventory", "WTG_Left", "Cycles_Left"]:
        if isinstance(meta, pd.DataFrame) and c in meta.columns:
            df[c] = meta[c].values[:len(df)]
    df["Active_Days"] = (df["Duration_Days"] - df["Downtime_Days"]).clip(lower=0)
    df["Cum_Start_Days"] = df["Duration_Days"].cumsum().shift(fill_value=0.0)
    df["Cum_End_Days"] = df["Duration_Days"].cumsum()
    df["Status"] = np.where(df["Downtime_Days"] > 1e-9, "Weather waiting + operation", "Active operation")
    return df, sim_col, target


def _plot_replay_progress(replay_df, selected_idx):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    if replay_df is None or replay_df.empty:
        ax.text(0.5, 0.5, "No replay data available", ha="center", va="center")
        ax.axis("off")
        return fig
    x = replay_df["Step"]
    ax.plot(x, replay_df["Cum_End_Days"], linewidth=2.5, label="Cumulative duration")
    if 0 <= selected_idx < len(replay_df):
        row = replay_df.iloc[selected_idx]
        ax.axvline(row["Step"], linestyle="--", linewidth=1.5)
        ax.scatter([row["Step"]], [row["Cum_End_Days"]], s=60, zorder=5)
    ax.set_xlabel("Replay step")
    ax.set_ylabel("Cumulative campaign days")
    ax.set_title("Weather Replay – campaign progress")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def _replay_map(lat_project, lon_project, lat_barge, lon_barge, lat_yard, lon_yard,
                loading_mode, progress_fraction, current_label="Vessel"):
    if folium is None:
        return None
    project = (float(lat_project), float(lon_project))
    if loading_mode == "Yard":
        load = (float(lat_yard), float(lon_yard))
    elif loading_mode == "Nearshore":
        load = (float(lat_barge), float(lon_barge))
    else:
        load = project
    frac = max(0.0, min(1.0, float(progress_fraction)))
    vessel_lat = load[0] + (project[0] - load[0]) * frac
    vessel_lon = load[1] + (project[1] - load[1]) * frac
    center = [(load[0] + project[0]) / 2, (load[1] + project[1]) / 2]
    m = folium.Map(location=center, zoom_start=4)
    folium.Marker(load, tooltip=f"Loading location: {loading_mode}", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(project, tooltip="Project site", icon=folium.Icon(color="blue")).add_to(m)
    folium.PolyLine([load, project], tooltip="Transit route", weight=3).add_to(m)
    folium.Marker((vessel_lat, vessel_lon), tooltip=current_label, icon=folium.Icon(color="red", icon="ship", prefix="fa")).add_to(m)
    return m


def render_advanced_weather_intelligence(excel_bytes, wtg_weather_results,
                                         df_project_weather, df_barge_weather, df_yard_wind,
                                         lat, lon, lat_barge, lon_barge, lat_yard, lon_yard):
    """Render post-simulation heatmaps, replay, and API concept note."""
    st.markdown("---")
    st.header("🌐 Advanced Weather Intelligence")
    st.caption(
        "Draft module: uses the uploaded Excel operation limits plus loaded ERA5 weather. "
        "The live MetOcean API concept is explained here but not connected yet."
    )

    tab_heat, tab_replay, tab_api = st.tabs([
        "Weather Window Heatmaps", "Weather Replay Mode", "Real MetOcean API concept"
    ])

    with tab_heat:
        st.subheader("Weather Window Heatmaps – month × hour")
        st.write(
            "This estimates how workable each month/hour combination is by checking the loaded weather "
            "against the individual operation limits in the selected vessel Excel sheet. A value of 100% "
            "means all checked operation rows are workable on average for that month/hour bucket."
        )
        case_labels = list(wtg_weather_results.keys())
        if not case_labels:
            st.info("Run at least one WTG simulation case to generate operation-limit heatmaps.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                case = st.selectbox("Select WTG case", case_labels, key="awi_heat_case")
            r = wtg_weather_results[case]
            mode = r.get("mode", "Yard")
            sheet = None
            # The result dict may not store sheet in older runs, so infer from no_wx metadata if needed.
            for src in [r.get("sheet"), r.get("wtg_sheet")]:
                if src:
                    sheet = src
            with c2:
                site_choice = st.selectbox(
                    "Weather location to evaluate",
                    ["Loading location", "Project installation/transit location"],
                    key="awi_heat_site",
                )
            if not sheet:
                st.warning("This draft could not identify the Excel sheet for the selected case. Re-run using this draft version to enable heatmaps.")
            else:
                op_limits = _read_operation_limits(io.BytesIO(excel_bytes), sheet)
                if site_choice.startswith("Loading"):
                    wx, check_waves, site_label = _site_weather_for_mode(mode, df_project_weather, df_barge_weather, df_yard_wind)
                else:
                    wx, check_waves, site_label = df_project_weather, True, "Project / installation weather (wind + waves)"
                heat = _weather_operability_heatmap(wx, op_limits, check_waves=check_waves)
                st.pyplot(_plot_heatmap_table(heat, f"{case} – {site_label}"))
                st.dataframe(heat.round(1), use_container_width=True)

    with tab_replay:
        st.subheader("Weather Replay Mode")
        st.write(
            "Replay uses the simulation closest to P50 or P90 total duration. The first draft provides a "
            "step slider, progress curve, operation details, and a vessel marker moving between the loading "
            "location and project site as the campaign progresses."
        )
        case_labels = list(wtg_weather_results.keys())
        if not case_labels:
            st.info("Run at least one WTG simulation case to enable replay.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                case = st.selectbox("Select replay case", case_labels, key="awi_replay_case")
            with c2:
                pct_label = st.radio("Replay percentile", ["P50", "P90"], horizontal=True, key="awi_replay_pct")
            percentile = 0.5 if pct_label == "P50" else 0.9
            r = wtg_weather_results[case]
            replay, sim_col, target = _build_replay_table(r, percentile=percentile)
            if replay.empty:
                st.warning("No replay table could be generated for this case.")
            else:
                step = st.slider("Replay step", min_value=1, max_value=len(replay), value=1, key="awi_replay_step")
                idx = int(step) - 1
                row = replay.iloc[idx]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Simulation used", sim_col or "N/A")
                m2.metric("Target percentile total", f"{target:.1f} d" if target is not None else "N/A")
                m3.metric("Current campaign day", f"{row['Cum_End_Days']:.1f}")
                m4.metric("Step downtime", f"{row['Downtime_Days']:.2f} d")
                st.pyplot(_plot_replay_progress(replay, idx))
                st.markdown("**Current operation**")
                st.write({
                    "Step": int(row.get("Step", step)),
                    "Sequence": row.get("Sequence", ""),
                    "Description": row.get("Description", ""),
                    "Status": row.get("Status", ""),
                    "Duration days": round(float(row.get("Duration_Days", 0)), 3),
                    "Downtime days": round(float(row.get("Downtime_Days", 0)), 3),
                })
                if folium is not None and st_folium is not None:
                    frac = float(row["Cum_End_Days"] / replay["Cum_End_Days"].iloc[-1]) if replay["Cum_End_Days"].iloc[-1] > 0 else 0.0
                    fmap = _replay_map(lat, lon, lat_barge, lon_barge, lat_yard, lon_yard,
                                        r.get("mode", "Yard"), frac, current_label=f"{case} – day {row['Cum_End_Days']:.1f}")
                    if fmap is not None:
                        st_folium(fmap, height=430, use_container_width=True, key=f"awi_replay_map_{case}_{pct_label}_{step}")
                else:
                    st.info("Install folium and streamlit-folium to show the moving vessel map.")

    with tab_api:
        st.subheader("Real MetOcean APIs – idea and value")
        st.write(
            "The idea is to let the app switch between historical ERA5 analysis and live/forecast MetOcean data. "
            "ERA5 is excellent for long-term statistical planning, while a live MetOcean API would support actual "
            "campaign readiness decisions: should the vessel sail, wait, load, or delay based on the next days of "
            "forecast wind and wave conditions?"
        )
        st.markdown(
            """
**Why it is valuable**

- Turns the app from a planning simulator into an operational decision-support tool.
- Allows comparison of historical risk versus upcoming forecast risk.
- Enables short-term go/no-go checks for loading, transit, and installation.
- Makes the same operation-limit logic useful both before tendering and during execution.

**Recommended later implementation path**

1. Add a provider interface, for example `get_forecast(provider, lat, lon)`.
2. Start with a simple JSON-based provider such as Open-Meteo Marine for proof of concept.
3. Later plug in company-approved providers such as StormGeo, DTN, ECMWF, or other contracted MetOcean services.
4. Convert each provider response into the same internal dataframe format your app already uses: `time`, `Adjusted_Windspeed`, and wave-height columns.
5. Reuse the exact same downtime and operability logic already implemented for ERA5.

**What you would need later**

- API base URL from the provider.
- API key or token.
- License confirmation that the forecast can be used inside the app.
- Parameter mapping: wind speed, wind direction, significant wave height, wave period, and wave direction.
- Rate limits and allowed forecast horizon.
            """
        )

def plot_bar_comparison(bar_rows, title="Campaign Duration Comparison"):
    df = pd.DataFrame(bar_rows)
    fig, ax1 = plt.subplots(figsize=(max(11, 1.6 * len(df) + 4), 5.5))
    fig.patch.set_facecolor("#f2f2f2"); ax1.set_facecolor("#f2f2f2")
    x = np.arange(len(df)); w = 0.34
    b1 = ax1.bar(x - w / 2, df["Net_P50"],   w, label="Net duration (P50)",   color="#1f3763")
    b2 = ax1.bar(x + w / 2, df["Total_P50"], w, label="Total duration (P50)", color="#6a9f3f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Case"], rotation=20, ha="right")
    ax1.set_ylabel("Duration (days)")
    ax1.set_title(title, fontweight="bold"); ax1.grid(axis="y", alpha=0.3)
    off = float(df["Total_P50"].max()) * 0.02
    for bars in (b1, b2):
        for b in bars:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + off,
                     f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(x, df["Downtime_Pct"], color="red", linewidth=2.2, label="Weather downtime (%)")
    ax2.set_ylabel("Weather downtime (%)")
    for xi, yi in zip(x, df["Downtime_Pct"]):
        ax2.text(xi, yi, f"{yi:.0f}%", ha="center", va="bottom", fontsize=9,
                 bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                           edgecolor="none", alpha=0.8))
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.20),
               ncol=3, frameon=False)
    for a in (ax1, ax2):
        for sp in a.spines.values(): sp.set_visible(False)
    plt.tight_layout(); return fig


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3-5 – RUN ALL SIMULATIONS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🏗️  3–5.  Run All Simulations")

# Determine the cases to run
def _enumerate_cases(vessel_info, phase):
    """Return (vessel, mode, label) tuples for a phase-specific vessel × loading-mode selection."""
    mode_key = "foundation_modes" if phase == "foundation" else "wtg_modes"
    prefix = "FDN" if phase == "foundation" else "WTG"
    cases = []
    for vessel, info in vessel_info.items():
        if not info["selected"]:
            continue
        for mode in info[mode_key]:
            cases.append((vessel, mode, f"{vessel} {prefix} ({mode})"))
    return cases


def _combined_case_label(vessel, fdn_mode, wtg_mode):
    return f"{vessel} FDN {fdn_mode} → WTG {wtg_mode}"


if not (uploaded_excel and df_project_weather is not None and df_barge_weather is not None):
    st.info("Upload the operations Excel file and weather ZIPs above to enable simulations.")
else:
    vessel_info = _populate_vessel_info()
    foundation_cases = _enumerate_cases(vessel_info, "foundation") if simulate_foundations else []
    wtg_cases = _enumerate_cases(vessel_info, "wtg")
    cases = wtg_cases

    # Preview of what will be run
    if not cases:
        st.warning("No vessel × loading-mode combinations selected.  "
                   "Tick at least one vessel **and** one loading mode for it.")
    else:
        st.subheader("Cases to be simulated")
        preview_rows = []
        preview_phase_cases = [("Foundation", foundation_cases), ("WTG", wtg_cases)]
        for phase_name, phase_cases in preview_phase_cases:
            for vessel, mode, label in phase_cases:
                info = vessel_info[vessel]
                td   = _mode_transit_distance(mode, info["transit_speed"],
                                              dock_distance, barge_distance)
                t_h  = td / info["transit_speed"]
                preview_rows.append({"Case": label,
                                     "Phase": phase_name,
                                     "Vessel": vessel, "Loading mode": mode,
                                     "Carry cap": info["carry_cap"],
                                     "Transit speed (kn)": info["transit_speed"],
                                     "Transit distance (Nm)": round(td, 2),
                                     "Transit time (hrs)": round(t_h, 2)})
        st.dataframe(pd.DataFrame(preview_rows).set_index("Case"))

    can_run = bool(wtg_cases) and (bool(foundation_cases) or not simulate_foundations)
    if can_run and st.button("▶️  Run All Simulations", type="primary"):

        wt_arr = df_project_weather["time"].to_numpy(dtype="datetime64[ns]")
        try:
            seasonal_eligible = get_seasonal_eligible_indices(wt_arr, Sim_start_date)
            st.info(f"Seasonal eligible start timestamps: {len(seasonal_eligible):,}")
        except Exception as e:
            st.error(f"Seasonal index error: {e}"); st.stop()

        # ─────────────────────────────────────────────────────────────────
        #  SECTION 3 – Foundation simulations
        # ─────────────────────────────────────────────────────────────────
        foundation_results = {}   # case_label -> dict(timeline, total, downtime, summary, no_wx)

        if simulate_foundations:
            st.subheader(f"3.  Foundation Installation Simulations — type: **{foundation_type}**")
            prog_fd = st.progress(0, text=f"Running {foundation_type} simulations …")

            for ci, (vessel, mode, label) in enumerate(foundation_cases):
                info = vessel_info[vessel]
                sheet = info["mp_sheet"] if foundation_type == "Monopile" else info["jacket_sheet"]
                cap   = info["carry_cap"] if foundation_type == "Monopile" else 1
                td    = _mode_transit_distance(mode, info["transit_speed"],
                                                dock_distance, barge_distance)
                spd   = info["transit_speed"]

                with st.spinner(f"Foundation – {label} …"):
                    try:
                        if foundation_type == "Monopile":
                            # Sanity-check: MP sheet must have Loading START/END
                            if not _sheet_has_loading_phase(io.BytesIO(excel_bytes), sheet):
                                raise ValueError(
                                    f"Sheet '{sheet}' has no Loading START/END markers — "
                                    "is this actually a Monopile sheet?")
                            tl = build_mp_timeline(io.BytesIO(excel_bytes), sheet,
                                                    cap, N_WTG, htd)
                        else:
                            tl = build_jacket_timeline(io.BytesIO(excel_bytes), sheet,
                                                        N_WTG, htd)

                        tot, dwn, summ = simulate_weather_impacts_foundation(
                            tl, df_project_weather,
                            io.BytesIO(excel_bytes), sheet,
                            transit_distance=td, transit_speed=spd,
                            simulations=int(Simulations),
                            eligible_indices=seasonal_eligible, htd=htd)
                        foundation_results[label] = {
                            "vessel": vessel, "mode": mode,
                            "timeline": tl, "total": tot,
                            "downtime": dwn, "summary": summ, "no_wx": tot,
                        }
                        p50 = round(summ["Total_Project_Days"].median(), 1)
                        st.success(f"✅  Foundation {label} — P50 total: {p50} days")
                    except Exception as e:
                        st.error(f"Foundation {label} failed: {e}")
                prog_fd.progress((ci + 1) / len(foundation_cases))

            if not foundation_results:
                st.error("No foundation results produced — stopping."); st.stop()

            # Cumulative duration plot
            st.subheader(f"Cumulative {foundation_type} Foundation Duration (With Weather)")
            st.pyplot(plot_cumulative({lbl: r["total"] for lbl, r in foundation_results.items()},
                                      f"Cumulative {foundation_type} Foundation Duration (With Weather)"))

            # Duration summary table
            st.subheader(f"{foundation_type} Foundation Project Duration Summary")
            fdn_rows = []
            for lbl, r in foundation_results.items():
                s = r["summary"]
                tot_p50 = s["Total_Project_Days"].median()
                dwn_p50 = s["Total_Downtime_Days"].median()
                fdn_rows.append({
                    "Case": lbl,
                    "P50 Total (d)":    round(tot_p50, 1),
                    "P90 Total (d)":    round(s["Total_Project_Days"].quantile(.9), 1),
                    "P50 Downtime (d)": round(dwn_p50, 1),
                    "Downtime %":       round(dwn_p50 / tot_p50 * 100 if tot_p50 > 0 else 0, 1),
                    "P50 Active (d)":   round(s["Total_Active_Days"].median(), 1)})
            st.dataframe(pd.DataFrame(fdn_rows).set_index("Case"))

            # Full project summary (P0 = no-weather baseline)
            st.subheader(f"{foundation_type} Foundation Full Project Summary (P0 = no-weather baseline)")
            fdn_full_frames = [build_project_summary(r["no_wx"], r["total"], r["downtime"], lbl)
                               for lbl, r in foundation_results.items()]
            st.dataframe(pd.concat(fdn_full_frames, ignore_index=True))

            # Bar chart
            st.subheader(f"{foundation_type} Foundation Campaign Duration Bar Chart")
            fdn_bar_rows = []
            for lbl, r in foundation_results.items():
                s = r["summary"]
                tp = float(s["Total_Project_Days"].median())
                dp = float(s["Total_Downtime_Days"].median())
                fdn_bar_rows.append({"Case": lbl,
                                     "Total_P50": round(tp, 1),
                                     "Net_P50":   round(tp - dp, 1),
                                     "Downtime_Pct": round(dp / tp * 100 if tp > 0 else 0, 1)})
            st.pyplot(plot_bar_comparison(
                fdn_bar_rows, title=f"{foundation_type} Campaign Duration Comparison"))

            # Foundation → WTG offset per case
            mp_end_offsets = {}
            for lbl, r in foundation_results.items():
                td_days = pd.to_numeric(r["summary"]["Total_Project_Days"],
                                          errors="coerce").to_numpy(dtype=float)
                mp_end_offsets[lbl] = td_days + float(intermediate_days)
        else:
            st.info("Foundation phase skipped — WTG installation will start from "
                    "the Monte-Carlo'd start date with no offset.")
            mp_end_offsets = {label: None for (_, _, label) in wtg_cases}

        # ─────────────────────────────────────────────────────────────────
        #  SECTION 4 – WTG duration (no-weather baseline)
        # ─────────────────────────────────────────────────────────────────
        st.subheader("4.  WTG Installation – No-Weather Baseline")
        wtg_duration_results = {}

        for ci, (vessel, mode, label) in enumerate(wtg_cases):
            info = vessel_info[vessel]
            sheet = info["wtg_sheet"]
            cap   = info["carry_cap"]
            spd   = info["transit_speed"]
            td    = _mode_transit_distance(mode, spd, dock_distance, barge_distance)

            with st.spinner(f"Building WTG timeline — {label} …"):
                try:
                    if vessel == "W90":
                        tl  = build_w90_timeline(io.BytesIO(excel_bytes), sheet, cap,
                                                  int(N_WTG), htd,
                                                  dock_distance=td, transit_speed=spd)
                        dur = add_duration_simulations_w90(
                            tl, io.BytesIO(excel_bytes), sheet,
                            simulations=int(Simulations), seed=42, htd=htd,
                            transit_distance=td, transit_speed=spd)
                    else:
                        tl  = build_sequential_wtg_timeline(io.BytesIO(excel_bytes), sheet,
                                                             int(cap), int(N_WTG), htd,
                                                             dock_distance=td, transit_speed=spd)
                        dur = add_duration_simulations_sequential(
                            tl, io.BytesIO(excel_bytes), sheet,
                            simulations=int(Simulations),
                            dock_distance=td, transit_speed=spd,
                            seed=42, htd=htd)
                    wtg_duration_results[label] = {
                        "vessel": vessel, "mode": mode, "sheet": sheet,
                        "timeline": tl, "duration": dur,
                        "mp_offset_days": mp_end_offsets.get(label),
                        "transit_distance": td, "transit_speed": spd, "cap": cap}
                    st.success(f"✅  WTG timeline built for {label}")
                except Exception as e:
                    st.error(f"WTG build failed for {label}: {e}")

        # ─────────────────────────────────────────────────────────────────
        #  SECTION 5 – WTG weather simulations
        # ─────────────────────────────────────────────────────────────────
        st.subheader("5.  WTG Installation – Weather Simulations")
        wtg_seasonal = get_seasonal_eligible_indices(wt_arr, Sim_start_date)
        wtg_weather_results = {}
        prog_wtg = st.progress(0, text="Running WTG weather simulations …")

        weather_jobs = []
        for wtg_label, res in wtg_duration_results.items():
            if simulate_foundations:
                matching_fdn = [(fl, fr) for fl, fr in foundation_results.items()
                                if fr["vessel"] == res["vessel"]]
                for fdn_label, fdn_res in matching_fdn:
                    combined_label = _combined_case_label(res["vessel"], fdn_res["mode"], res["mode"])
                    offset = pd.to_numeric(fdn_res["summary"]["Total_Project_Days"],
                                           errors="coerce").to_numpy(dtype=float) + float(intermediate_days)
                    weather_jobs.append((combined_label, res, offset, fdn_label))
            else:
                weather_jobs.append((wtg_label, res, None, None))

        for vi, (label, res, offset, fdn_label) in enumerate(weather_jobs):
            vessel = res["vessel"]; mode = res["mode"]
            sheet  = res["sheet"];  td = res["transit_distance"]; spd = res["transit_speed"]

            with st.spinner(f"WTG weather – {label} …"):
                try:
                    if vessel == "W90":
                        tot, dwn, summ = simulate_weather_impacts_w90(
                            res["duration"], df_project_weather, df_barge_weather,
                            df_yard_wind, io.BytesIO(excel_bytes), sheet,
                            transit_distance=td, transit_speed=spd,
                            simulations=int(Simulations),
                            eligible_indices=wtg_seasonal, seed=42, max_wait_hours=5000,
                            htd=htd, mp_offset_days=offset,
                            loading_mode=mode)
                    else:
                        tot, dwn, summ = simulate_sequential_wtg_weather(
                            res["duration"], df_project_weather, df_barge_weather,
                            df_yard_wind, io.BytesIO(excel_bytes), sheet,
                            transit_distance=td, transit_speed=spd,
                            simulations=int(Simulations),
                            eligible_indices=wtg_seasonal, seed=42, max_wait_hours=5000,
                            htd=htd, mp_offset_days=offset,
                            loading_mode=mode)
                    wtg_weather_results[label] = {
                        "vessel": vessel, "mode": mode, "foundation_case": fdn_label,
                        "total": tot, "downtime": dwn,
                        "summary": summ, "no_wx": res["duration"], "sheet": sheet}
                    p50 = round(summ["Total_Project_Days"].median(), 1)
                    st.success(f"✅  WTG {label} — P50 total: {p50} days")
                except Exception as e:
                    st.error(f"WTG weather failed for {label}: {e}")
            prog_wtg.progress((vi + 1) / max(len(weather_jobs), 1))

        if not wtg_weather_results:
            st.warning("No WTG weather results produced."); st.stop()

        # ── Final outputs ────────────────────────────────────────────────
        st.markdown("---")
        st.header("📊  Final Results")

        st.subheader("Cumulative WTG Campaign Duration (With Weather)")
        st.pyplot(plot_cumulative({lbl: r["total"] for lbl, r in wtg_weather_results.items()},
                                   "Cumulative WTG Duration (With Weather)"))

        st.subheader("WTG Project Duration Summary")
        wtg_rows = []
        for lbl, r in wtg_weather_results.items():
            s = r["summary"]
            tot_p50 = s["Total_Project_Days"].median()
            dwn_p50 = s["Total_Downtime_Days"].median()
            wtg_rows.append({
                "Case": lbl,
                "P50 Total (d)":    round(tot_p50, 1),
                "P90 Total (d)":    round(s["Total_Project_Days"].quantile(.9), 1),
                "P50 Downtime (d)": round(dwn_p50, 1),
                "Downtime %":       round(dwn_p50 / tot_p50 * 100 if tot_p50 > 0 else 0, 1),
                "P50 Active (d)":   round(s["Total_Active_Days"].median(), 1)})
        st.dataframe(pd.DataFrame(wtg_rows).set_index("Case"))

        st.subheader("Full Project Summary (P0 = no-weather baseline)")
        full_frames = [build_project_summary(r["no_wx"], r["total"], r["downtime"], lbl)
                       for lbl, r in wtg_weather_results.items()]
        st.dataframe(pd.concat(full_frames, ignore_index=True))

        st.subheader("WTG Campaign Duration Bar Chart")
        bar_rows = []
        for lbl, r in wtg_weather_results.items():
            s = r["summary"]
            tp = float(s["Total_Project_Days"].median())
            dp = float(s["Total_Downtime_Days"].median())
            bar_rows.append({"Case": lbl,
                              "Total_P50": round(tp, 1),
                              "Net_P50":   round(tp - dp, 1),
                              "Downtime_Pct": round(dp / tp * 100 if tp > 0 else 0, 1)})
        st.pyplot(plot_bar_comparison(bar_rows,
                                       title="WTG Campaign Duration Comparison"))

        # Combined full-project bar chart (foundation + WTG if both available)
        if simulate_foundations and foundation_results:
            st.subheader("Full Campaign (Foundation + WTG) — P50 Totals")
            comb_rows = []
            for lbl, r in wtg_weather_results.items():
                wtg_tp = float(r["summary"]["Total_Project_Days"].median())
                wtg_dp = float(r["summary"]["Total_Downtime_Days"].median())
                fdn_key = r.get("foundation_case")
                fdn = foundation_results.get(fdn_key) if fdn_key else None
                if fdn:
                    fdn_tp = float(fdn["summary"]["Total_Project_Days"].median())
                    fdn_dp = float(fdn["summary"]["Total_Downtime_Days"].median())
                else:
                    fdn_tp = fdn_dp = 0.0
                total = fdn_tp + float(intermediate_days) + wtg_tp
                down  = fdn_dp + wtg_dp
                comb_rows.append({"Case": lbl,
                                   "Total_P50": round(total, 1),
                                   "Net_P50":   round(total - down, 1),
                                   "Downtime_Pct": round(down / total * 100 if total > 0 else 0, 1)})
            st.pyplot(plot_bar_comparison(comb_rows,
                                          title="Full Campaign P50 (Foundation + intermediate + WTG)"))
            st.dataframe(pd.DataFrame(comb_rows).set_index("Case"))


        render_advanced_weather_intelligence(
            excel_bytes=excel_bytes,
            wtg_weather_results=wtg_weather_results,
            df_project_weather=df_project_weather,
            df_barge_weather=df_barge_weather,
            df_yard_wind=df_yard_wind,
            lat=lat, lon=lon,
            lat_barge=lat_barge, lon_barge=lon_barge,
            lat_yard=lat_yard, lon_yard=lon_yard,
        )

        st.subheader("⬇️  Download Simulation Data")
        export_bytes = build_simulation_excel_export(
            foundation_results=foundation_results,
            wtg_weather_results=wtg_weather_results,
            simulate_foundations=simulate_foundations,
            intermediate_days=intermediate_days,
        )
        st.download_button(
            label="Download all simulation data as Excel",
            data=export_bytes,
            file_name=f"w90_simulation_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help=("Includes one row per simulated sequence step with description, expected duration, "
                  "minimum, P90, max, expected downtime, and variance, plus the raw simulation dataframes."),
        )

        st.success("🎉  All simulations complete!")


# ═════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "W90 Monte Carlo v8 · Streamlit · "
    "Three weather locations (project, barge, yard), deterministic per-phase "
    "weather routing, coordinate-gated loading modes, and the new per-turbine "
    "parallel-loading W90 model."
)
