# F1 Race Insights AI - Main Dashboard
# Your complete F1 analytics platform!

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle

# ============ PAGE CONFIGURATION ============

st.set_page_config(
    page_title="F1 Race Insights AI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS - F1 THEME ============

st.markdown("""
    <style>
    /* Main background - F1 dark theme */
    .stApp {
        background: linear-gradient(to bottom, #15151E 0%, #1a1a2e 100%);
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        color: #00D2BE;
        font-weight: 700;
    }

    /* Headers */
    h1 {
        color: #E10600;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 3px solid #E10600;
        padding-bottom: 10px;
    }

    h2, h3 {
        color: #FFFFFF;
        font-weight: 700;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #1E1E2E 0%, #15151E 100%);
        border-right: 2px solid #E10600;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #E10600 0%, #A00400 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #FF0800 0%, #E10600 100%);
        border: 2px solid #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

# ============ CONSTANTS ============

AVAILABLE_YEARS = [2024, 2023, 2022, 2021, 2020]

# Display name → FastF1 event name
GP_OPTIONS = {
    'Monaco': 'Monaco',
    'British (Silverstone)': 'British',
    'Belgian (Spa)': 'Belgian',
    'Singapore': 'Singapore',
    'Italian (Monza)': 'Italian',
    'Japanese': 'Japanese',
    'Abu Dhabi': 'Abu Dhabi',
    'Hungarian': 'Hungarian',
    'Austrian': 'Austrian',
    'Spanish': 'Spanish',
    'Bahrain': 'Bahrain',
    'Australian': 'Australian',
    'Canadian': 'Canadian',
    'Dutch': 'Dutch',
    'United States': 'United States',
    'Mexico City': 'Mexico City',
    'São Paulo': 'São Paulo',
}

# Official team colors by driver abbreviation
DRIVER_COLORS = {
    'VER': '#3671C6', 'PER': '#3671C6',
    'HAM': '#27F4D2', 'RUS': '#27F4D2',
    'LEC': '#E8002D', 'SAI': '#E8002D',
    'NOR': '#FF8000', 'PIA': '#FF8000',
    'ALO': '#358C75', 'STR': '#358C75',
    'GAS': '#0093CC', 'OCO': '#0093CC',
    'ALB': '#005AFF', 'SAR': '#005AFF', 'COL': '#005AFF',
    'TSU': '#6692FF', 'RIC': '#6692FF', 'LAW': '#6692FF', 'HAD': '#6692FF',
    'BOT': '#C92D4B', 'ZHO': '#C92D4B',
    'MAG': '#B6BABD', 'HUL': '#B6BABD', 'BEA': '#B6BABD',
}
COLOR_CYCLE = ['#E10600', '#00D2BE', '#FFF500', '#FFFFFF', '#FF8700',
               '#2293D1', '#358C75', '#6692FF', '#C92D4B', '#B6BABD']

# ============ SESSION STATE INITIALIZATION ============

if 'race_loaded' not in st.session_state:
    st.session_state.race_loaded = False
    st.session_state.race_year = 2024
    st.session_state.race_gp = 'Monaco'
    st.session_state.race_label = 'Monaco GP 2024'
    st.session_state.ml_model = None
    st.session_state.ml_mae = 0.92
    st.session_state.ml_r2 = 0.319

# ============ CACHED DATA LOADING ============

@st.cache_data(show_spinner=False)
def fetch_race_laps(year, gp):
    """Load FastF1 race laps; result is cached per (year, gp) pair."""
    import fastf1
    fastf1.Cache.enable_cache('f1_cache')
    session = fastf1.get_session(year, gp, 'R')
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    return pd.DataFrame(session.laps)   # plain DataFrame — no FastF1 methods needed

@st.cache_data(show_spinner=False)
def fetch_driver_names(year, gp):
    """Return {abbreviation: full_name} for all drivers in the session.
    session.results is always populated by session.load(); the second call
    is instant because FastF1 serves it from its file cache."""
    import fastf1
    fastf1.Cache.enable_cache('f1_cache')
    session = fastf1.get_session(year, gp, 'R')
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    results = session.results
    if results is None or results.empty:
        return {}
    names = {}
    for _, row in results.iterrows():
        abbr = row.get('Abbreviation', '')
        full = row.get('FullName', '').strip()
        if not full:
            first = row.get('FirstName', '').strip()
            last  = row.get('LastName',  '').strip()
            full  = f"{first} {last}".strip()
        if abbr and full:
            names[abbr] = full
    return names

@st.cache_data(show_spinner=False)
def fetch_race_results(year, gp):
    """Return session.results as a plain DataFrame (cached per race)."""
    import fastf1
    fastf1.Cache.enable_cache('f1_cache')
    session = fastf1.get_session(year, gp, 'R')
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    if session.results is None or session.results.empty:
        return pd.DataFrame()
    # reset_index so the DF has a clean 0-based integer index;
    # session.results uses driver numbers as the index which causes
    # "Unalignable boolean Series" errors when boolean masks built from
    # helper Series (which default to 0-based) are applied to this DF.
    return pd.DataFrame(session.results).reset_index(drop=True)

# ============ CACHED RAG RESOURCES ============

@st.cache_resource
def load_rag_resources():
    import os
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS

    # Prefer Streamlit secrets (cloud); fall back to .env (local dev)
    api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    embeddings = OpenAIEmbeddings(request_timeout=15)
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, request_timeout=20, max_tokens=512)
    return vectorstore, llm

# ============ VISUALIZATION HELPERS ============

def _get_top_drivers(laps, n=3):
    """Return top-N driver abbreviations ordered by finishing position."""
    clean = laps[laps['LapNumber'] > 0].copy()
    if 'Position' not in clean.columns or clean['Position'].isna().all():
        return clean['Driver'].unique()[:n].tolist()
    last = clean.sort_values('LapNumber').groupby('Driver').last()
    last = last.dropna(subset=['Position'])
    last['Position'] = last['Position'].astype(float)
    return last.nsmallest(n, 'Position').index.tolist()

def _driver_color(driver, idx):
    return DRIVER_COLORS.get(driver, COLOR_CYCLE[idx % len(COLOR_CYCLE)])

def _apply_f1_style(fig, ax):
    fig.patch.set_facecolor('#1E1E2E')
    ax.set_facecolor('#15151E')
    ax.tick_params(colors='#C0C0C0')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#38383F')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='#38383F', linestyle='--')

# ---- Race Results helpers ----

_NON_FINISHER = {'DNF', 'DNS', 'NC', 'R', 'W', 'D', 'E', 'N', 'G', 'F', 'DQ'}

def _fmt_time(row, is_winner):
    """Format Time/Status for the results table."""
    classified = str(row.get('ClassifiedPosition', '')).strip().upper()
    status     = str(row.get('Status', '')).strip()
    t          = row.get('Time')

    if classified in _NON_FINISHER:
        return status if status else classified

    if is_winner:
        if pd.isnull(t):
            return '—'
        ts = t.total_seconds()
        h  = int(ts // 3600)
        m  = int((ts % 3600) // 60)
        s  = ts % 60
        return f"{h}:{m:02d}:{s:06.3f}" if h else f"{m}:{s:06.3f}"
    else:
        if pd.isnull(t):
            return status if status not in ('Finished', '') else '—'
        return f"+{t.total_seconds():.3f}s"

def build_results_df(results):
    """Return (display_df, styled_df) for the race results table.

    Uses pandas Styler so Streamlit's st.dataframe() renders proper
    row-level colours without any raw HTML being shown to the user.
    """
    cols_present = set(results.columns)

    def _get(row, key, default=''):
        v = row.get(key, default)
        try:
            return default if pd.isnull(v) else v
        except (TypeError, ValueError):
            return v

    rows      = []
    row_tags  = []   # 'gold' | 'silver' | 'bronze' | 'points' | 'dnf' | ''

    _POS_EMOJI = {'1': '🥇  P1', '2': '🥈  P2', '3': '🥉  P3'}

    for _, row in results.iterrows():
        classified = str(_get(row, 'ClassifiedPosition', _get(row, 'Position', ''))).strip()
        if not classified:
            classified = 'NC'
        c = classified.upper()

        is_winner = (c == '1')
        is_dnf    = c in _NON_FINISHER

        # Position display
        if c in _NON_FINISHER:
            pos_str = c
        else:
            pos_str = _POS_EMOJI.get(c, f'P{c}')

        # Full name
        full_name = str(_get(row, 'FullName', '')).strip()
        if not full_name:
            first     = str(_get(row, 'FirstName', '')).strip()
            last      = str(_get(row, 'LastName',  '')).strip()
            full_name = f"{first} {last}".strip() or str(_get(row, 'Abbreviation', '—'))

        # Fastest lap — append ⚡ to driver name (no HTML needed)
        fl_raw  = _get(row, 'FastestLap',     '')
        fl_rank = _get(row, 'FastestLapRank', '')
        is_fl   = (str(fl_raw).strip()  in ('True', '1', 'true') or
                   str(fl_rank).strip() == '1')
        driver_str = f"{full_name}  ⚡" if is_fl else full_name

        # Team
        team = str(_get(row, 'TeamName', '—'))

        # Time / gap
        time_str = _fmt_time(row, is_winner)

        # Points
        pts_raw = _get(row, 'Points', '')
        try:
            pts     = int(float(pts_raw))
            pts_str = str(pts) if pts > 0 else ('—' if is_dnf else '0')
        except (ValueError, TypeError):
            pts_str = '—'

        rows.append({
            'Pos':        pos_str,
            'Driver':     driver_str,
            'Team':       team,
            'Time / Gap': time_str,
            'Pts':        pts_str,
        })

        # Tag for row colouring
        if c == '1':
            row_tags.append('gold')
        elif c == '2':
            row_tags.append('silver')
        elif c == '3':
            row_tags.append('bronze')
        elif is_dnf:
            row_tags.append('dnf')
        else:
            try:
                row_tags.append('points' if int(c) <= 10 else '')
            except ValueError:
                row_tags.append('')

    display_df = pd.DataFrame(rows)

    # Row-level background colours via pandas Styler
    _ROW_BG = {
        'gold':   'background-color: #3a3000; color: #FFD700;',
        'silver': 'background-color: #2a2a2a; color: #D0D0D0;',
        'bronze': 'background-color: #2d1f0e; color: #CD9B5A;',
        'points': 'background-color: #1a1e2e;',
        'dnf':    'background-color: #1e1212; color: #777777;',
        '':       '',
    }

    def _style_row(row):
        tag = row_tags[row.name] if row.name < len(row_tags) else ''
        css = _ROW_BG.get(tag, '')
        return [css] * len(row)

    styled = display_df.style.apply(_style_row, axis=1)
    return styled

# ---- End Race Results helpers ----

def make_driver_fig(laps, year, gp_display):
    """Multi-driver lap time comparison chart."""
    clean = laps[(laps['LapNumber'] > 0) & laps['LapTime'].notna()].copy()
    top_drivers = _get_top_drivers(clean)

    fig, ax = plt.subplots(figsize=(14, 7))
    _apply_f1_style(fig, ax)

    for idx, driver in enumerate(top_drivers):
        d = clean[clean['Driver'] == driver]
        if d.empty:
            continue
        ax.plot(d['LapNumber'], d['LapTime'].dt.total_seconds(),
                color=_driver_color(driver, idx), linewidth=2.5,
                marker='o', markersize=3, label=driver, alpha=0.85)

    ax.set_title(f'{gp_display} {year} - Driver Comparison',
                 fontsize=18, fontweight='bold', color='white', pad=20)
    ax.set_xlabel('Lap Number', fontsize=13, color='#C0C0C0')
    ax.set_ylabel('Lap Time (seconds)', fontsize=13, color='#C0C0C0')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9,
              facecolor='#1E1E2E', edgecolor='#E10600', labelcolor='white')
    fig.tight_layout()
    return fig


def make_heatmap_fig(laps, year, gp_display):
    """Tire degradation heatmap."""
    clean = laps[(laps['LapNumber'] > 0) & laps['LapTime'].notna()
                 & laps['Compound'].notna()].copy()

    compounds = [c for c in ['SOFT', 'MEDIUM', 'HARD'] if c in clean['Compound'].unique()]
    if not compounds:
        return None

    max_lap = int(clean['LapNumber'].max())
    heatmap_data = np.full((len(compounds), max_lap), np.nan)

    for idx, compound in enumerate(compounds):
        comp = clean[clean['Compound'] == compound]
        for lap_num in range(1, max_lap + 1):
            times = comp[comp['LapNumber'] == lap_num]['LapTime'].dt.total_seconds()
            valid = times[times < 200]
            if not valid.empty:
                heatmap_data[idx, lap_num - 1] = valid.mean()

    valid_vals = heatmap_data[~np.isnan(heatmap_data)]
    vmin = np.percentile(valid_vals, 5) if len(valid_vals) else 75
    vmax = np.percentile(valid_vals, 95) if len(valid_vals) else 95

    # x-tick labels: lap number every 5 laps, blank otherwise
    xlabels = [str(i) if i % 5 == 0 else '' for i in range(1, max_lap + 1)]

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor('#1E1E2E')
    ax.set_facecolor('#15151E')
    sns.heatmap(heatmap_data, cmap='RdYlGn_r', ax=ax,
                cbar_kws={'label': 'Avg Lap Time (s)'},
                yticklabels=compounds, xticklabels=xlabels,
                linewidths=0.3, linecolor='#15151E',
                vmin=vmin, vmax=vmax)
    ax.set_title(f'{gp_display} {year} - Tire Degradation Heatmap',
                 fontsize=18, fontweight='bold', color='#FFFFFF', pad=20)
    ax.set_xlabel('Lap Number', fontsize=12, color='#C0C0C0')
    ax.set_ylabel('Tire Compound', fontsize=12, color='#C0C0C0')
    ax.tick_params(colors='#C0C0C0')
    # Colorbar text contrast
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color('#C0C0C0')
    cbar.ax.tick_params(colors='#C0C0C0')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def make_strategy_fig(laps, year, gp_display, driver_names=None):
    """Tire strategy chart for the race winner."""
    clean = laps[(laps['LapNumber'] > 0) & laps['LapTime'].notna()].copy()
    top = _get_top_drivers(clean, n=1)
    driver = top[0] if top else clean['Driver'].iloc[0]

    # Resolve display name: prefer full name from results, fall back to code
    full_name = (driver_names or {}).get(driver, '').strip()
    driver_display = full_name if full_name else driver

    d = clean[clean['Driver'] == driver].sort_values('LapNumber')
    lap_numbers = d['LapNumber'].values
    lap_times   = d['LapTime'].dt.total_seconds().values
    compounds   = d['Compound'].values

    compound_colors = {'SOFT': '#DC0000', 'MEDIUM': '#FFD700', 'HARD': '#FFFFFF'}

    fig, ax = plt.subplots(figsize=(14, 8))
    _apply_f1_style(fig, ax)

    for compound in dict.fromkeys(c for c in compounds if pd.notna(c)):
        mask = compounds == compound
        ax.scatter(lap_numbers[mask], lap_times[mask],
                   c=compound_colors.get(compound, '#C0C0C0'),
                   label=f'{compound} tire', s=80, alpha=0.9,
                   edgecolors='#555555', linewidths=0.5, zorder=3)

    ax.plot(lap_numbers, lap_times, color='#3671C6',
            linewidth=1.5, alpha=0.4, zorder=2)

    # Mark pit stops (compound changes)
    pit_laps = [lap_numbers[i] for i in range(1, len(compounds))
                if pd.notna(compounds[i]) and pd.notna(compounds[i-1])
                and compounds[i] != compounds[i-1]]
    y_max = lap_times.max() if len(lap_times) else 100
    for i, pit in enumerate(pit_laps):
        ax.axvline(x=pit, color='#00D2BE', linestyle='--', linewidth=2, alpha=0.7,
                   label='Pit stop' if i == 0 else None)
        ax.text(pit, y_max * 0.97, f'PIT\nLap {int(pit)}', color='#00D2BE',
                fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#1E1E2E', edgecolor='#00D2BE'))

    ax.set_title(f'{driver_display} - {gp_display} {year} - Tire Strategy',
                 fontsize=16, fontweight='bold', color='white', pad=20)
    ax.set_xlabel('Lap Number', fontsize=13, color='#C0C0C0')
    ax.set_ylabel('Lap Time (seconds)', fontsize=13, color='#C0C0C0')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95,
              facecolor='#1E1E2E', edgecolor='#E10600', labelcolor='white')
    fig.tight_layout()
    return fig

# ============ IN-APP ML TRAINING ============

def train_lap_model(laps):
    """Train RandomForest on the loaded race; returns (model, mae, r2)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    df = laps[(laps['LapNumber'] > 0) & laps['LapTime'].notna()
              & laps['Compound'].notna()].copy()
    df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
    df = df[df['LapTimeSeconds'] < 200]

    compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
    df['CompoundEncoded'] = df['Compound'].map(compound_map).fillna(1)
    df['TireAge']  = df['TyreLife'] if 'TyreLife' in df.columns else 0
    df['Position'] = df['Position'].fillna(10) if 'Position' in df.columns else 10
    df['LapNum']   = df['LapNumber']

    features = ['CompoundEncoded', 'TireAge', 'Position', 'LapNum']
    X = df[features].fillna(df[features].mean())
    y = df['LapTimeSeconds']

    if len(X) < 20:
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=15,
                                   min_samples_split=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)

# ============ CHATBOT KNOWLEDGE UPDATE ============

def _generate_race_knowledge(laps, year, gp_display):
    """Build a plain-text knowledge document from race laps."""
    clean = laps[(laps['LapNumber'] > 0) & laps['LapTime'].notna()].copy()
    total_laps   = int(clean['LapNumber'].max()) if not clean.empty else 0
    compounds    = clean['Compound'].dropna().unique().tolist()
    top3         = _get_top_drivers(clean, n=3)
    times        = clean['LapTime'].dt.total_seconds()
    valid_times  = times[times < 200]
    fastest      = valid_times.min() if not valid_times.empty else 0
    avg_time     = valid_times.mean() if not valid_times.empty else 0

    return f"""{gp_display} Grand Prix {year} - Race Analysis

Race Summary:
- Event: {gp_display} Grand Prix {year}
- Total Race Laps: {total_laps}
- Tire Compounds Used: {', '.join(str(c) for c in compounds)}
- Fastest Lap: {fastest:.2f}s
- Average Lap Time: {avg_time:.2f}s

Top Finishers: {', '.join(top3)}

This data was sourced from the FastF1 API for the {gp_display} {year} Formula 1 race.
The ML model was retrained on {len(clean)} laps from this race.
"""

def rebuild_chatbot_index(laps, year, gp_display, gp_fastf1):
    """Rebuild FAISS index with a new race-knowledge document appended."""
    import os
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    os.makedirs('race_knowledge', exist_ok=True)
    fname = f'race_knowledge/{gp_fastf1.lower().replace(" ", "_")}_{year}.txt'
    with open(fname, 'w') as f:
        f.write(_generate_race_knowledge(laps, year, gp_display))

    loader = DirectoryLoader('race_knowledge/', glob="*.txt", loader_cls=TextLoader)
    docs   = loader.load()
    texts  = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

    embeddings  = OpenAIEmbeddings(request_timeout=15)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("faiss_index")
    load_rag_resources.clear()   # force reload on next chatbot use

# ============ SIDEBAR NAVIGATION ============

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🏁 NAVIGATION")

    page = st.radio(
        "Select Module:",
        ["🏠 Overview", "📊 Race Analysis", "🔥 Tire Heatmap",
         "⚔️ Strategy Comparison", "🏁 Race Results", "🤖 Lap Predictor", "💬 AI Race Engineer"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ SESSION INFO")
    race_label = st.session_state.get('race_label', 'Monaco GP 2024')
    st.info(f"**Track:** {race_label}\n**Session:** Race")

    st.markdown("---")
    st.markdown("### 📌 ABOUT")
    st.caption("F1 Race Insights AI - Built by Brinda Gandhi\n\n"
               "End-to-end analytics platform combining data science, ML, and AI.")

# ============ MAIN HEADER ============

col1, col2 = st.columns([4, 1])
with col1:
    st.markdown("# 🏎️ F1 RACE INSIGHTS AI")
    st.markdown("### Advanced Telemetry & Performance Analysis Platform")
with col2:
    st.markdown(
        '<div style="background: #E10600; color: white; padding: 8px 16px; '
        'border-radius: 4px; text-align: center; font-weight: bold; '
        'font-size: 14px; margin-top: 20px;">● LIVE</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ============ PAGE 1: OVERVIEW ============

if page == "🏠 Overview":
    st.header("📊 DASHBOARD OVERVIEW")

    # --- Dynamic metrics ---
    if st.session_state.race_loaded:
        laps_meta = fetch_race_laps(st.session_state.race_year, st.session_state.race_gp)
        total_laps_count = int(laps_meta['LapNumber'].max()) if not laps_meta.empty else 0
        r2_pct  = f"{st.session_state.ml_r2 * 100:.1f}%" if st.session_state.ml_r2 is not None else "N/A"
        mae_val = f"±{st.session_state.ml_mae:.2f}s"      if st.session_state.ml_mae is not None else "N/A"
    else:
        total_laps_count = 245
        r2_pct  = "31.9%"
        mae_val = "±0.92s"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏁 RACE LOADED", st.session_state.race_label)
    with col2:
        st.metric("⏱️ TOTAL LAPS", str(total_laps_count))
    with col3:
        st.metric("🤖 MODEL ACCURACY", r2_pct, mae_val + " precision")
    with col4:
        st.metric("📈 VISUALIZATIONS", "5", "Ready to view")

    st.markdown("---")

    # ============ RACE SELECTOR ============
    st.subheader("🔄 RACE SELECTOR")
    st.markdown("Select a race and click **Load Race Data** to refresh all pages with new data.")

    sel_col1, sel_col2, sel_col3 = st.columns(3)
    with sel_col1:
        selected_year = st.selectbox("📅 Year", AVAILABLE_YEARS, index=0)
    with sel_col2:
        gp_display_list = list(GP_OPTIONS.keys())
        selected_gp_display = st.selectbox("🏟️ Grand Prix", gp_display_list, index=0)
    with sel_col3:
        st.selectbox("🏁 Session", ["Race"], index=0)

    update_chatbot = st.checkbox("Also update AI chatbot knowledge base (requires OpenAI API)", value=False)

    if st.button("🏁 Load Race Data"):
        selected_gp_fastf1 = GP_OPTIONS[selected_gp_display]
        new_label = f"{selected_gp_display} {selected_year}"

        # Step 1: Fetch race data
        with st.spinner(f"⏳ Fetching {new_label} from FastF1 (may take 30-60s on first load)..."):
            try:
                laps = fetch_race_laps(selected_year, selected_gp_fastf1)
            except Exception as e:
                st.error(f"⚠️ Could not load race data: {e}")
                st.stop()

        st.success(f"✅ Race data loaded: {len(laps)} laps across {laps['Driver'].nunique()} drivers.")

        # Step 2: Retrain ML model
        with st.spinner("🤖 Retraining ML model on new race data..."):
            model, mae, r2 = train_lap_model(laps)

        if model is not None:
            st.session_state.ml_model = model
            st.session_state.ml_mae   = mae
            st.session_state.ml_r2    = r2
            # Persist to disk so the predictor page can also load it
            with open('lap_time_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            st.success(f"✅ ML model retrained — R²: {r2:.3f}, MAE: ±{mae:.2f}s")
        else:
            st.warning("⚠️ Not enough clean laps to retrain the model.")

        # Step 3: Optionally rebuild chatbot index
        if update_chatbot:
            with st.spinner("💬 Rebuilding chatbot knowledge base (calls OpenAI API)..."):
                try:
                    rebuild_chatbot_index(laps, selected_year, selected_gp_display, selected_gp_fastf1)
                    st.success("✅ Chatbot knowledge base updated.")
                except Exception as e:
                    st.warning(f"⚠️ Chatbot update failed (check OpenAI API key): {e}")

        # Commit to session state
        st.session_state.race_year   = selected_year
        st.session_state.race_gp     = selected_gp_fastf1
        st.session_state.race_label  = new_label
        st.session_state.race_loaded = True

        st.info("🏎️ Navigate to Race Analysis, Tire Heatmap, Strategy Comparison, and Lap Predictor pages to see the updated visualizations.")

    st.markdown("---")

    # Project overview
    st.subheader("🎯 PROJECT OVERVIEW")
    st.markdown(f"""
    **F1 Race Insights AI** is an end-to-end data analytics platform that analyzes Formula 1 race performance
    using machine learning and interactive visualizations.

    **Key Features:**
    - 📊 **Race Analysis:** Lap-by-lap telemetry and performance tracking
    - 🔥 **Tire Degradation Heatmap:** Visual analysis of tire performance over race distance
    - ⚔️ **Strategy Comparison:** Multi-driver strategy analysis with pit stop detection
    - 🤖 **ML Lap Predictor:** Random Forest model predicting lap times

    **Currently Loaded:** {st.session_state.race_label}

    **Tech Stack:** Python, FastF1, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit
    """)

    if not st.session_state.race_loaded:
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            **🔴 KEY INSIGHT: Tire Strategy**

            Monaco 2024 saw most drivers opt for a 1-stop strategy, starting on mediums
            and finishing on hards. Verstappen's late pit stop (Lap 53) proved optimal.
            """)
        with col2:
            st.success("""
            **✅ MODEL PERFORMANCE**

            The ML model achieves ±0.92 second accuracy using only 4 features:
            tire compound, tire age, track position, and lap number.
            """)

# ============ PAGE 2: RACE ANALYSIS ============

elif page == "📊 Race Analysis":
    st.header("📊 RACE ANALYSIS - LAP BY LAP")

    if st.session_state.race_loaded:
        year, gp, label = st.session_state.race_year, st.session_state.race_gp, st.session_state.race_label
        st.markdown(f"Analyzing lap times for the top 3 finishers — **{label}**.")

        with st.spinner("Generating driver comparison chart..."):
            laps = fetch_race_laps(year, gp)
            fig  = make_driver_fig(laps, year, label.replace(str(year), '').strip())
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.markdown("Analyzing lap times for Max Verstappen, Charles Leclerc, and Lando Norris — Monaco GP 2024.")
        try:
            img = Image.open('driver_comparison.png')
            st.image(img, caption='Multi-Driver Lap Time Comparison', use_container_width=True)
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("**🔵 MAX VERSTAPPEN**\n- Consistent pace throughout\n- Winner of Monaco GP 2024\n- Strategy: Medium → Hard")
            with col2:
                st.error("**🔴 CHARLES LECLERC**\n- Home race advantage\n- Ferrari strategy\n- Close competition with Verstappen")
            with col3:
                st.warning("**🟠 LANDO NORRIS**\n- McLaren pace\n- Strong midfield performance\n- Consistent lap times")
        except FileNotFoundError:
            st.error("⚠️ Chart not found. Load a race on the Overview page or run `compare_drivers.py` first.")

# ============ PAGE 3: TIRE HEATMAP ============

elif page == "🔥 Tire Heatmap":
    st.header("🔥 TIRE DEGRADATION HEATMAP")
    st.markdown("Visual analysis showing average lap times by tire compound. **Green = Fast**, **Red = Slow**.")

    if st.session_state.race_loaded:
        year, gp, label = st.session_state.race_year, st.session_state.race_gp, st.session_state.race_label
        with st.spinner("Generating tire heatmap..."):
            laps = fetch_race_laps(year, gp)
            fig  = make_heatmap_fig(laps, year, label.replace(str(year), '').strip())
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("⚠️ Not enough compound data to generate a heatmap for this race.")

        st.markdown("---")
        st.subheader("📈 KEY INSIGHTS")
        laps_clean = laps[(laps['LapNumber'] > 0) & laps['LapTime'].notna() & laps['Compound'].notna()]
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            if compound in laps_clean['Compound'].unique():
                comp_laps = laps_clean[laps_clean['Compound'] == compound]
                laps_range = f"Laps {int(comp_laps['LapNumber'].min())}–{int(comp_laps['LapNumber'].max())}"
                avg_t = comp_laps['LapTime'].dt.total_seconds()
                avg_t = avg_t[avg_t < 200].mean()
                st.metric(f"{compound} Tires", laps_range, f"Avg: {avg_t:.2f}s")
    else:
        try:
            img = Image.open('tire_degradation_heatmap.png')
            st.image(img, caption='Tire Performance Heatmap', use_container_width=True)
            st.markdown("---")
            st.subheader("📈 KEY INSIGHTS")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 SOFT TIRES",   "Laps 1-40",  "Fast but short-lived")
            with col2:
                st.metric("🟡 MEDIUM TIRES", "Laps 1-60",  "Best versatility")
            with col3:
                st.metric("⚪ HARD TIRES",   "Laps 50-78", "Most consistent")
            st.info("**💡 STRATEGIC INSIGHT:**\n\nThe heatmap reveals that hard tires maintained the most consistent performance.")
        except FileNotFoundError:
            st.error("⚠️ Heatmap not found. Load a race on the Overview page or run `tire_heatmap.py` first.")

# ============ PAGE 4: STRATEGY COMPARISON ============

elif page == "⚔️ Strategy Comparison":
    st.header("⚔️ TIRE STRATEGY ANALYSIS")

    if st.session_state.race_loaded:
        year, gp, label = st.session_state.race_year, st.session_state.race_gp, st.session_state.race_label
        with st.spinner("Generating strategy chart..."):
            laps         = fetch_race_laps(year, gp)
            driver_names = fetch_driver_names(year, gp)
            winner       = _get_top_drivers(laps, n=1)[0]
            winner_name  = driver_names.get(winner, winner)
            fig          = make_strategy_fig(laps, year, label.replace(str(year), '').strip(), driver_names)
        st.markdown(f"Detailed tire strategy analysis for **{winner_name}** — **{label}**.")
        st.pyplot(fig)
        plt.close(fig)

        # Summarise stints
        st.markdown("---")
        d = laps[(laps['Driver'] == winner) & (laps['LapNumber'] > 0)
                 & laps['Compound'].notna()].sort_values('LapNumber')
        stints, current = [], None
        for _, row in d.iterrows():
            comp = row['Compound']
            if comp != current:
                stints.append({'compound': comp, 'start': int(row['LapNumber']), 'end': int(row['LapNumber'])})
                current = comp
            else:
                stints[-1]['end'] = int(row['LapNumber'])

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**✅ {winner} Strategy**\n\n" +
                       "\n".join(f"- **{s['compound']}:** Laps {s['start']}–{s['end']}" for s in stints))
        with col2:
            pit_count = len(stints) - 1
            st.info(f"**🔧 Pit Stop Summary**\n\n- Pit stops: {pit_count}\n- Stints: {len(stints)}")
    else:
        st.markdown("Detailed analysis of Max Verstappen's tire strategy — Monaco GP 2024.")
        try:
            img = Image.open('tire_strategy_analysis.png')
            st.image(img, caption='Verstappen Tire Strategy - Monaco GP 2024', use_container_width=True)
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.success("**✅ WINNING STRATEGY**\n\n- **Start:** Medium tires (Laps 1-53)\n- **Pit Stop:** Lap 53\n- **Finish:** Hard tires (Laps 54-78)")
            with col2:
                st.info("**🔧 PIT STOP ANALYSIS**\n\n- **Timing:** Lap 53 (late strategy)\n- **Compound Change:** Medium → Hard\n- **Remaining Laps:** 25")
        except FileNotFoundError:
            st.error("⚠️ Strategy chart not found. Load a race on the Overview page or run `tire_strategy_chart.py` first.")

# ============ PAGE 5: RACE RESULTS ============

elif page == "🏁 Race Results":
    st.header(f"🏁 RACE RESULTS — {st.session_state.race_label.upper()}")

    if not st.session_state.race_loaded:
        st.info("👆 Select a race on the **Overview** page and click **Load Race Data** to see the official results.")
        st.stop()

    year, gp, label = st.session_state.race_year, st.session_state.race_gp, st.session_state.race_label

    with st.spinner("Fetching official results..."):
        results = fetch_race_results(year, gp)

    if results.empty:
        st.error("⚠️ No results data available for this race. Try a different year or event.")
        st.stop()

    # ---- Summary metrics ----
    cols_present = set(results.columns)

    def _safe_get_col(df, *keys):
        for k in keys:
            if k in df.columns:
                return df[k]
        return pd.Series([''] * len(df))

    classified_col = _safe_get_col(results, 'ClassifiedPosition', 'Position')
    classified_clean = classified_col.astype(str).str.strip().str.upper()

    finishers  = classified_clean[~classified_clean.isin(_NON_FINISHER)].count()
    dnf_count  = classified_clean[classified_clean.isin(_NON_FINISHER)].count()

    # Winner name
    winner_row  = results[classified_clean == '1']
    if not winner_row.empty:
        wr = winner_row.iloc[0]
        winner_name = str(wr.get('FullName', '') or
                          f"{wr.get('FirstName','')} {wr.get('LastName','')}").strip()
        if not winner_name:
            winner_name = str(wr.get('Abbreviation', '—'))
    else:
        winner_name = '—'

    # Fastest lap holder
    fl_col  = _safe_get_col(results, 'FastestLapRank', 'FastestLap')
    fl_row  = results[(fl_col.astype(str).str.strip().isin(['1', 'True', 'true']))]
    if not fl_row.empty:
        fr = fl_row.iloc[0]
        fl_name = str(fr.get('FullName', '') or
                      f"{fr.get('FirstName','')} {fr.get('LastName','')}").strip()
        fl_time_raw = fr.get('FastestLapTime', None)
        fl_time_str = ''
        if fl_time_raw is not None and not pd.isnull(fl_time_raw):
            ts = fl_time_raw.total_seconds()
            fl_time_str = f"  {int(ts//60)}:{ts%60:06.3f}"
        fl_display = f"{fl_name}{fl_time_str}" if fl_name else '—'
    else:
        fl_display = '—'

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("🏆 Winner",        winner_name)
    with m2:
        st.metric("✅ Classified",    str(finishers))
    with m3:
        st.metric("❌ DNF / DNS",     str(dnf_count))
    with m4:
        st.metric("⚡ Fastest Lap",   fl_display)

    st.markdown("---")

    # ---- Legend ----
    st.markdown(
        '<div style="display:flex;gap:20px;margin-bottom:12px;font-size:13px;color:#888;">'
        '<span><span style="background:#FFD700;color:#111;padding:1px 7px;border-radius:3px;font-weight:700;">P1</span> '
        ' Podium</span>'
        '<span style="color:#00D2BE;font-weight:700;">Pts</span> Points-scoring (P1–P10)'
        '<span style="color:#A855F7;">⚡</span> Fastest lap'
        '<span style="background:#E10600;color:#fff;padding:1px 7px;border-radius:3px;font-weight:700;font-size:12px;">DNF</span>'
        ' Not classified'
        '</div>',
        unsafe_allow_html=True
    )

    # ---- Main table ----
    styled_df = build_results_df(results)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(f"Official classification — {label} | Data: FastF1 / Ergast API")

# ============ PAGE 6: ML LAP PREDICTOR ============

elif page == "🤖 Lap Predictor":
    st.header("🤖 LAP TIME PREDICTOR - ML MODEL")

    race_label = st.session_state.race_label
    r2_val  = st.session_state.ml_r2  if st.session_state.ml_r2  is not None else 0.319
    mae_val = st.session_state.ml_mae if st.session_state.ml_mae is not None else 0.92

    st.markdown(f"""
    Machine Learning model trained on **{race_label}** data to predict lap times
    based on tire compound, tire age, track position, and lap number.
    """)

    try:
        img = Image.open('ml_model_performance.png')
        st.image(img, caption='ML Model Performance Visualization', use_container_width=True)
    except FileNotFoundError:
        pass  # No static image; show metrics below regardless

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 R² Score", f"{r2_val:.3f}", f"{r2_val*100:.1f}% accuracy")
    with col2:
        st.metric("📉 Mean Error", f"±{mae_val:.2f}s", "Average prediction error")
    with col3:
        st.metric("🌲 Model Type", "Random Forest", "100 estimators")

    st.markdown("---")
    st.subheader("🔮 MAKE A PREDICTION")

    col1, col2 = st.columns(2)
    with col1:
        compound  = st.selectbox("🛞 Tire Compound", ["SOFT", "MEDIUM", "HARD"])
        tire_age  = st.slider("⏱️ Tire Age (laps)", 0, 40, 10)
    with col2:
        position  = st.slider("📍 Track Position", 1, 20, 5)
        lap_num   = st.slider("🏁 Lap Number", 1, 78, 40)

    if st.button("🚀 PREDICT LAP TIME"):
        # Prefer the in-session model; fall back to saved .pkl
        model = st.session_state.ml_model
        if model is None:
            try:
                with open('lap_time_model.pkl', 'rb') as f:
                    model = pickle.load(f)
            except FileNotFoundError:
                st.error("⚠️ No model found. Load a race on the Overview page or run `ml_lap_predictor.py` first.")
                model = None

        if model is not None:
            compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
            input_data = pd.DataFrame({
                'CompoundEncoded': [compound_map[compound]],
                'TireAge':         [tire_age],
                'Position':        [position],
                'LapNum':          [lap_num]
            })
            prediction = model.predict(input_data)[0]
            st.success(f"""
            ### 🏎️ PREDICTED LAP TIME: **{prediction:.2f} seconds**

            **Input Parameters:**
            - Tire: {compound} ({tire_age} laps old)
            - Position: P{position}
            - Lap: {lap_num}

            **Model trained on:** {race_label} | **MAE:** ±{mae_val:.2f}s
            """)

# ============ PAGE 6: AI CHATBOT ============

elif page == "💬 AI Race Engineer":
    st.header("💬 AI RACE ENGINEER - STRATEGY ADVISOR")
    st.markdown("""
    Ask questions about F1 race strategy, tire performance, pit stops, and driver analysis.
    Powered by RAG (Retrieval-Augmented Generation) with OpenAI GPT-3.5.
    """)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about tire strategy, lap times, pit windows..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🏎️ Race Engineer analyzing..."):
                try:
                    vectorstore, llm = load_rag_resources()
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                    docs      = retriever.invoke(prompt)
                    context   = "\n\n".join([doc.page_content for doc in docs])

                    full_prompt = f"""You are an expert F1 race engineer. Use the following context to answer the question.

Context:
{context}

Question: {prompt}

Answer as a knowledgeable F1 race engineer, keeping responses concise and technical."""

                    response = llm.invoke(full_prompt)
                    answer   = response.content
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}\n\nMake sure you've run `chatbot_setup.py` first!")

# ============ FOOTER ============

st.markdown("---")
st.caption("F1 Race Insights AI | Built with Python, FastF1, Scikit-learn & Streamlit | © 2026 Brinda Gandhi")
