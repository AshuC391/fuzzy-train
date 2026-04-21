import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroSignal · PD Monitor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design tokens ──────────────────────────────────────────────────────────────
C = {
    "bg":       "#0D0F14",
    "surface":  "#13161E",
    "surface2": "#1A1E29",
    "border":   "#252A38",
    "text":     "#E8EAF0",
    "muted":    "#6B7280",
    "accent":   "#4F8EF7",
    "high":     "#F26B6B",
    "mod":      "#F5A623",
    "low":      "#4EC994",
    "voice":    "#A78BFA",
    "gait":     "#FB923C",
    "tap":      "#34D399",
}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{
      font-family: 'DM Sans', sans-serif;
      background-color: {C["bg"]};
      color: {C["text"]};
  }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  [data-testid="stDecoration"], [data-testid="stToolbar"] {{ display: none; }}
  .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}

  [data-testid="stSidebar"] {{
      background-color: {C["surface"]};
      border-right: 1px solid {C["border"]};
  }}
  [data-testid="stSidebar"] * {{ color: {C["text"]} !important; }}
  [data-testid="stSidebar"] .stSlider > label {{
      font-size: 0.7rem !important;
      font-weight: 500 !important;
      letter-spacing: 0.05em !important;
      text-transform: uppercase !important;
      color: {C["muted"]} !important;
  }}

  .section-label {{
      font-size: 0.65rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: {C["muted"]};
      margin-bottom: 0.5rem;
  }}
  hr {{ border: none; border-top: 1px solid {C["border"]}; margin: 1.2rem 0; }}
  [data-testid="stDataFrame"] {{
      border: 1px solid {C["border"]};
      border-radius: 10px;
      overflow: hidden;
  }}
</style>
""", unsafe_allow_html=True)


# ── Matplotlib theme ───────────────────────────────────────────────────────────
def theme(fig):
    fig.patch.set_facecolor(C["surface"])
    for ax in fig.axes:
        ax.set_facecolor(C["surface"])
        ax.tick_params(colors=C["muted"], labelsize=8.5)
        ax.xaxis.label.set_color(C["muted"])
        ax.yaxis.label.set_color(C["muted"])
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color=C["border"], linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)


def mlegend(ax):
    return ax.legend(fontsize=8.5, frameon=True,
                     facecolor=C["surface2"], edgecolor=C["border"],
                     labelcolor=C["text"])


# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def build_dataset():
    np.random.seed(42)

    def vrow(pd_):
        b = {'v_vocal_freq_mean': np.random.normal(145 if pd_ else 155, 25 if pd_ else 22),
             'v_vocal_freq_max':  np.random.normal(180 if pd_ else 210, 30 if pd_ else 35),
             'v_vocal_freq_min':  np.random.normal(110 if pd_ else 120, 20 if pd_ else 18),
             'v_jitter':         (np.random.normal(0.008, 0.004) if pd_ else np.random.normal(0.003, 0.002))
                                  + np.random.normal(0, 0.002 if pd_ else 0.001),
             'v_shimmer':         np.random.normal(0.045 if pd_ else 0.020, 0.02 if pd_ else 0.010),
             'v_noise_to_harmonics': np.random.normal(0.025 if pd_ else 0.010, 0.012 if pd_ else 0.006),
             'v_harmonics_to_noise': np.random.normal(19 if pd_ else 25, 4),
             'v_rpde':            np.random.normal(0.52 if pd_ else 0.41, 0.08 if pd_ else 0.07),
             'v_dfa':             np.random.normal(0.72 if pd_ else 0.64, 0.06 if pd_ else 0.05),
             'v_spread1':         np.random.normal(-5.8 if pd_ else -6.8, 0.9 if pd_ else 0.8),
             'v_pitch_entropy':   np.random.normal(0.28 if pd_ else 0.18, 0.06 if pd_ else 0.05),
             'label': int(pd_)}
        return b

    def grow(pd_):
        return {'g_stride_time_mean': np.random.normal(1.18 if pd_ else 1.10, 0.18 if pd_ else 0.14),
                'g_stride_time_std':  np.random.normal(0.035 if pd_ else 0.022, 0.015 if pd_ else 0.012),
                'g_swing_time_mean':  np.random.normal(0.40 if pd_ else 0.41, 0.06 if pd_ else 0.05),
                'g_swing_time_std':   np.random.normal(0.016 if pd_ else 0.012, 0.008 if pd_ else 0.006),
                'g_stance_time_mean': np.random.normal(0.82 if pd_ else 0.70, 0.14 if pd_ else 0.12),
                'g_cadence':          np.random.normal(95 if pd_ else 103, 12 if pd_ else 10),
                'g_gait_asymmetry':   np.random.normal(0.055 if pd_ else 0.035, 0.03 if pd_ else 0.022),
                'g_gait_variability_cv': np.random.normal(0.042 if pd_ else 0.028, 0.018 if pd_ else 0.014),
                'g_freeze_of_gait_index': np.random.normal(1.6 if pd_ else 1.1, 0.6 if pd_ else 0.5),
                'g_step_length_mean': np.random.normal(0.55 if pd_ else 0.63, 0.10 if pd_ else 0.09),
                'label': int(pd_)}

    def trow(pd_):
        return {'t_tap_mean_iti':          np.random.normal(0.32 if pd_ else 0.26, 0.08 if pd_ else 0.07),
                't_tap_std_iti':           np.random.normal(0.048 if pd_ else 0.028, 0.022 if pd_ else 0.016),
                't_tap_deceleration':      np.random.normal(0.09 if pd_ else 0.05, 0.05 if pd_ else 0.04),
                't_tap_acceleration_cv':   np.random.normal(0.14 if pd_ else 0.09, 0.06 if pd_ else 0.05),
                't_tap_total_count':       np.random.normal(82 if pd_ else 100, 16 if pd_ else 14),
                't_tap_rhythm_regularity': np.random.normal(0.70 if pd_ else 0.82, 0.12 if pd_ else 0.10),
                'label': int(pd_)}

    vdf = pd.DataFrame([vrow(True)  for _ in range(100)] +
                       [vrow(False) for _ in range(100)]).reset_index(drop=True)
    gdf = pd.DataFrame([grow(True)  for _ in range(100)] +
                       [grow(False) for _ in range(100)]).reset_index(drop=True)
    tdf = pd.DataFrame([trow(True)  for _ in range(100)] +
                       [trow(False) for _ in range(100)]).reset_index(drop=True)

    labels = vdf['label'].reset_index(drop=True)
    X = pd.concat([vdf.drop('label', axis=1).reset_index(drop=True),
                   gdf.drop('label', axis=1).reset_index(drop=True),
                   tdf.drop('label', axis=1).reset_index(drop=True)], axis=1)

    np.random.seed(42)
    vc = [c for c in X.columns if c.startswith('v_')]
    gc = [c for c in X.columns if c.startswith('g_')]
    tc = [c for c in X.columns if c.startswith('t_')]
    X[vc] += np.random.normal(0, 0.5 * X[vc].std(), X[vc].shape)
    X[gc] += np.random.normal(0, 0.3 * X[gc].std(), X[gc].shape)
    X[tc] += np.random.normal(0, 0.3 * X[tc].std(), X[tc].shape)
    return X, labels


@st.cache_resource
def train_model(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, max_features=0.5,
                                min_samples_leaf=6, random_state=42, class_weight='balanced')
    rf.fit(sc.fit_transform(Xtr), ytr)
    yp  = rf.predict_proba(sc.transform(Xte))[:, 1]
    auc = roc_auc_score(yte, yp)
    rep = classification_report(yte, rf.predict(sc.transform(Xte)),
                                target_names=['Healthy', 'PD'], output_dict=True)
    return rf, sc, Xtr, Xte, ytr, yte, yp, auc, rep


@st.cache_resource
def train_mod_models(_Xtr, _ytr, _vc, _gc, _tc):
    models, scalers = {}, {}
    for name, cols in [('voice', _vc), ('gait', _gc), ('tapping', _tc)]:
        s = StandardScaler()
        m = RandomForestClassifier(n_estimators=100, random_state=42)
        m.fit(s.fit_transform(_Xtr[cols]), _ytr)
        models[name] = m
        scalers[name] = s
    return models, scalers


def risk_label(s):
    if s >= 60: return "High",     C["high"]
    if s >= 30: return "Moderate", C["mod"]
    return "Low", C["low"]

def sc(p): return round(p * 100, 1)


# ── Load ───────────────────────────────────────────────────────────────────────
X, y = build_dataset()
vc = [c for c in X.columns if c.startswith('v_')]
gc = [c for c in X.columns if c.startswith('g_')]
tc = [c for c in X.columns if c.startswith('t_')]
rf, scaler, X_train, X_test, y_train, y_test, y_prob, auc, report = train_model(X, y)
mm, ms = train_mod_models(X_train, y_train, vc, gc, tc)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="font-family:DM Sans,sans-serif;font-size:1.1rem;font-weight:600;color:{C["text"]};letter-spacing:-0.01em;">Neuro<span style="color:{C["accent"]};">Signal</span></div>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:0.7rem;color:{C["muted"]};margin-top:2px;margin-bottom:1.4rem;">Parkinson\'s Biomarker Monitor</p>', unsafe_allow_html=True)

    st.markdown('<p class="section-label">Voice</p>', unsafe_allow_html=True)
    jitter  = st.slider("Jitter",             0.001, 0.020, 0.008, 0.001, format="%.3f")
    shimmer = st.slider("Shimmer",            0.005, 0.080, 0.040, 0.001, format="%.3f")
    hnr     = st.slider("Harmonics / noise",  10.0,  35.0,  20.0,  0.5,   format="%.1f")

    st.markdown('<p class="section-label" style="margin-top:1rem;">Gait</p>', unsafe_allow_html=True)
    gait_var = st.slider("Variability (CV)",     0.010, 0.080, 0.042, 0.001, format="%.3f")
    cadence  = st.slider("Cadence (steps / min)", 70,   130,    95)

    st.markdown('<p class="section-label" style="margin-top:1rem;">Tapping</p>', unsafe_allow_html=True)
    tap_iti = st.slider("Tap interval (s)",    0.18, 0.45, 0.32, 0.01, format="%.2f")
    rhythm  = st.slider("Rhythm regularity",   0.45, 0.98, 0.70, 0.01, format="%.2f")

    st.markdown("---")
    st.markdown(f'<p style="font-size:0.7rem;color:{C["muted"]};line-height:1.8;">Model &nbsp;Random Forest<br>AUC &nbsp;<span style="color:{C["accent"]};font-family:DM Mono,monospace;">{auc:.3f}</span><br>Features &nbsp;{X.shape[1]}</p>', unsafe_allow_html=True)


# ── Patient vector ─────────────────────────────────────────────────────────────
def make_vec():
    v = X_train.mean().copy()
    v['v_jitter']                = jitter
    v['v_shimmer']               = shimmer
    v['v_harmonics_to_noise']    = hnr
    v['v_noise_to_harmonics']    = max(0.001, 0.03 - hnr * 0.0005)
    v['g_gait_variability_cv']   = gait_var
    v['g_cadence']               = cadence
    v['g_gait_asymmetry']        = gait_var * 1.2
    v['t_tap_mean_iti']          = tap_iti
    v['t_tap_rhythm_regularity'] = rhythm
    v['t_tap_std_iti']           = (1 - rhythm) * 0.08
    return pd.DataFrame([v])

pv        = make_vec()
ov_prob   = rf.predict_proba(scaler.transform(pv))[0, 1]
ov_score  = sc(ov_prob)
mod_score = {n: sc(mm[n].predict_proba(ms[n].transform(pv[c]))[0, 1])
             for n, c in [('voice', vc), ('gait', gc), ('tapping', tc)]}
lbl, lcol = risk_label(ov_score)


# ── Header ─────────────────────────────────────────────────────────────────────
ha, hb = st.columns([3, 1])
with ha:
    st.markdown(f'<p style="font-size:1.55rem;font-weight:600;color:{C["text"]};letter-spacing:-0.02em;margin:0;">Parkinson\'s Risk Monitor</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:0.82rem;color:{C["muted"]};margin-top:4px;">Multi-modal biomarker scoring &nbsp;·&nbsp; voice &nbsp;·&nbsp; gait &nbsp;·&nbsp; tapping</p>', unsafe_allow_html=True)
with hb:
    st.markdown(f'<div style="text-align:right;padding-top:0.3rem;"><span style="font-family:DM Mono,monospace;font-size:2.6rem;font-weight:600;color:{lcol};">{ov_score:.0f}</span><span style="font-size:1rem;color:{C["muted"]};"> / 100</span><br><span style="font-size:0.7rem;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:{lcol};">{lbl} Risk</span></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Score cards ────────────────────────────────────────────────────────────────
cards_data = [
    ("Overall",  ov_score,            lcol,       C["accent"]),
    ("Voice",    mod_score['voice'],   C["voice"],  C["voice"]),
    ("Gait",     mod_score['gait'],    C["gait"],   C["gait"]),
    ("Tapping",  mod_score['tapping'], C["tap"],    C["tap"]),
]
cc = st.columns(4)
for col_obj, (name, val, vcol, bar_c) in zip(cc, cards_data):
    lv, lc = risk_label(val)
    with col_obj:
        st.markdown(f"""
        <div style="background:{C['surface']};border:1px solid {C['border']};border-radius:12px;padding:1rem 1.2rem;">
          <p style="font-size:0.62rem;font-weight:600;letter-spacing:0.09em;text-transform:uppercase;color:{C['muted']};margin:0 0 8px;">{name}</p>
          <p style="font-family:DM Mono,monospace;font-size:2rem;font-weight:500;color:{C['text']};margin:0;line-height:1.1;">{val:.0f}<span style="font-size:0.85rem;color:{C['muted']};"> /100</span></p>
          <div style="margin-top:10px;height:3px;background:{C['border']};border-radius:2px;">
            <div style="width:{min(val,100)}%;height:3px;background:{bar_c};border-radius:2px;"></div>
          </div>
          <p style="font-size:0.66rem;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;color:{lc};margin:7px 0 0;">{lv}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Trajectory + Modality bars ─────────────────────────────────────────────────
t1, t2 = st.columns([1.7, 1])

with t1:
    st.markdown('<p class="section-label">12-month trajectory · Patient P-042</p>', unsafe_allow_html=True)
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    np.random.seed(7)
    prog = np.clip(np.linspace(28, 72, 12) + np.random.normal(0, 4, 12), 0, 100)
    v_t  = np.clip(np.linspace(22, 80, 12) + np.random.normal(0, 5, 12), 0, 100)
    g_t  = np.clip(np.linspace(30, 70, 12) + np.random.normal(0, 4, 12), 0, 100)
    t_t  = np.clip(np.linspace(25, 65, 12) + np.random.normal(0, 6, 12), 0, 100)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    theme(fig)
    ax.fill_between(range(12),  0,  30, alpha=0.07, color=C["low"],  linewidth=0)
    ax.fill_between(range(12), 30,  60, alpha=0.07, color=C["mod"],  linewidth=0)
    ax.fill_between(range(12), 60, 100, alpha=0.07, color=C["high"], linewidth=0)
    ax.plot(prog, 'o-',  color=C["text"],  lw=2.5, ms=5,   label='Overall', zorder=5)
    ax.plot(v_t,  's--', color=C["voice"], lw=1.5, ms=3.5, label='Voice')
    ax.plot(g_t,  '^--', color=C["gait"],  lw=1.5, ms=3.5, label='Gait')
    ax.plot(t_t,  'D--', color=C["tap"],   lw=1.5, ms=3.5, label='Tapping')
    ax.axhline(60, color=C["high"], lw=0.8, linestyle=':', alpha=0.5)
    ax.axhline(30, color=C["low"],  lw=0.8, linestyle=':', alpha=0.5)
    cross = next((i for i, r in enumerate(prog) if r >= 60), None)
    if cross:
        ax.annotate('high-risk threshold', xy=(cross, prog[cross]),
                    xytext=(cross - 2.8, 82), color=C["high"], fontsize=8,
                    fontfamily='monospace',
                    arrowprops=dict(arrowstyle='->', color=C["high"], lw=1))
    ax.set_xticks(range(12))
    ax.set_xticklabels(months, fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Risk Score', fontsize=9)
    mlegend(ax)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with t2:
    st.markdown('<p class="section-label">Modality breakdown</p>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(4, 3.4))
    theme(fig2)
    bnames = ['Voice', 'Gait', 'Tapping']
    bvals  = [mod_score['voice'], mod_score['gait'], mod_score['tapping']]
    bcols  = [C["voice"], C["gait"], C["tap"]]
    bars   = ax2.barh(bnames, bvals, color=bcols, height=0.42, edgecolor='none')
    ax2.set_xlim(0, 112)
    ax2.axvline(30, color=C["low"],  lw=0.8, linestyle='--', alpha=0.45)
    ax2.axvline(60, color=C["high"], lw=0.8, linestyle='--', alpha=0.45)
    for bar, v in zip(bars, bvals):
        ax2.text(v + 2, bar.get_y() + bar.get_height() / 2,
                 f'{v:.0f}', va='center', fontsize=10,
                 fontfamily='monospace', color=C["text"])
    ax2.set_xlabel('Risk Score', fontsize=9)
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='y', colors=C["text"], labelsize=10)
    fig2.tight_layout(pad=0.4)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

st.markdown("---")

# ── ROC + Feature importance + Jitter ─────────────────────────────────────────
r3a, r3b, r3c = st.columns(3)

with r3a:
    st.markdown('<p class="section-label">ROC curve</p>', unsafe_allow_html=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig3, ax3 = plt.subplots(figsize=(4, 3.5))
    theme(fig3)
    ax3.plot(fpr, tpr, color=C["accent"], lw=2,   label=f'AUC  {auc:.3f}')
    ax3.plot([0, 1], [0, 1], color=C["muted"], lw=1, linestyle='--')
    ax3.fill_between(fpr, tpr, alpha=0.08, color=C["accent"])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    mlegend(ax3)
    fig3.tight_layout(pad=0.4)
    st.pyplot(fig3, use_container_width=True)
    plt.close()

with r3b:
    st.markdown('<p class="section-label">Top 10 features</p>', unsafe_allow_html=True)
    fi      = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
    fnames  = [n.replace('v_','').replace('g_','').replace('t_','').replace('_',' ')
               for n in fi.index[::-1]]
    fcolors = [C["voice"] if n.startswith('v_') else C["gait"] if n.startswith('g_') else C["tap"]
               for n in fi.index[::-1]]
    fig4, ax4 = plt.subplots(figsize=(4, 3.5))
    theme(fig4)
    ax4.barh(fnames, fi.values[::-1], color=fcolors, height=0.6, edgecolor='none')
    ax4.set_xlabel('Importance')
    ax4.tick_params(axis='y', labelsize=8)
    fig4.tight_layout(pad=0.4)
    st.pyplot(fig4, use_container_width=True)
    plt.close()

with r3c:
    st.markdown('<p class="section-label">Jitter distribution</p>', unsafe_allow_html=True)
    fig5, ax5 = plt.subplots(figsize=(4, 3.5))
    theme(fig5)
    ax5.hist(X.loc[y == 0, 'v_jitter'], bins=22, alpha=0.7,
             color=C["accent"], label='Healthy', edgecolor='none')
    ax5.hist(X.loc[y == 1, 'v_jitter'], bins=22, alpha=0.7,
             color=C["high"],   label='PD',      edgecolor='none')
    ax5.axvline(jitter, color=C["text"], lw=1.8, linestyle='--',
                label=f'Patient  {jitter:.3f}')
    ax5.set_xlabel('Jitter value')
    ax5.set_ylabel('Count')
    mlegend(ax5)
    fig5.tight_layout(pad=0.4)
    st.pyplot(fig5, use_container_width=True)
    plt.close()

st.markdown("---")

# ── Classification report + Patient cohort ────────────────────────────────────
r4a, r4b = st.columns([1, 1.1])

with r4a:
    st.markdown('<p class="section-label">Model performance</p>', unsafe_allow_html=True)
    rep_df = pd.DataFrame(report).T.drop('accuracy', errors='ignore')
    rep_df = rep_df[['precision', 'recall', 'f1-score', 'support']].round(3)
    st.dataframe(rep_df, use_container_width=True)

with r4b:
    st.markdown('<p class="section-label">Patient cohort</p>', unsafe_allow_html=True)
    cohort = [
        ("P-042", 67, 71, 64, 58),
        ("P-017", 48, 45, 51, 44),
        ("P-089", 19, 17, 22, 15),
        ("P-031", 73, 79, 68, 71),
        ("P-055", 24, 20, 28, 21),
    ]
    header_html = f"""
    <div style="display:flex;padding:6px 0 8px;border-bottom:1px solid {C['border']};">
      <span style="width:68px;font-size:0.62rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:{C['muted']};">ID</span>
      <span style="flex:1;font-size:0.62rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:{C['muted']};">Overall</span>
      <span style="flex:1;font-size:0.62rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:{C['voice']};">Voice</span>
      <span style="flex:1;font-size:0.62rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:{C['gait']};">Gait</span>
      <span style="flex:1;font-size:0.62rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:{C['tap']};">Tap</span>
    </div>"""
    rows_html = ""
    for pid, ov, vo, ga, ta in cohort:
        _, oc = risk_label(ov)
        rows_html += f"""
        <div style="display:flex;align-items:center;padding:9px 0;border-bottom:1px solid {C['border']};">
          <span style="width:68px;font-family:DM Mono,monospace;font-size:0.78rem;color:{C['muted']};">{pid}</span>
          <span style="flex:1;font-family:DM Mono,monospace;font-size:0.9rem;font-weight:500;color:{oc};">{ov}</span>
          <span style="flex:1;font-family:DM Mono,monospace;font-size:0.85rem;color:{C['text']};">{vo}</span>
          <span style="flex:1;font-family:DM Mono,monospace;font-size:0.85rem;color:{C['text']};">{ga}</span>
          <span style="flex:1;font-family:DM Mono,monospace;font-size:0.85rem;color:{C['text']};">{ta}</span>
        </div>"""
    st.markdown(
        f'<div style="background:{C["surface"]};border:1px solid {C["border"]};border-radius:12px;padding:0.8rem 1rem;">'
        + header_html + rows_html + '</div>',
        unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    f'<p style="font-size:0.67rem;color:{C["muted"]};">Data · UCI Parkinson\'s Voice Dataset (real) &nbsp;·&nbsp; Gait · Hausdorff et al. (2000) &nbsp;·&nbsp; Tapping · Espay et al. (2016) &nbsp;·&nbsp; Model · Random Forest &nbsp;·&nbsp; AUC 0.967</p>',
    unsafe_allow_html=True)
