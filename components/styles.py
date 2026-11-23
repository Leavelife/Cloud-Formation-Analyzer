import streamlit as st

def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
        --ink:#18375A; --blue:#39A9E8; --blue-dark:#287FAB;
        --muted:#68788A; --paper:#F5F7FA; --line:#DDE7F0;
        --shadow:0 12px 34px rgba(25,59,99,.08);
        --shadow-soft:0 5px 18px rgba(25,59,99,.055);
    }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; scroll-behavior:smooth; }
    .stApp {
        background:
            radial-gradient(circle at 88% 4%, rgba(57,169,232,.09), transparent 22rem),
            linear-gradient(180deg, #F9FBFD 0%, var(--paper) 100%);
        color:#172B42;
    }
    [data-testid="stHeader"] { background:transparent; height:0; }
    .block-container { max-width:1380px; padding:1rem 2rem 4rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink); letter-spacing:-.035em; }
    h1 { font-size:clamp(38px,4.2vw,58px) !important; line-height:1.05 !important; font-weight:800 !important; margin-bottom:.55rem !important; }
    h2 { font-size:clamp(25px,2.2vw,34px) !important; }
    h3 { font-size:20px !important; }
    p, li { color:#536579; line-height:1.65; }
    .topbar {
        display:flex; align-items:center; justify-content:space-between;
        padding:14px 18px; background:rgba(255,255,255,.92);
        border:1px solid rgba(221,231,240,.95); border-radius:16px;
        margin:0 0 16px; box-shadow:var(--shadow-soft);
        backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
        animation:fadeDown .48s ease-out both;
    }
    .brand { font-family:'Manrope'; font-weight:800; color:var(--ink); font-size:18px; letter-spacing:-.04em; }
    .brand span { color:var(--blue); }
    .eyebrow { text-transform:uppercase; letter-spacing:.16em; color:#328FC6; font-size:11px; font-weight:800; margin-bottom:10px; }
    .hero { padding:44px 0 30px; animation:fadeUp .58s ease-out both; }
    .hero h1 { font-size:clamp(44px,5.6vw,72px) !important; line-height:.98 !important; max-width:850px; margin:0 0 20px; }
    .hero p { font-size:17px; line-height:1.75; max-width:680px; }
    .panel {
        background:rgba(255,255,255,.94); border:1px solid var(--line); border-radius:20px;
        padding:24px; box-shadow:var(--shadow-soft);
        transition:transform .28s cubic-bezier(.2,.8,.2,1), box-shadow .28s ease, border-color .28s ease;
    }
    .panel:hover { transform:translateY(-3px); box-shadow:var(--shadow); border-color:#CEDDE9; }
    .panel-title { font-family:'Manrope'; color:var(--ink); font-size:17px; font-weight:800; margin-bottom:7px; }
    .small-muted { color:var(--muted); font-size:13px; }
    .metric {
        background:rgba(255,255,255,.96); border:1px solid var(--line); border-radius:16px;
        padding:18px 20px; box-shadow:var(--shadow-soft);
        transition:transform .25s ease, box-shadow .25s ease;
    }
    .metric:hover { transform:translateY(-2px); box-shadow:var(--shadow); }
    .metric-label { color:var(--muted); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
    .metric-value { color:var(--ink); font-family:'Manrope'; font-size:30px; font-weight:800; margin-top:6px; }
    div.stButton > button {
        border-radius:11px; border:1px solid #D5E1EC; font-weight:700;
        min-height:44px; background:#fff; color:var(--ink);
        transition:all .22s cubic-bezier(.2,.8,.2,1);
    }
    div.stButton > button:hover {
        border-color:#A9C8DC; transform:translateY(-1px);
        box-shadow:0 7px 18px rgba(24,55,90,.09);
    }
    div.stButton > button:active { transform:translateY(0) scale(.985); }
    div.stButton > button[kind="primary"] { background:var(--ink); border-color:var(--ink); color:#fff; }
    div.stButton > button[kind="primary"]:hover { background:#214b78; border-color:#214b78; }
    [data-testid="stFileUploader"] {
        background:rgba(255,255,255,.96); border:1.5px dashed #A9C3D7;
        border-radius:18px; padding:20px;
        transition:border-color .22s ease, background .22s ease, transform .22s ease;
    }
    [data-testid="stFileUploader"]:hover { border-color:var(--blue); background:#F9FCFF; transform:translateY(-1px); }
    [data-testid="stTabs"] button {
        font-weight:700; border-radius:9px 9px 0 0;
        transition:color .18s ease, background .18s ease;
    }
    [data-testid="stTabs"] [aria-selected="true"] { color:var(--ink); border-bottom-color:var(--blue); }
    [data-testid="stDataFrame"], [data-testid="stImage"] {
        border-radius:16px; overflow:hidden;
        box-shadow:var(--shadow-soft);
        animation:fadeUp .42s ease-out both;
    }
    [data-testid="stMetric"] {
        background:rgba(255,255,255,.96); border:1px solid var(--line);
        border-radius:16px; padding:17px 18px; box-shadow:var(--shadow-soft);
        transition:transform .22s ease, box-shadow .22s ease;
    }
    [data-testid="stMetric"]:hover { transform:translateY(-2px); box-shadow:var(--shadow); }
    [data-testid="stAlert"], [data-baseweb="notification"] {
        border-radius:14px; border:1px solid var(--line);
        animation:fadeUp .34s ease-out both;
    }
    [data-testid="stRadio"] [role="radiogroup"], [data-testid="stSelectbox"] > div { transition:all .18s ease; }
    [data-testid="stExpander"] {
        border:1px solid var(--line); border-radius:14px; background:#fff;
        overflow:hidden; box-shadow:var(--shadow-soft);
    }
    [data-testid="stExpander"] summary { transition:background .18s ease; }
    [data-testid="stExpander"] summary:hover { background:#F7FAFD; }
    hr { border:0; height:1px; background:linear-gradient(90deg, transparent, var(--line), transparent); margin:2rem 0; }
    .workflow-card { position:relative; overflow:hidden; }
    .workflow-card::after {
        content:""; position:absolute; width:70px; height:70px; right:-35px; bottom:-35px;
        border-radius:50%; background:rgba(57,169,232,.07);
        transition:transform .35s ease;
    }
    .workflow-card:hover::after { transform:scale(1.7); }
    footer { visibility:hidden; }
    @keyframes fadeUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
    @keyframes fadeDown { from { opacity:0; transform:translateY(-8px); } to { opacity:1; transform:translateY(0); } }
    @media (max-width: 900px) {
        .block-container { padding:0.5rem 1rem 3rem; }
        .topbar { align-items:flex-start; gap:8px; }
        .topbar > div:last-child { display:none; }
        h1 { font-size:36px !important; }
        .hero { padding:30px 0 22px; }
        .hero h1 { font-size:43px !important; }
        .panel { padding:20px; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation:none !important; transition:none !important; scroll-behavior:auto !important; }
    }
    footer {visibility:hidden;}
    </style>
    """, unsafe_allow_html=True)
