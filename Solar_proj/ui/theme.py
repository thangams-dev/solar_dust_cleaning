def dashboard_css(dark: bool = False) -> str:
    background = "#0B1220" if dark else "#F4F7FB"
    card = "#111B2E" if dark else "#FFFFFF"
    text = "#E5EAF2" if dark else "#111827"
    secondary = "#94A3B8" if dark else "#64748B"
    border = "#263449" if dark else "#E2E8F0"
    metric_bg = "#111B2E" if dark else "#FFFFFF"
    return f"""
    <style>
    :root {{
      --app-bg: {background}; --card-bg: {card}; --text-main: {text};
      --text-secondary: {secondary}; --border-color: {border};
      --sidebar-bg: #111827; --brand-blue: #2563EB;
    }}
    html, body, [data-testid="stAppViewContainer"] {{
      background: var(--app-bg); color: var(--text-main);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu, footer {{ visibility:hidden; }}
    .stAppDeployButton, [data-testid="stToolbarActions"] {{ display:none; }}
    [data-testid="stMain"] {{ background: var(--app-bg); }}
    .block-container {{ padding: 1.4rem 2rem 2.5rem; max-width: 1540px; }}
    [data-testid="stSidebar"] {{ background: var(--sidebar-bg); border-right: 1px solid #263244; }}
    [data-testid="stSidebar"] > div:first-child {{ background: var(--sidebar-bg); }}
    [data-testid="stSidebar"] * {{ color: #E5E7EB; }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color: #9CA3AF; }}
    [data-testid="stSidebar"] [role="radiogroup"] {{ gap: .3rem; }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{
      border-radius: 8px; padding: .55rem .65rem; margin: 0;
      color: #CBD5E1; transition: background .15s ease;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: #1F2937; }}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
      background: #1D4ED8; color: #FFFFFF;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label p {{ color: inherit !important; }}
    [data-testid="stVerticalBlockBorderWrapper"] > div {{
      background: var(--card-bg); border-color: var(--border-color) !important;
      border-radius: 12px !important; box-shadow: 0 1px 3px rgba(15, 23, 42, .04);
    }}
    [data-testid="stMetric"] {{
      background: var(--card-bg); border: 1px solid var(--border-color);
      border-radius: 12px; padding: 1rem 1.1rem; box-shadow: 0 1px 3px rgba(15,23,42,.04);
    }}
    [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {{ color: var(--text-secondary); }}
    [data-testid="stMetricValue"] {{ color: var(--text-main); font-size: 1.65rem; font-weight: 650; }}
    h1 {{ font-size: 1.55rem !important; letter-spacing: -.025em; color: var(--text-main); }}
    h2 {{ font-size: 1.25rem !important; color: var(--text-main); }}
    h3 {{ font-size: 1rem !important; color: var(--text-main); }}
    p, label, [data-testid="stCaptionContainer"] {{ color: var(--text-secondary); }}
    .brand-mark {{
      display:flex; align-items:center; gap:.65rem; margin:.2rem 0 1.6rem;
    }}
    .brand-icon {{
      width:38px; height:38px; border-radius:10px; background:#F59E0B;
      display:flex; align-items:center; justify-content:center; color:#111827;
      font-size:21px; font-weight:700;
    }}
    .brand-name {{ color:#F9FAFB; font-size:1.04rem; font-weight:700; letter-spacing:.02em; }}
    .brand-subtitle {{ color:#9CA3AF; font-size:.72rem; margin-top:2px; }}
    .top-header {{
      display:flex; justify-content:space-between; align-items:flex-start;
      gap:1rem; border-bottom:1px solid var(--border-color);
      padding:0 0 1rem; margin:0 0 1.15rem;
    }}
    .top-header-copy h1 {{ margin:0 0 .22rem; }}
    .top-header-copy p {{ margin:0; color:var(--text-secondary); font-size:.9rem; }}
    .top-header h1 {{ margin:0 0 .22rem; }}
    .top-header p {{ margin:0; color:var(--text-secondary); font-size:.9rem; }}
    .header-meta {{ display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; justify-content:flex-end; }}
    .status-badge {{
      display:inline-flex; align-items:center; gap:.4rem; border-radius:999px;
      padding:.3rem .62rem; font-size:.74rem; font-weight:650; white-space:nowrap;
      border:1px solid transparent;
    }}
    .status-dot {{ width:7px; height:7px; border-radius:50%; background:currentColor; }}
    .status-good {{ color:#15803D; background:#F0FDF4; border-color:#BBF7D0; }}
    .status-warning {{ color:#B45309; background:#FFFBEB; border-color:#FDE68A; }}
    .status-critical {{ color:#B91C1C; background:#FEF2F2; border-color:#FECACA; }}
    .status-info {{ color:#1D4ED8; background:#EFF6FF; border-color:#BFDBFE; }}
    .status-neutral {{ color:#64748B; background:#F1F5F9; border-color:#E2E8F0; }}
    .metric-icon {{
      width:38px; height:38px; display:flex; align-items:center; justify-content:center;
      border-radius:10px; background:#EFF6FF; color:#2563EB; font-size:1.15rem;
    }}
    .metric-label {{ color:var(--text-secondary); font-size:.8rem; font-weight:550; }}
    .metric-value {{ color:var(--text-main); font-size:1.55rem; font-weight:700; line-height:1.3; }}
    .metric-unit {{ color:var(--text-secondary); font-size:.86rem; font-weight:500; }}
    .section-heading {{ display:flex; align-items:baseline; gap:.7rem; margin:.15rem 0 .8rem; }}
    .section-heading h3 {{ margin:0; }}
    .section-heading span {{ color:var(--text-secondary); font-size:.78rem; }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{ border:1px solid var(--border-color); border-radius:10px; }}
    div[data-testid="stAlert"] {{ border-radius:10px; }}
    div.stButton > button {{ border-radius:8px; font-weight:600; min-height:2.55rem; }}
    div[data-testid="stDownloadButton"] button {{ border-radius:8px; font-weight:600; }}
    @media (max-width: 900px) {{
      .block-container {{ padding:1rem 1rem 2rem; }}
      .top-header {{ flex-direction:column; }}
      .header-meta {{ justify-content:flex-start; }}
    }}
    </style>
    """
