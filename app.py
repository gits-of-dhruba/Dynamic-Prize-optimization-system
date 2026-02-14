"""
Dynamic Pricing Optimization - Coralogix-style browser
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from dynamic_pricing_prototype import (
    load_data,
    add_price_column,
    estimate_price_elasticity,
    optimal_price,
    revenue_curve,
    seasonal_trends,
)

st.set_page_config(
    page_title="Commerce Cloud - Dynamic Pricing",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----- Theme state (sidebar removed; theme in header) -----
if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"
if "section" not in st.session_state:
    st.session_state["section"] = "Collect Data"

# ----- Shared CSS: dark header + mega menu + light/dark body -----
HEADER_AND_MEGA_CSS = """
<style>
    /* Dark top nav bar - always dark like Coralogix */
    .nav-bar {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        padding: 0.6rem 1.5rem;
        margin: -1rem -1rem 0 -1rem;
        margin-bottom: 0;
        border-bottom: 1px solid #30363d;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .nav-bar .logo { font-size: 1.25rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 0.5rem; }
    .nav-bar .logo-dot { width: 10px; height: 10px; border-radius: 50%; background: linear-gradient(135deg, #34d399 0%, #10b981 100%); box-shadow: 0 0 12px rgba(52, 211, 153, 0.5); }
    .nav-bar .nav-links { display: flex; gap: 0.5rem; align-items: center; }
    .nav-bar .nav-links a, .nav-bar .nav-links label { color: #c9d1d9 !important; font-size: 0.95rem; }
    .nav-bar .theme-wrap { display: flex; align-items: center; gap: 0.5rem; color: #c9d1d9; }
    .nav-bar > div { display: flex !important; width: 100% !important; align-items: center !important; gap: 1rem !important; }
    .nav-bar button { background: transparent !important; color: #c9d1d9 !important; border: 1px solid #30363d !important; border-radius: 6px !important; }
    .nav-bar button:hover { background: rgba(52, 211, 153, 0.15) !important; color: #6ee7b7 !important; border-color: #34d399 !important; }
    /* Mega menu panel - light, attractive */
    .mega-menu {
        background: linear-gradient(145deg, #f0fdfa 0%, #ecfeff 50%, #f5f3ff 100%);
        border: 1px solid #99f6e4;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.08), 0 2px 6px rgba(0,0,0,0.04);
    }
    .mega-menu.dark {
        background: linear-gradient(145deg, #1e293b 0%, #1a1d24 100%);
        border-color: #334155;
        box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    }
    .mega-menu h4 { margin-bottom: 0.75rem; font-size: 1rem; color: #0d9488; }
    .mega-menu.dark h4 { color: #5eead4; }
    .mega-menu ul { margin: 0; padding-left: 1.25rem; color: #0f766e; font-size: 0.9rem; line-height: 1.8; }
    .mega-menu.dark ul { color: #a5f3fc; }
    .mega-menu .col-block { cursor: pointer; padding: 0.5rem; border-radius: 8px; transition: background 0.15s; }
    .mega-menu .col-block:hover { background: rgba(45, 212, 191, 0.12); }
    .mega-menu.dark .col-block:hover { background: rgba(94, 234, 212, 0.15); }
    /* FAQ section */
    .faq-title { font-size: 1.5rem; font-weight: 600; color: #0d9488; margin-bottom: 0.5rem; }
    .faq-title.dark { color: #5eead4; }
    .faq-underline { height: 3px; background: linear-gradient(90deg, #2dd4bf 0%, #22d3ee 100%); margin-bottom: 1.25rem; border-radius: 2px; }
    .faq-underline.dark { background: linear-gradient(90deg, #5eead4 0%, #67e8f9 100%); }
    /* Hide Streamlit branding for cleaner nav */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
"""

# Light/Dark body theme
DARK_CSS = """
<style>
    [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #0f172a 0%, #0e1117 100%); }
    [data-testid="stHeader"] { background-color: #0e1117; }
    .stApp { background: linear-gradient(180deg, #0f172a 0%, #0e1117 100%); }
    p, span, label, .stMarkdown { color: #e2e8f0 !important; }
    h1, h2, h3 { color: #f1f5f9 !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; }
    [data-testid="stMetricDelta"] { color: #94a3b8 !important; }
    div[data-testid="column"] { background-color: transparent; }
    [data-testid="stVerticalBlockBorderWrapper"] { background: linear-gradient(145deg, #1e293b 0%, #172033 100%) !important; border: 1px solid #334155 !important; border-radius: 10px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
    [data-testid="stExpander"] { border: 1px solid #334155 !important; border-radius: 8px !important; background: rgba(30, 41, 59, 0.5) !important; }
    hr { border-color: #334155 !important; }
</style>
"""
LIGHT_CSS = """
<style>
    [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #f0fdfa 0%, #ecfeff 30%, #faf5ff 100%); }
    [data-testid="stHeader"] { background: transparent; }
    .stApp { background: linear-gradient(180deg, #f0fdfa 0%, #ecfeff 30%, #faf5ff 100%); }
    p, span, label, .stMarkdown { color: #334155 !important; }
    h1, h2, h3 { color: #0f766e !important; }
    [data-testid="stMetricLabel"] { color: #64748b !important; }
    [data-testid="stMetricValue"] { color: #0d9488 !important; font-weight: 600; }
    [data-testid="stMetricDelta"] { color: #64748b !important; }
    div[data-testid="column"] { background-color: transparent; }
    [data-testid="stVerticalBlockBorderWrapper"] { background: linear-gradient(145deg, #f0fdfa 0%, #e0f2fe 50%, #f5f3ff 100%) !important; border: 1px solid #99f6e4 !important; border-radius: 12px !important; box-shadow: 0 2px 12px rgba(6, 182, 212, 0.06); }
    [data-testid="stExpander"] { border: 1px solid #a5f3fc !important; border-radius: 10px !important; background: rgba(236, 254, 255, 0.6) !important; }
    hr { border-color: #99f6e4 !important; }
    .stSlider label { color: #0f766e !important; }
    .stCaption { color: #64748b !important; }
    /* Content area buttons - light teal style (nav bar buttons keep nav style) */
    .stButton > button { background: linear-gradient(135deg, #99f6e4 0%, #a5f3fc 100%) !important; color: #0f766e !important; border: 1px solid #5eead4 !important; border-radius: 8px !important; font-weight: 500 !important; }
    .stButton > button:hover { background: linear-gradient(135deg, #5eead4 0%, #67e8f9 100%) !important; color: #134e4a !important; border-color: #2dd4bf !important; box-shadow: 0 2px 8px rgba(45, 212, 191, 0.3); }
    .nav-bar .stButton > button { background: transparent !important; color: #c9d1d9 !important; border: 1px solid #30363d !important; }
    .nav-bar .stButton > button:hover { background: rgba(52, 211, 153, 0.15) !important; color: #6ee7b7 !important; border-color: #34d399 !important; box-shadow: none; }
</style>
"""

# ----- Dark header bar -----
st.markdown(HEADER_AND_MEGA_CSS, unsafe_allow_html=True)
st.markdown('<div class="nav-bar">', unsafe_allow_html=True)

col_logo, col_nav, col_theme = st.columns([1, 2, 1])
with col_logo:
    st.markdown('<div class="logo"><span class="logo-dot"></span> Commerce Cloud</div>', unsafe_allow_html=True)
with col_nav:
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Collect Data", key="nav1"):
            st.session_state["section"] = "Collect Data"
            st.rerun()
    with c2:
        if st.button("Analyze Elasticity", key="nav2"):
            st.session_state["section"] = "Analyze Elasticity"
            st.rerun()
    with c3:
        if st.button("Apply Model", key="nav3"):
            st.session_state["section"] = "Apply Model"
            st.rerun()
with col_theme:
    theme = st.radio("Theme", ["Light", "Dark"], horizontal=True, key="theme_radio", label_visibility="collapsed")
    if theme != st.session_state.get("theme"):
        st.session_state["theme"] = theme
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ----- Mega menu: How the system works (3 columns) -----
is_dark = (theme == "Dark")
mega_class = "mega-menu dark" if is_dark else "mega-menu"
st.markdown(
    '<div style="background: linear-gradient(90deg, #0d9488 0%, #0891b2 50%, #6366f1 100%); padding: 0.55rem 1rem; border-radius: 10px 10px 0 0; margin-bottom: 0; box-shadow: 0 2px 8px rgba(13, 148, 136, 0.2);"><span style="color: #fff; font-weight: 600;">How the system works</span> <span style="color: #cffafe;">&gt;</span></div>',
    unsafe_allow_html=True,
)
st.markdown(f'<div class="{mega_class}">', unsafe_allow_html=True)
st.markdown("Select a section from the nav bar or use the links below.")
mm1, mm2, mm3 = st.columns(3)
with mm1:
    st.markdown("#### Collect Data")
    st.markdown("""
    - Sales history  
    - Demand trends  
    - Competitor pricing  
    - Inventory levels  
    - Market factors  
    """)
    if st.button("View Collect Data →", key="m1"):
        st.session_state["section"] = "Collect Data"
        st.rerun()
with mm2:
    st.markdown("#### Analyze Demand Elasticity")
    st.markdown("""
    - How much demand changes when price changes  
    - Elasticity coefficient  
    - Model fit (R²)  
    """)
    if st.button("View Analyze Elasticity →", key="m2"):
        st.session_state["section"] = "Analyze Elasticity"
        st.rerun()
with mm3:
    st.markdown("#### Apply Mathematical Model")
    st.markdown("""
    - Predict sales at different prices  
    - Revenue & profit curves  
    - Optimal price  
    """)
    if st.button("View Apply Model →", key="m3"):
        st.session_state["section"] = "Apply Model"
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# Apply body theme
st.session_state["theme"] = theme
if theme == "Dark":
    st.markdown(DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

# ----- Data (cached) -----
@st.cache_data
def get_data():
    df = load_data()
    return add_price_column(df)

df = get_data()
base_sales = df['Weekly_Sales'].median()
base_price = df['Price'].median()
cost = 2.5
result = estimate_price_elasticity(df)
curve = revenue_curve(base_sales, base_price, result['elasticity'], cost)
section = st.session_state.get("section", "Collect Data")

# ----- Section content -----
if section == "Collect Data":
    st.header("1. Collect Data")
    st.markdown("The system gathers data from multiple sources to understand the market.")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Sales history")
            st.metric("Weekly records", f"{len(df):,}")
            st.metric("Stores", df['Store'].nunique())
    with col2:
        with st.container(border=True):
            st.markdown("#### Demand trends")
            seasonal = seasonal_trends(df)
            peak_month = seasonal['seasonal_factor'].idxmax()
            st.metric("Peak month", f"Month {int(peak_month)}", f"{seasonal.loc[peak_month, 'seasonal_factor']:.2f}x average")
    st.markdown("**Other inputs:** Competitor pricing (Fuel_Price, CPI), Inventory levels, Market factors (Temperature, Holiday, Unemployment).")
    with st.expander("View sample data"):
        st.dataframe(df[['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Price']].head(20), use_container_width=True)

elif section == "Analyze Elasticity":
    st.header("2. Analyze Demand Elasticity")
    st.markdown("**How much does demand change when price changes?** We use a log-log model; the elasticity coefficient gives the % change in quantity for a 1% change in price.")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Price elasticity", f"{result['elasticity']:.2f}", "Elastic" if result['elasticity'] < -1 else "Inelastic")
        with c2:
            st.metric("Interpretation", result['interpretation'].capitalize(), "")
        with c3:
            st.metric("Model fit (R²)", f"{result['r2']:.2f}", "")
    st.markdown("**Example:** If elasticity = -1.2, a 10% price increase leads to about a 12% drop in quantity sold.")
    fig_el = plt.figure(figsize=(8, 3), facecolor="#0e1117" if is_dark else "white")
    ax = fig_el.add_subplot(111, facecolor="#0e1117" if is_dark else "white")
    ax.plot(curve['Price'], curve['Quantity'] / 1e5, color='teal', linewidth=2)
    ax.set_xlabel("Price ($)", color="white" if is_dark else "black")
    ax.set_ylabel("Quantity (100k units)", color="white" if is_dark else "black")
    ax.set_title("Demand curve: quantity vs price", color="white" if is_dark else "black")
    ax.tick_params(colors="white" if is_dark else "black")
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color("white" if is_dark else "black")
    st.pyplot(fig_el, facecolor="#0e1117" if is_dark else "white")
    plt.close()

else:  # Apply Model
    st.header("3. Apply Mathematical Model")
    st.markdown("**Predict sales and revenue at different prices.** Quantity = Base Q × (Price / Base price)^(elasticity). Revenue = Price × Quantity.")
    price_choice = st.slider("Choose a price ($)", min_value=1.0, max_value=15.0, value=float(base_price), step=0.5)
    q_pred = (base_sales / base_price) * (price_choice / base_price) ** result['elasticity']
    rev_pred = price_choice * q_pred
    st.metric("Predicted weekly revenue at this price", f"${rev_pred:,.0f}", f"Quantity ~ {q_pred:,.0f} units")
    st.markdown("#### Revenue and profit at different prices")
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0e1117" if is_dark else "white")
    ax.set_facecolor("#0e1117" if is_dark else "white")
    ax.plot(curve['Price'], curve['Revenue'] / 1e6, label="Revenue (M$)", linewidth=2)
    ax.plot(curve['Price'], curve['Profit'] / 1e6, label="Profit (M$)", linewidth=2)
    ax.axvline(base_price, color="gray", linestyle="--", label=f"Current ${base_price:.1f}")
    opt_p = optimal_price(result['elasticity'], cost)
    if not pd.isna(opt_p):
        ax.axvline(opt_p, color="green", linestyle="--", label=f"Optimal ${opt_p:.1f}")
    ax.axvline(price_choice, color="orange", linestyle="-", linewidth=2, label=f"Your choice ${price_choice:.1f}")
    ax.set_xlabel("Price ($)", color="white" if is_dark else "black")
    ax.set_ylabel("Million $", color="white" if is_dark else "black")
    ax.tick_params(colors="white" if is_dark else "black")
    ax.legend(facecolor="#1a1d24" if is_dark else "white", labelcolor="white" if is_dark else "black")
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color("white" if is_dark else "black")
    st.pyplot(fig, facecolor="#0e1117" if is_dark else "white")
    plt.close()

st.divider()

# ----- FAQ -----
faq_title_class = "faq-title dark" if is_dark else "faq-title"
faq_ul_class = "faq-underline dark" if is_dark else "faq-underline"
st.markdown('<p class="' + faq_title_class + '">Frequently Asked Questions</p>', unsafe_allow_html=True)
st.markdown('<div class="' + faq_ul_class + '"></div>', unsafe_allow_html=True)

with st.expander("How does dynamic pricing work differently than static pricing?"):
    st.markdown("""
    **Static pricing** keeps the same price for long periods (e.g. a fixed tag in a store).  
    **Dynamic pricing** adjusts prices over time using data (sales history, demand, competition, seasonality).  
    This system estimates how demand responds to price (elasticity), then recommends or applies prices that improve revenue or profit. Prices can change by store, week, or product depending on the model.
    """)

with st.expander("Why use a dynamic fee or price instead of a flat fee?"):
    st.markdown("""
    A **flat fee** is simple but ignores that demand and costs vary (by season, location, or product).  
    A **dynamic** approach lets you:
    - **Capture more value** when demand is high (e.g. holidays) without leaving money on the table.
    - **Stay competitive** when demand is low by adjusting instead of always charging the same.
    - **Balance revenue and volume** using elasticity: sometimes a lower price brings more total revenue.
    In this prototype, the “optimal price” is the one that maximizes profit given the estimated demand curve.
    """)

with st.expander("Why don’t you have coverage in my city or store?"):
    st.markdown("""
    This prototype uses the **Walmart weekly sales** dataset, which includes 45 stores over a fixed date range.  
    “Coverage” (which stores or regions are included) is limited to that dataset.  
    In a full system, you would connect your own data (by city, store, or region) so the model runs for your locations and time period.
    """)

with st.expander("What is price elasticity and how is it used?"):
    st.markdown("""
    **Price elasticity of demand** is the percentage change in quantity sold when price changes by 1%.  
    For example, elasticity of -1.2 means a 10% price increase leads to about a 12% drop in quantity.  
    We estimate it with a **log-log regression** (quantity and price in logs). The model then uses this elasticity to predict sales at different prices and to find the revenue- or profit-maximizing price.
    """)

with st.expander("Where does the “price” come from in this prototype?"):
    st.markdown("""
    The original dataset has **weekly sales** (revenue) but no separate price column.  
    We add a **synthetic price** using CPI and Fuel_Price as a market price proxy (normalized into a realistic range).  
    So elasticity and “optimal price” are illustrative. With real price and quantity data, you’d get more accurate estimates.
    """)

st.divider()
st.caption("Commerce Cloud — Dynamic Pricing Optimization | Data: Walmart weekly sales")
