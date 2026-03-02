"""
Veridi Logistics — Delivery Performance Audit Dashboard
Mirrors the logic in AMALITECH_organized.ipynb exactly.

Run:  streamlit run dashboard.py
Needs the Olist CSV files in the same folder (or upload via sidebar).
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Veridi Logistics · Delivery Audit",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# THEME — dark industrial, Syne + DM Mono
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
  --bg:      #0c0e13; --surface: #13171f; --surface2: #1a2030;
  --border:  #222b3a; --accent:  #e84040; --accent2: #f5a623;
  --accent3: #3a9de8; --text:    #e6e9f0; --muted:   #64748b;
  --good:    #22c55e; --warn:    #f59e0b; --bad:     #ef4444;
}
html, body, [class*="css"] {
  font-family: 'DM Mono', monospace !important;
  background: var(--bg) !important; color: var(--text) !important;
}
section[data-testid="stSidebar"]          { background: var(--surface) !important; border-right:1px solid var(--border); }
section[data-testid="stSidebar"] *        { color: var(--text) !important; }
#MainMenu, footer, header                  { visibility:hidden; }
div[data-testid="stFileUploader"] label   { font-size:11px !important; }

/* KPI cards */
.kpi-row   { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:24px; }
.kpi       { background:var(--surface); border:1px solid var(--border); border-radius:10px;
             padding:20px 22px; position:relative; overflow:hidden; }
.kpi::before { content:''; position:absolute; top:0;left:0;right:0; height:3px; }
.kpi.red::before    { background:var(--bad); }
.kpi.amber::before  { background:var(--warn); }
.kpi.green::before  { background:var(--good); }
.kpi.blue::before   { background:var(--accent3); }
.kpi .lbl  { font-size:9px; letter-spacing:.13em; text-transform:uppercase; color:var(--muted); margin-bottom:6px; }
.kpi .val  { font-family:'Syne',sans-serif; font-size:34px; font-weight:800; line-height:1; margin-bottom:3px; }
.kpi .val.red   { color:var(--bad); }
.kpi .val.amber { color:var(--warn); }
.kpi .val.green { color:var(--good); }
.kpi .val.blue  { color:var(--accent3); }
.kpi .sub  { font-size:10px; color:var(--muted); }

/* Section headers */
.sh { font-family:'Syne',sans-serif; font-size:11px; font-weight:700; letter-spacing:.15em;
      text-transform:uppercase; color:var(--muted); border-left:3px solid var(--accent);
      padding-left:10px; margin:26px 0 14px; }

/* Page title */
.ptitle { font-family:'Syne',sans-serif; font-size:26px; font-weight:800;
          letter-spacing:-.02em; color:var(--text); margin-bottom:2px; }
.psub   { font-size:10px; color:var(--muted); letter-spacing:.08em;
          text-transform:uppercase; margin-bottom:24px; }

/* Insight box */
.ib { background:var(--surface2); border:1px solid var(--border);
      border-left:4px solid var(--accent2); border-radius:8px;
      padding:13px 16px; font-size:11px; line-height:1.75; margin-top:14px; }
.ib strong { color:var(--accent2); }

/* Story tag */
.stag { display:inline-block; background:var(--surface2); border:1px solid var(--border);
        border-radius:4px; padding:2px 8px; font-size:9px; letter-spacing:.1em;
        text-transform:uppercase; color:var(--accent2); margin-bottom:10px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# PLOTLY DEFAULTS
# ──────────────────────────────────────────────────────────────
BG, GRID, TC, FM = "rgba(0,0,0,0)", "#1e2a3a", "#e6e9f0", "DM Mono, monospace"
PAL = {"On Time": "#22c55e", "Late": "#f59e0b", "Super Late": "#ef4444"}

def theme(fig, h=340, xt=None, yt=None):
    fig.update_layout(
        height=h, paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FM, color=TC, size=11),
        margin=dict(l=8, r=8, t=36, b=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
        xaxis=dict(gridcolor=GRID, zeroline=False, tickfont_size=10, title=xt),
        yaxis=dict(gridcolor=GRID, zeroline=False, tickfont_size=10, title=yt),
    )
    return fig

# ──────────────────────────────────────────────────────────────
# DATA PIPELINE  (mirrors notebook exactly — pd.read_csv directly)
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Building master dataset…")
def build_master():
    # — Load CSVs exactly like the notebook (cell 03) —
    orders    = pd.read_csv("olist_orders_dataset.csv")
    reviews   = pd.read_csv("olist_order_reviews_dataset.csv")
    customers = pd.read_csv("olist_customers_dataset.csv")

    # — datetime conversion (notebook cell 09) —
    for c in ["order_purchase_timestamp","order_approved_at",
              "order_delivered_carrier_date","order_delivered_customer_date",
              "order_estimated_delivery_date"]:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")
    for c in ["review_creation_date","review_answer_timestamp"]:
        reviews[c] = pd.to_datetime(reviews[c], errors="coerce")

    # — Story 1: deduplicate reviews & join (cells 11-12) —
    reviews_dedup = (
        reviews.sort_values(["order_id","review_answer_timestamp","review_creation_date"])
               .drop_duplicates("order_id", keep="last")[["order_id","review_score"]]
    )
    master = orders.merge(customers, on="customer_id", how="left", validate="many_to_one")
    master = master.merge(reviews_dedup, on="order_id", how="left", validate="one_to_one")

    # — Story 2: flags + delay calculator (cells 16-20) —
    master["is_delivered"] = (
        (master["order_status"] == "delivered") &
        master["order_delivered_customer_date"].notna() &
        master["order_estimated_delivery_date"].notna()
    )
    master["never_delivered"] = master["order_status"].isin(["canceled","unavailable"])

    def classify_delivery(s):
        if s == "delivered":               return "Delivered"
        elif s in ["canceled","unavailable"]: return "Failed"
        else:                              return "In Progress"
    master["delivery_category"] = master["order_status"].apply(classify_delivery)

    master["Days_Difference"] = (
        master["order_estimated_delivery_date"] - master["order_delivered_customer_date"]
    ).dt.days

    delivered = master[master["is_delivered"]].copy()
    delivered["delay_days"] = (
        delivered["order_delivered_customer_date"] - delivered["order_estimated_delivery_date"]
    ).dt.days
    delivered["delay_status"] = np.select(
        [delivered["delay_days"] <= 0,
         (delivered["delay_days"] > 0) & (delivered["delay_days"] <= 5),
         delivered["delay_days"] > 5],
        ["On Time","Late","Super Late"], default="Unknown"
    )
    delivered["purchase_month"] = (
        delivered["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    )

    # — Story 5: category translation (cells 33-34) —
    try:
        products   = pd.read_csv("olist_products_dataset.csv")
        items      = pd.read_csv("olist_order_items_dataset.csv")
        items = items.sort_values(["order_id", "order_item_id"])
        items_d    = items.drop_duplicates("order_id", keep="first")[["order_id","product_id"]]

        try:
            trans = pd.read_csv("product_category_name_translation.csv")
            trans.columns = ["product_category_name","product_category_name_english"]
        except FileNotFoundError:
            manual_map = {
                "cama_mesa_banho":"bed_bath_table","beleza_saude":"health_beauty",
                "esporte_lazer":"sports_leisure","informatica_acessorios":"computers_accessories",
                "moveis_decoracao":"furniture_decor","eletrodomesticos":"home_appliances",
                "brinquedos":"toys","relogios_presentes":"watches_gifts",
                "ferramentas_jardim":"garden_tools","automotivo":"auto",
                "eletronicos":"electronics","bebes":"baby",
                "utilidades_domesticas":"housewares","papelaria":"office_supplies",
                "telefonia":"telephony","livros_tecnicos":"books_technical",
                "construcao_ferramentas_construcao":"construction_tools",
                "agro_industria_e_comercio":"agro_industry_commerce",
                "musica":"music","cool_stuff":"cool_stuff","perfumaria":"perfumery",
                "instrumentos_musicais":"musical_instruments","casa_conforto":"home_comfort",
                "fashion_bolsas_e_acessorios":"fashion_bags_accessories",
                "pets":"pet_shop","fraldas_higiene":"diapers_hygiene","flores":"flowers",
            }
            trans = pd.DataFrame(manual_map.items(),
                                 columns=["product_category_name","product_category_name_english"])

        prod_t    = products.merge(trans, on="product_category_name", how="left")
        order_cat = items_d.merge(
            prod_t[["product_id","product_category_name_english"]], on="product_id", how="left"
        )
        delivered = delivered.merge(order_cat, on="order_id", how="left")
    except FileNotFoundError:
        pass  # category page will show a warning if these files are missing

    return master, delivered

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:14px 0 6px'>
      <span style='font-family:Syne,sans-serif;font-size:19px;font-weight:800'>📦 VERIDI</span><br>
      <span style='font-size:9px;color:#64748b;letter-spacing:.12em;text-transform:uppercase'>Logistics Intelligence</span>
    </div>
    <hr style='border-color:#222b3a;margin:10px 0'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:9px;color:#64748b;letter-spacing:.12em;text-transform:uppercase;margin-bottom:6px'>Navigate</div>", unsafe_allow_html=True)
    page = st.radio("", [
        "📊  Executive Overview",
        "🗺️  Story 3 — Geographic",
        "💬  Story 4 — Sentiment",
        "📦  Story 5 & 6 — Categories",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#222b3a;margin:12px 0'>", unsafe_allow_html=True)

    try:
        master, delivered = build_master()
        st.markdown("<div style='font-size:10px;color:#22c55e;margin-bottom:4px'>✅ CSV files loaded</div>", unsafe_allow_html=True)
    except FileNotFoundError as e:
        st.error(f"❌ Missing file: {e.filename}  \n\nPlace all Olist CSV files in the same folder as dashboard.py, then rerun.")
        st.stop()

    st.markdown("<hr style='border-color:#222b3a;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:9px;color:#64748b;letter-spacing:.12em;text-transform:uppercase;margin-bottom:6px'>Filters</div>", unsafe_allow_html=True)

    min_d = delivered["order_purchase_timestamp"].min().date()
    max_d = delivered["order_purchase_timestamp"].max().date()
    date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    all_states = sorted(delivered["customer_state"].dropna().unique())
    sel_states = st.multiselect("States", all_states, placeholder="All states")

    st.markdown("<hr style='border-color:#222b3a;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:9px;color:#64748b'>{len(delivered):,} delivered orders loaded</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# APPLY FILTERS
# ──────────────────────────────────────────────────────────────
df = delivered.copy()
if len(date_range) == 2:
    df = df[(df["order_purchase_timestamp"] >= pd.Timestamp(date_range[0])) &
            (df["order_purchase_timestamp"] <= pd.Timestamp(date_range[1]))]
if sel_states:
    df = df[df["customer_state"].isin(sel_states)]

total      = len(df)
n_ontime   = (df["delay_status"] == "On Time").sum()
n_late     = (df["delay_status"] == "Late").sum()
n_super    = (df["delay_status"] == "Super Late").sum()
pct_late   = (n_late + n_super) / total * 100 if total else 0
avg_review = df["review_score"].mean()
avg_delay  = df.loc[df["delay_days"] > 0, "delay_days"].mean()

# ══════════════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE OVERVIEW  (Stories 1 + 2)
# ══════════════════════════════════════════════════════════════
if page.startswith("📊"):

    st.markdown("<div class='ptitle'>Delivery Performance Audit</div>", unsafe_allow_html=True)
    st.markdown("<div class='psub'>Veridi Logistics · Executive Overview · Stories 1 & 2</div>", unsafe_allow_html=True)

    # ── KPI row ──
    st.markdown(f"""
    <div class='kpi-row'>
      <div class='kpi red'>
        <div class='lbl'>Combined Late Rate</div>
        <div class='val red'>{pct_late:.1f}%</div>
        <div class='sub'>{n_late+n_super:,} of {total:,} orders</div>
      </div>
      <div class='kpi amber'>
        <div class='lbl'>Super Late &gt;5 days</div>
        <div class='val amber'>{n_super:,}</div>
        <div class='sub'>{n_super/total*100:.1f}% of deliveries</div>
      </div>
      <div class='kpi green'>
        <div class='lbl'>On-Time Deliveries</div>
        <div class='val green'>{n_ontime:,}</div>
        <div class='sub'>{n_ontime/total*100:.1f}% success rate</div>
      </div>
      <div class='kpi blue'>
        <div class='lbl'>Avg Review Score</div>
        <div class='val blue'>{avg_review:.2f}</div>
        <div class='sub'>out of 5.00</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Story 1: master dataset summary ──
    st.markdown("<div class='stag'>Story 1 — Schema Builder</div>", unsafe_allow_html=True)
    st.markdown("<div class='sh'>Master Dataset — Join Validation</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders",    f"{len(master):,}")
    c2.metric("Unique Order IDs", f"{master['order_id'].nunique():,}")
    c3.metric("Duplicate Rows",  f"{len(master) - master['order_id'].nunique():,}")
    c4.metric("Delivered Orders", f"{master['is_delivered'].sum():,}")

    with st.expander("📋 Order Status Breakdown (Story 1 validation)"):
        status_counts = master["delivery_category"].value_counts().reset_index()
        status_counts.columns = ["Category","Count"]
        status_counts["Pct"] = (status_counts["Count"] / len(master) * 100).round(2)
        st.dataframe(status_counts, use_container_width=True, hide_index=True)

    st.markdown("<hr style='border-color:#222b3a;margin:10px 0'>", unsafe_allow_html=True)

    # ── Story 2: delay calculator ──
    st.markdown("<div class='stag'>Story 2 — Delay Calculator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sh'>Delivery Status Distribution</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # donut (mirrors notebook cell 21 bar chart)
    with col1:
        counts = df["delay_status"].value_counts().reindex(["On Time","Late","Super Late"]).fillna(0)
        fig = go.Figure(go.Pie(
            labels=counts.index, values=counts.values, hole=0.6,
            marker_colors=[PAL["On Time"], PAL["Late"], PAL["Super Late"]],
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
            textfont_size=11,
        ))
        fig.add_annotation(text=f"<b>{total:,}</b><br>orders",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(family="Syne,sans-serif", size=16, color=TC))
        theme(fig, h=300)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.08),
                          title="Delay Status Breakdown")
        st.plotly_chart(fig, use_container_width=True)

    # delay summary table (notebook cell 20)
    with col2:
        delay_summary = (
            df["delay_status"].value_counts()
              .reindex(["On Time","Late","Super Late"]).fillna(0)
              .rename_axis("Status").to_frame("Orders")
        )
        delay_summary["Percentage"] = (delay_summary["Orders"] / delay_summary["Orders"].sum() * 100).round(2)
        delay_summary = delay_summary.reset_index()

        st.markdown("<div class='sh'>Summary Table (notebook cell 20)</div>", unsafe_allow_html=True)
        st.dataframe(delay_summary, use_container_width=True, hide_index=True)

        # monthly trend
        st.markdown("<div class='sh'>Monthly Late-Rate Trend</div>", unsafe_allow_html=True)
        monthly_total = df.groupby("purchase_month").size().reset_index(name="total")
        monthly_late  = (df[df["delay_status"].isin(["Late","Super Late"])]
                           .groupby("purchase_month").size().reset_index(name="late"))
        trend = monthly_total.merge(monthly_late, on="purchase_month", how="left").fillna(0)
        trend["pct"] = trend["late"] / trend["total"] * 100

        fig2 = go.Figure(go.Scatter(
            x=trend["purchase_month"], y=trend["pct"],
            mode="lines+markers",
            line=dict(color="#ef4444", width=2),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
            hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
        ))
        theme(fig2, h=200, xt=None, yt="Late Rate (%)")
        fig2.update_layout(title="Monthly Late Rate %", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # delay distribution histogram (notebook cell 21 reimagined)
    st.markdown("<div class='sh'>Distribution of Delay Days (Late Orders Only)</div>", unsafe_allow_html=True)
    late_days = df[df["delay_days"] > 0]["delay_days"].clip(upper=60)
    fig3 = px.histogram(late_days, nbins=50, color_discrete_sequence=["#ef4444"],
                        labels={"value":"Days Late"})
    fig3.update_traces(marker_line_width=0, opacity=0.85)
    theme(fig3, h=240, xt="Days Late (capped at 60)", yt="Number of Orders")
    fig3.update_layout(showlegend=False, title="Delay Distribution — Late Orders", bargap=0.04)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f"""
    <div class='ib'>
      <strong>Story 2 Insight:</strong>
      {pct_late:.1f}% of delivered orders arrived late.
      <strong>{n_super:,}</strong> orders were Super Late (>5 days), representing
      <strong>{n_super/total*100:.1f}%</strong> of all deliveries.
      The average delay for late orders is <strong>{avg_delay:.1f} days</strong>.
      8 orders were excluded — marked 'delivered' but missing delivery timestamps (notebook cell 18).
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — GEOGRAPHIC HEATMAP  (Story 3)
# ══════════════════════════════════════════════════════════════
elif page.startswith("🗺️"):

    st.markdown("<div class='ptitle'>Geographic Heatmap</div>", unsafe_allow_html=True)
    st.markdown("<div class='psub'>Veridi Logistics · Story 3 — Which states are failing customers?</div>", unsafe_allow_html=True)
    st.markdown("<div class='stag'>Story 3 — Geographic Heatmap</div>", unsafe_allow_html=True)

    # — notebook cell 23: state_pivot —
    state_pivot = (
        df.groupby(["customer_state","delay_status"])
          .size().unstack(fill_value=0).reset_index()
    )
    for col in ["On Time","Late","Super Late"]:
        if col not in state_pivot.columns: state_pivot[col] = 0
    state_pivot["orders"]         = state_pivot[["On Time","Late","Super Late"]].sum(axis=1)
    state_pivot["pct_late"]       = state_pivot["Late"]       / state_pivot["orders"]
    state_pivot["pct_super_late"] = state_pivot["Super Late"] / state_pivot["orders"]
    state_pivot["pct_combined"]   = state_pivot["pct_late"] + state_pivot["pct_super_late"]

    avg_rev_s = df.groupby("customer_state")["review_score"].mean().reset_index()
    avg_rev_s.columns = ["customer_state","avg_review"]
    state_stats = state_pivot.merge(avg_rev_s, on="customer_state")

    # — notebook cell 25: remote flag —
    remote_states = ["AC","RO","RR","AP","AM","TO","MT","MS","PA"]
    state_stats["is_remote"] = state_stats["customer_state"].isin(remote_states)

    col1, col2 = st.columns([3,2])

    # notebook cell 24: top states bar
    with col1:
        st.markdown("<div class='sh'>Late Delivery Rate by State (Top 20)</div>", unsafe_allow_html=True)
        top20 = state_stats.sort_values("pct_combined", ascending=True).tail(20)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=top20["customer_state"], x=top20["pct_late"]*100,
                             orientation="h", name="Late (1–5 days)", marker_color=PAL["Late"]))
        fig.add_trace(go.Bar(y=top20["customer_state"], x=top20["pct_super_late"]*100,
                             orientation="h", name="Super Late (>5 days)", marker_color=PAL["Super Late"]))
        theme(fig, h=520, xt="Late Rate (%)")
        fig.update_layout(barmode="stack", yaxis_title=None,
                          legend=dict(orientation="h", y=1.04),
                          title="Stacked Late Rate by State")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # notebook cell 26 & 27: remote vs non-remote
        st.markdown("<div class='sh'>Remote vs Non-Remote (cell 26–27)</div>", unsafe_allow_html=True)
        rem = state_stats.groupby("is_remote")[["pct_late","pct_super_late"]].mean() * 100
        rem = rem.reset_index()
        rem["label"] = rem["is_remote"].map({True:"Remote", False:"Non-Remote"})

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=rem["label"], y=rem["pct_late"],
                              name="Late %", marker_color=PAL["Late"]))
        fig2.add_trace(go.Bar(x=rem["label"], y=rem["pct_super_late"],
                              name="Super Late %", marker_color=PAL["Super Late"]))
        theme(fig2, h=240, yt="Rate (%)")
        fig2.update_layout(barmode="group", xaxis_title=None,
                           legend=dict(orientation="h", y=1.04),
                           title="Remote vs Non-Remote Avg Late Rate")
        st.plotly_chart(fig2, use_container_width=True)

        # avg review vs late rate scatter
        st.markdown("<div class='sh'>Late Rate vs Avg Review Score</div>", unsafe_allow_html=True)
        fig3 = px.scatter(state_stats, x="pct_combined", y="avg_review",
                          text="customer_state", size="orders",
                          color="is_remote",
                          color_discrete_map={True:"#f59e0b", False:"#3a9de8"},
                          labels={"pct_combined":"Combined Late Rate (frac)","avg_review":"Avg Review","is_remote":"Remote"},
                          size_max=40)
        fig3.update_traces(textfont_size=8, textposition="top center")
        theme(fig3, h=250)
        fig3.update_layout(title="State: Late Rate vs Review Score")
        st.plotly_chart(fig3, use_container_width=True)

    # full table
    st.markdown("<div class='sh'>Full State Performance Table (notebook cell 23)</div>", unsafe_allow_html=True)
    tbl = state_stats[[
        "customer_state","orders","pct_late","pct_super_late","pct_combined","avg_review","is_remote"
    ]].copy()
    tbl.columns = ["State","Orders","Late %","Super Late %","Combined %","Avg Review","Remote"]
    tbl["Late %"]      = (tbl["Late %"]      * 100).round(2)
    tbl["Super Late %"]= (tbl["Super Late %"]* 100).round(2)
    tbl["Combined %"]  = (tbl["Combined %"]  * 100).round(2)
    tbl["Avg Review"]  = tbl["Avg Review"].round(2)
    tbl = tbl.sort_values("Combined %", ascending=False).reset_index(drop=True)
    st.dataframe(
        tbl.style
           .background_gradient(subset=["Combined %"], cmap="Reds")
           .background_gradient(subset=["Avg Review"],  cmap="Greens")
           .format({"Late %":"{:.2f}","Super Late %":"{:.2f}","Combined %":"{:.2f}","Avg Review":"{:.2f}"}),
        use_container_width=True, hide_index=True
    )

    worst = tbl.iloc[0]
    best  = tbl[tbl["Orders"] > 200].sort_values("Combined %").iloc[0]
    st.markdown(f"""
    <div class='ib'>
      <strong>Story 3 Insight:</strong>
      Worst state: <strong>{worst['State']}</strong> — combined late rate
      <strong>{worst['Combined %']:.1f}%</strong>, avg review <strong>{worst['Avg Review']:.2f}/5</strong>.
      Best performer (min 200 orders): <strong>{best['State']}</strong> at {best['Combined %']:.1f}%.
      Non-remote states average a <em>higher</em> late rate than remote ones —
      confirming this is a <strong>nationwide problem</strong>, not a last-mile rural issue.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — SENTIMENT CORRELATION  (Story 4)
# ══════════════════════════════════════════════════════════════
elif page.startswith("💬"):

    st.markdown("<div class='ptitle'>Sentiment Correlation</div>", unsafe_allow_html=True)
    st.markdown("<div class='psub'>Veridi Logistics · Story 4 — Do late deliveries cause bad reviews?</div>", unsafe_allow_html=True)
    st.markdown("<div class='stag'>Story 4 — Sentiment Correlation</div>", unsafe_allow_html=True)

    # notebook cell 29
    avg_scores = (
        df.groupby("delay_status")["review_score"].mean()
          .reindex(["On Time","Late","Super Late"])
    )
    a_on, a_late, a_sup = avg_scores["On Time"], avg_scores["Late"], avg_scores["Super Late"]
    p1_late   = (df.loc[df["delay_days"] > 5, "review_score"] == 1).mean() * 100
    p1_ontime = (df.loc[df["delay_days"] <= 0,"review_score"] == 1).mean() * 100

    st.markdown(f"""
    <div class='kpi-row'>
      <div class='kpi green'>
        <div class='lbl'>On-Time Avg Review</div>
        <div class='val green'>{a_on:.2f}</div>
        <div class='sub'>baseline</div>
      </div>
      <div class='kpi amber'>
        <div class='lbl'>Late Avg Review</div>
        <div class='val amber'>{a_late:.2f}</div>
        <div class='sub'>−{a_on-a_late:.2f} vs on-time</div>
      </div>
      <div class='kpi red'>
        <div class='lbl'>Super Late Avg Review</div>
        <div class='val red'>{a_sup:.2f}</div>
        <div class='sub'>−{a_on-a_sup:.2f} vs on-time</div>
      </div>
      <div class='kpi red'>
        <div class='lbl'>1-Star Rate (Super Late)</div>
        <div class='val red'>{p1_late:.0f}%</div>
        <div class='sub'>vs {p1_ontime:.0f}% on-time</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # notebook cell 30: avg review by status bar chart
    with col1:
        st.markdown("<div class='sh'>Avg Review Score by Delay Status (cell 30)</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=["On Time","Late","Super Late"],
            y=[a_on, a_late, a_sup],
            marker_color=[PAL["On Time"], PAL["Late"], PAL["Super Late"]],
            text=[f"{v:.2f}" for v in [a_on, a_late, a_sup]],
            textposition="outside", textfont=dict(size=13, family="Syne,sans-serif"),
        ))
        theme(fig, h=320, xt=None, yt="Avg Review Score (1–5)")
        fig.update_layout(yaxis_range=[0,5.6], showlegend=False,
                          title="Do Late Deliveries Drive Bad Reviews?")
        st.plotly_chart(fig, use_container_width=True)

    # review score distribution
    with col2:
        st.markdown("<div class='sh'>Score Distribution by Status</div>", unsafe_allow_html=True)
        score_dist = (
            df[df["delay_status"].isin(["On Time","Late","Super Late"])]
              .groupby(["delay_status","review_score"]).size().reset_index(name="count")
        )
        score_dist["pct"] = score_dist.groupby("delay_status")["count"] \
                                       .transform(lambda x: x / x.sum() * 100)
        fig2 = px.bar(score_dist, x="review_score", y="pct", color="delay_status",
                      barmode="group", color_discrete_map=PAL,
                      labels={"review_score":"Review Score","pct":"% of Status Orders","delay_status":"Status"},
                      category_orders={"delay_status":["On Time","Late","Super Late"]})
        theme(fig2, h=320, xt="Review Score (1–5)", yt="% of Orders in Status")
        fig2.update_layout(xaxis=dict(tickmode="linear", dtick=1),
                           title="Review Score Distribution by Delay Status")
        st.plotly_chart(fig2, use_container_width=True)

    # notebook cell 31: scatter delay_days vs review_score
    st.markdown("<div class='sh'>Delivery Delay vs Review Score (cell 31 — sampled 5,000 orders)</div>", unsafe_allow_html=True)
    sample = df.dropna(subset=["delay_days","review_score"])
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)

    fig3 = px.scatter(sample, x="delay_days", y="review_score",
                      color="delay_status", color_discrete_map=PAL, opacity=0.25,
                      labels={"delay_days":"Delay Days (+ = late)","review_score":"Review Score","delay_status":"Status"},
                      category_orders={"delay_status":["On Time","Late","Super Late"]},
                      hover_data={"customer_state":True})
    # moving average trend line
    binned = (df.assign(b=df["delay_days"].clip(-30,60).round())
                .groupby("b")["review_score"].mean().reset_index())
    fig3.add_trace(go.Scatter(x=binned["b"], y=binned["review_score"],
                              mode="lines", line=dict(color="#fff", width=2, dash="dot"),
                              name="Avg trend"))
    fig3.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1.5,
                   annotation_text="Estimated date", annotation_font_color="#64748b")
    theme(fig3, h=340, xt="Days Relative to Estimated Date", yt="Review Score")
    fig3.update_layout(xaxis_range=[-35,65],
                       title="Delivery Delay vs Customer Review Score")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f"""
    <div class='ib'>
      <strong>Story 4 Insight:</strong>
      On-time orders average <strong>{a_on:.2f}/5</strong>.
      Super Late orders drop to <strong>{a_sup:.2f}/5</strong> — a
      <strong>{a_on-a_sup:.2f}-point</strong> fall.
      <strong>{p1_late:.0f}%</strong> of Super Late orders receive 1-star reviews, vs
      only <strong>{p1_ontime:.0f}%</strong> for on-time orders.
      The scatter and trend line confirm a clear negative correlation.
      <strong>Logistics performance is the primary driver of customer dissatisfaction.</strong>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — CATEGORY DRILL-DOWN  (Stories 5 & 6)
# ══════════════════════════════════════════════════════════════
elif page.startswith("📦"):

    st.markdown("<div class='ptitle'>Category Drill-Down</div>", unsafe_allow_html=True)
    st.markdown("<div class='psub'>Veridi Logistics · Stories 5 & 6 — Translation Challenge + Candidate's Choice</div>", unsafe_allow_html=True)

    if "product_category_name_english" not in df.columns:
        st.warning("⚠️ Category data not available. Upload `olist_products_dataset.csv` and `olist_order_items_dataset.csv` via the sidebar to enable this page.")
        st.stop()

    st.markdown("<div class='stag'>Story 5 — Translation Challenge</div>", unsafe_allow_html=True)

    # notebook cell 35: preview translated categories
    with st.expander("📋 Translation preview (notebook cell 35)"):
        prev = df[["order_id","product_category_name_english"]].dropna().head(10)
        st.dataframe(prev, use_container_width=True, hide_index=True)

    st.markdown("<hr style='border-color:#222b3a;margin:10px 0'>", unsafe_allow_html=True)
    st.markdown("<div class='stag'>Story 6 — Candidate's Choice: Bubble Chart</div>", unsafe_allow_html=True)

    # notebook cells 37-38
    cat_df = delivered.dropna(subset=["product_category_name_english"]).copy()
    cat_df["is_late"] = cat_df["delay_days"] > 0

    cat_stats = cat_df.groupby("product_category_name_english").agg(
        orders     = ("order_id",     "count"),
        pct_late   = ("is_late",      "mean"),
        avg_review = ("review_score", "mean"),
        avg_delay  = ("delay_days",   "mean"),
    ).reset_index()
    cat_stats = cat_stats[cat_stats["orders"] > 200]
    cat_stats["pct_late_pct"] = (cat_stats["pct_late"] * 100).round(2)
    cat_stats["avg_review"]   = cat_stats["avg_review"].round(2)
    cat_stats["avg_delay"]    = cat_stats["avg_delay"].round(1)

    # bubble chart (notebook cell 37)
    st.markdown("<div class='sh'>Bubble Chart: Late Rate × Review Score × Volume (cell 37)</div>", unsafe_allow_html=True)
    fig = px.scatter(
        cat_stats, x="pct_late_pct", y="avg_review", size="orders",
        color="pct_late_pct", color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
        hover_name="product_category_name_english",
        hover_data={"orders":True,"pct_late_pct":":.1f","avg_review":":.2f","avg_delay":":.1f"},
        labels={"pct_late_pct":"Late Rate (%)","avg_review":"Avg Review","orders":"Order Volume"},
        size_max=55,
    )
    worst5 = cat_stats.nlargest(5,"pct_late_pct")
    for _, row in worst5.iterrows():
        fig.add_annotation(x=row["pct_late_pct"], y=row["avg_review"],
                           text=row["product_category_name_english"].replace("_"," "),
                           showarrow=False, yshift=16, font=dict(size=8, color=TC))
    theme(fig, h=460, xt="Late Rate (%)", yt="Avg Review Score")
    fig.update_coloraxes(colorbar=dict(title="Late %", tickfont_size=9))
    fig.update_layout(title="Delay Impact by Product Category  (bubble = order volume)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='sh'>Worst 10 Categories (cell 38)</div>", unsafe_allow_html=True)
        w10 = cat_stats.nlargest(10,"pct_late_pct").sort_values("pct_late_pct")
        fig2 = px.bar(w10, x="pct_late_pct", y="product_category_name_english",
                      orientation="h", color="pct_late_pct",
                      color_continuous_scale=["#f59e0b","#ef4444"],
                      text="pct_late_pct",
                      labels={"pct_late_pct":"Late %","product_category_name_english":"Category"})
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont_size=10)
        theme(fig2, h=340)
        fig2.update_layout(coloraxis_showscale=False, showlegend=False,
                           yaxis_title=None, title="Top 10 Worst Categories")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("<div class='sh'>Best 10 Categories</div>", unsafe_allow_html=True)
        b10 = cat_stats.nsmallest(10,"pct_late_pct").sort_values("pct_late_pct", ascending=False)
        fig3 = px.bar(b10, x="pct_late_pct", y="product_category_name_english",
                      orientation="h", color="pct_late_pct",
                      color_continuous_scale=["#22c55e","#3a9de8"],
                      text="pct_late_pct",
                      labels={"pct_late_pct":"Late %","product_category_name_english":"Category"})
        fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont_size=10)
        theme(fig3, h=340)
        fig3.update_layout(coloraxis_showscale=False, showlegend=False,
                           yaxis_title=None, title="Top 10 Best Categories")
        st.plotly_chart(fig3, use_container_width=True)

    # full sortable table (notebook cell 38)
    st.markdown("<div class='sh'>Full Category Table (notebook cell 38)</div>", unsafe_allow_html=True)
    full = cat_stats[[
        "product_category_name_english","orders","pct_late_pct","avg_review","avg_delay"
    ]].sort_values("pct_late_pct", ascending=False).reset_index(drop=True)
    full.columns = ["Category (English)","Orders","Late Rate (%)","Avg Review","Avg Delay (days)"]
    st.dataframe(
        full.style
            .background_gradient(subset=["Late Rate (%)"], cmap="Reds")
            .background_gradient(subset=["Avg Review"],    cmap="Greens")
            .format({"Late Rate (%)":"{:.2f}","Avg Review":"{:.2f}","Avg Delay (days)":"{:.1f}"}),
        use_container_width=True, hide_index=True
    )

    worst_c = cat_stats.nlargest(1,"pct_late_pct").iloc[0]
    best_c  = cat_stats.nsmallest(1,"pct_late_pct").iloc[0]
    st.markdown(f"""
    <div class='ib'>
      <strong>Stories 5 & 6 Insight:</strong>
      Portuguese categories have been translated to English (Story 5).
      <strong>{worst_c['product_category_name_english'].replace('_',' ').title()}</strong>
      is the worst category: <strong>{worst_c['pct_late_pct']:.1f}%</strong> late rate,
      avg review <strong>{worst_c['avg_review']:.2f}/5</strong>.
      <strong>{best_c['product_category_name_english'].replace('_',' ').title()}</strong>
      performs best at <strong>{best_c['pct_late_pct']:.1f}%</strong> late (Story 6 bubble chart).
      Operations should target high-volume, high-delay categories first for SLA renegotiation.
    </div>
    """, unsafe_allow_html=True)
