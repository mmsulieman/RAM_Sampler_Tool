# app.py
# Streamlit app: Single-file workflow for PPS (village) + Household systematic sampling (Stratified & Unstratified)
# Input columns (case-insensitive):
#   Woreda | Kebele | Village | Eligibility | Household Head Name [| HH_ID | Phone | Other ID]
# PPS village counts are derived from this roster (choose: All HHs / Eligible-only / Non-eligible-only)

from pathlib import Path
import io
import math
import hashlib
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional: lightweight PDFs for bulk packs
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# ----------------- Branding -----------------
ASSETS_LOGO = Path("assets/wfp_logo.png")
ASSETS_FAVICON = Path("assets/favicon.png")
PAGE_ICON = str(ASSETS_FAVICON) if ASSETS_FAVICON.exists() else (str(ASSETS_LOGO) if ASSETS_LOGO.exists() else None)

st.set_page_config(page_title="One-file PPS Village & HH Sampler", page_icon=PAGE_ICON, layout="wide")

# Header
def render_header():
    left, mid, right = st.columns([0.12, 0.76, 0.12])
    with left:
        if ASSETS_LOGO.exists():
            st.image(str(ASSETS_LOGO), width=90)
    with mid:
        st.markdown(
            """
            <div style="padding-top:6px;">
              <h1 style="margin-bottom:0;">One-file PPS Village & Household Sampler</h1>
              <p style="margin-top:4px; color: gray;">WFP Ethiopia – Somali Region (Jijiga AO)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("\n\n")
        st.button("📘 Help", key="help_button")
    st.markdown("""
    <style>.block-container { padding-top: 1.0rem !important; }</style>
    """, unsafe_allow_html=True)

render_header()

with st.sidebar:
    if ASSETS_LOGO.exists():
        st.image(str(ASSETS_LOGO), width=140)
    with st.expander("📘 Help & Guide", expanded=False):
        st.markdown(
            """
            **Single file input**  
            Upload one roster with columns (case-insensitive):  
            `Woreda | Kebele | Village | Eligibility | Household Head Name [| HH_ID | Phone | Other ID]`  
            
            **PPS base**  
            Choose whether village 'HHs' used for PPS are derived from:  
            • All households  
            • Eligible-only households  
            • Non-eligible-only households

            **Household sampling modes**  
            • **Stratified**: separate quotas for Eligible & Non-eligible  
            • **Unstratified**: one combined quota (all households)

            **Outputs**  
            • `Sampled_Villages.(csv/xlsx)` with Diagnostics  
            • `HH_Sample.(csv/xlsx)` with HH_Summary  
            • Optional `Bulk_Pack.zip` (per-village CSV + PDF field sheet)
            """
        )

# ----------------- Readers & Normalizers -----------------

def norm_cols(cols):
    return [str(c).strip().lower().replace("\n"," ").replace("\t"," ") for c in cols]


def read_single_file(file) -> pd.DataFrame:
    name = getattr(file, "name", "uploaded").lower()
    ext = name.split(".")[-1]
    try:
        if ext in ["xlsx","xls"]:
            df = pd.read_excel(file, engine=None)
        else:
            try:
                df = pd.read_csv(file)
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding="cp1252")
    except Exception as e:
        raise ValueError(f"Could not read input file: {e}")

    df.columns = norm_cols(df.columns)
    # Map variants
    rename = {}
    for c in df.columns:
        if c in ["woreda"]: rename[c] = "woreda"
        if c in ["kebele"]: rename[c] = "kebele"
        if c in ["village","village / ea","ea","enumeration area"]: rename[c] = "village"
        if c in ["eligibility","eligible_flag","status"]: rename[c] = "eligibility"
        if c in ["household head name","hh head name","head name","hh_name","household_name"]: rename[c] = "head_name"
        if c in ["hh_id","hh id","household id","registration #","registration","id"]: rename[c] = "hh_id"
        if c in ["phone","phone (optional)","phone_number"]: rename[c] = "phone"
        if c in ["other id","other_id","alt id","alt_id"]: rename[c] = "other_id"
    df = df.rename(columns=rename)

    required = ["woreda","kebele","village","eligibility","head_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Ensure at least Woreda, Kebele, Village, Eligibility, Household Head Name.")

    for c in ["woreda","kebele","village","eligibility","head_name"]:
        df[c] = df[c].astype(str).str.strip()

    # Normalize eligibility
    df["eligibility"] = df["eligibility"].str.strip().str.lower().map({
        "eligible":"Eligible",
        "non-eligible":"Non-eligible",
        "non eligible":"Non-eligible",
        "noneligible":"Non-eligible",
        "ineligible":"Non-eligible",
        "e":"Eligible",
        "ne":"Non-eligible",
    }).fillna(df["eligibility"].str.title())

    # Ensure optional columns
    for col in ["hh_id","phone","other_id"]:
        if col not in df.columns:
            df[col] = ""

    return df[["woreda","kebele","village","eligibility","head_name","hh_id","phone","other_id"]]

# ----------------- Utilities -----------------

def rng_for_group(seed_base, *keys):
    if not seed_base:
        return np.random.default_rng()
    s = "::".join([str(seed_base)] + [str(k) for k in keys])
    h = int(hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest(), 16)
    return np.random.default_rng(h)


def _systematic_indices(N: int, n: int, rng: np.random.Generator):
    if N <= 0 or n <= 0:
        return [], None, None
    n_eff = min(int(n), int(N))
    interval = N / n_eff
    start = rng.uniform(0.0, interval)
    idxs = [int(np.floor(start + k*interval)) for k in range(n_eff)]
    seen, out = set(), []
    for i in idxs:
        j = min(max(i,0), N-1)
        if j not in seen:
            seen.add(j); out.append(j)
        if len(out) == n_eff:
            break
    return out, interval, start

# ----------------- Build PPS frame from roster -----------------

def build_pps_frame(roster: pd.DataFrame, pps_base: str, dedup_names: bool):
    r = roster.copy()
    # Optional dedup by exact head_name within (w,k,v)
    if dedup_names:
        r = r.sort_values(["woreda","kebele","village","head_name"]).drop_duplicates(["woreda","kebele","village","head_name"], keep="first")

    if pps_base == "All households":
        grp = r.groupby(["woreda","kebele","village"], as_index=False).size().rename(columns={"size":"hhs"})
    elif pps_base == "Eligible-only":
        grp = r[r["eligibility"]=="Eligible"].groupby(["woreda","kebele","village"], as_index=False).size().rename(columns={"size":"hhs"})
    else:  # Non-eligible-only
        grp = r[r["eligibility"]=="Non-eligible"].groupby(["woreda","kebele","village"], as_index=False).size().rename(columns={"size":"hhs"})

    grp["hhs"] = grp["hhs"].astype(float)
    # Keep only villages with hhs>0
    grp = grp[grp["hhs"]>0].copy()
    return grp

# ----------------- PPS selection per kebele -----------------

def pps_select_systematic(cum_high, rng, m):
    n = len(cum_high)
    if n == 0: return []
    interval = 1.0 / max(m,1)
    start = rng.random() * interval
    idxs = []
    for k in range(m):
        u = start + k*interval
        if u >= 1.0:
            u -= math.floor(u)
        i = int(np.searchsorted(cum_high, u, side="left"))
        if i >= n: i = n-1
        idxs.append(i)
    # unique preserving order
    seen, out = set(), []
    for i in idxs:
        if i not in seen:
            seen.add(i); out.append(i)
    return out


def sample_villages_pps(vdf: pd.DataFrame, method: str, m_default: int, threshold_n: int, m_large: int, use_fixed_m: bool, fixed_m: int, seed_base: str):
    # vdf columns: woreda,kebele,village,hhs
    sampled_rows, diag_rows, kebele_summary = [], [], []
    for (w,k), g in vdf.groupby(["woreda","kebele"], sort=False):
        g = g.reset_index(drop=True)
        nvill = len(g)
        m = max(int(fixed_m if use_fixed_m else (m_large if nvill >= int(threshold_n) else m_default)), 1)
        # compute p and cumulative
        g = g.copy()
        total = g["hhs"].sum()
        g["p"] = g["hhs"] / total if total>0 else 0
        g["cum_high"] = g["p"].cumsum(); g["cum_low"] = g["cum_high"] - g["p"]
        rng = rng_for_group(seed_base, w, k)
        if method == "Systematic":
            idxs = pps_select_systematic(g["cum_high"].to_numpy(), rng, m)
        else:
            # independent PPS with de-dup
            draws = rng.random(m)
            idxs = []
            for u in draws:
                i = int(np.searchsorted(g["cum_high"].to_numpy(), u, side="left"))
                if i >= nvill: i = nvill-1
                if i not in idxs:
                    idxs.append(i)
            # fallback fill if duplicates reduced length
            c=0
            while len(idxs) < min(m, nvill) and c < m*3:
                u = rng.random()
                i = int(np.searchsorted(g["cum_high"].to_numpy(), u, side="left"))
                if i >= nvill: i = nvill-1
                if i not in idxs: idxs.append(i)
                c+=1
        sel = g.iloc[idxs][["woreda","kebele","village","hhs"]]
        sampled_rows.append(sel)
        # diagnostics
        d = g.copy(); d["method"] = method; d["m_requested"] = m; d["m_final"] = len(idxs); d["selected"] = False
        d.loc[d.index.isin(idxs), "selected"] = True
        diag_rows.append(d)
        kebele_summary.append({"Woreda":w, "Kebele":k, "#Villages":nvill, "Method":method, "m_used":int(m), "Total HHs":int(total)})
    sampled = pd.concat(sampled_rows, ignore_index=True) if sampled_rows else vdf.iloc[0:0]
    diagnostics = pd.concat(diag_rows, ignore_index=True) if diag_rows else vdf.iloc[0:0]
    summary = pd.DataFrame(kebele_summary)
    return sampled, diagnostics, summary

# ----------------- HH Systematic Sampling -----------------

def to_excel_bytes(df1: pd.DataFrame, name1: str, df2: pd.DataFrame, name2: str) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        df1.to_excel(w, index=False, sheet_name=name1)
        df2.to_excel(w, index=False, sheet_name=name2)
    return out.getvalue()

# Stratified: Eligible / Non-eligible

def sample_households_stratified(roster: pd.DataFrame, sampled_villages: pd.DataFrame, default_nE: int, default_nNE: int, order_by: str, seed_base: str):
    sv = sampled_villages[["woreda","kebele","village"]].drop_duplicates().copy()
    out_rows, sum_rows = [], []

    for _, row in sv.iterrows():
        w,k,v = row.woreda, row.kebele, row.village
        sub = roster[(roster["woreda"]==w) & (roster["kebele"]==k) & (roster["village"]==v)].copy()
        for g, n_target in [("Eligible", default_nE), ("Non-eligible", default_nNE)]:
            gdf = sub[sub["eligibility"]==g].copy(); N = len(gdf)
            if N==0 or n_target<=0:
                sum_rows.append({"Woreda":w,"Kebele":k,"Village":v,"Group":g,"N":int(N),"n":int(n_target),"Interval":None,"Start":None,"Picked":0}); continue
            # Choose order
            col = order_by.strip().lower()
            if col in gdf.columns:
                gdf = gdf.sort_values(col, kind="mergesort")
            else:
                gdf = gdf.sort_values(["head_name","hh_id"], na_position="last", kind="mergesort")
            rng = rng_for_group(seed_base, w,k,v,g)
            idxs, interval, start = _systematic_indices(N, int(n_target), rng)
            pick = gdf.iloc[idxs].copy()
            pick.insert(0, "Sample_Order", range(1, len(pick)+1))
            pick.insert(0, "Eligibility", g)
            pick.insert(0, "Village", v)
            pick.insert(0, "Kebele", k)
            pick.insert(0, "Woreda", w)
            out_rows.append(pick[["Woreda","Kebele","Village","Eligibility","Sample_Order","hh_id","head_name","phone","other_id"]])
            sum_rows.append({"Woreda":w,"Kebele":k,"Village":v,"Group":g,"N":int(N),"n":int(n_target),"Interval":round(interval,3),"Start":round(float(start),3),"Picked":len(pick)})

    hh_sample = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=["Woreda","Kebele","Village","Eligibility","Sample_Order","hh_id","head_name","phone","other_id"])
    hh_summary = pd.DataFrame(sum_rows)
    return hh_sample, hh_summary

# Unstratified: All households combined

def sample_households_unstratified(roster: pd.DataFrame, sampled_villages: pd.DataFrame, n_default: int, order_by: str, seed_base: str):
    sv = sampled_villages[["woreda","kebele","village"]].drop_duplicates().copy()
    out_rows, sum_rows = [], []

    for _, row in sv.iterrows():
        w,k,v = row.woreda, row.kebele, row.village
        sub = roster[(roster["woreda"]==w) & (roster["kebele"]==k) & (roster["village"]==v)].copy()
        N = len(sub)
        n_target = int(n_default)
        if N==0 or n_target<=0:
            sum_rows.append({"Woreda":w,"Kebele":k,"Village":v,"Group":"All","N":int(N),"n":int(n_target),"Interval":None,"Start":None,"Picked":0}); continue
        # Choose order
        col = order_by.strip().lower()
        if col in sub.columns:
            sub = sub.sort_values(col, kind="mergesort")
        else:
            sub = sub.sort_values(["head_name","hh_id"], na_position="last", kind="mergesort")
        rng = rng_for_group(seed_base, w,k,v,"All")
        idxs, interval, start = _systematic_indices(N, int(n_target), rng)
        pick = sub.iloc[idxs].copy()
        pick.insert(0, "Sample_Order", range(1, len(pick)+1))
        # Keep Eligibility column for information (not stratification)
        pick.insert(0, "Eligibility", pick.get("eligibility",""))
        pick.insert(0, "Village", v)
        pick.insert(0, "Kebele", k)
        pick.insert(0, "Woreda", w)
        out_rows.append(pick[["Woreda","Kebele","Village","Eligibility","Sample_Order","hh_id","head_name","phone","other_id"]])
        sum_rows.append({"Woreda":w,"Kebele":k,"Village":v,"Group":"All","N":int(N),"n":int(n_target),"Interval":round(interval,3),"Start":round(float(start),3),"Picked":len(pick)})

    hh_sample = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=["Woreda","Kebele","Village","Eligibility","Sample_Order","hh_id","head_name","phone","other_id"])
    hh_summary = pd.DataFrame(sum_rows)
    return hh_sample, hh_summary

# ----------------- Bulk Pack -----------------

def _safe_name(s: str) -> str:
    return ''.join([c if c.isalnum() or c in [' ','_','-'] else '_' for c in str(s)]).strip().replace(' ','_')


def _village_pdf_bytes(w, k, v, hh_df: pd.DataFrame, sum_df: pd.DataFrame, app_title: str = "PPS & HH Sampler") -> bytes:
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, leftMargin=1.6*cm, rightMargin=1.6*cm, topMargin=1.6*cm, bottomMargin=1.6*cm)
    styles = getSampleStyleSheet(); title = styles['Title']; title.textColor = colors.HexColor('#1F77B4')
    h2 = styles['Heading2']; h2.textColor = colors.HexColor('#1F77B4'); body = styles['BodyText']
    elems = []
    elems.append(Paragraph(f"<b>{app_title}</b>", title)); elems.append(Spacer(1,8))
    elems.append(Paragraph(f"<b>Woreda:</b> {w} &nbsp;&nbsp; <b>Kebele:</b> {k} &nbsp;&nbsp; <b>Village:</b> {v}", body))
    sums = sum_df[(sum_df['Woreda']==w) & (sum_df['Kebele']==k) & (sum_df['Village']==v)].copy()
    data = [["Group","N (frame)","n (target)","Interval","Start","Picked"]]
    for _, r in sums.iterrows():
        data.append([r.get('Group',''), r.get('N',''), r.get('n',''), r.get('Interval',''), r.get('Start',''), r.get('Picked','')])
    t = Table(data, hAlign='LEFT'); t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#E2F0D9')),
        ('GRID',(0,0),(-1,-1), 0.4, colors.grey),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('ALIGN',(1,1),(-1,-1),'CENTER')
    ]))
    elems.append(Spacer(1,6)); elems.append(Paragraph("Household sampling summary", h2)); elems.append(t)
    head = hh_df[(hh_df['Woreda']==w) & (hh_df['Kebele']==k) & (hh_df['Village']==v)].head(12)
    if not head.empty:
        td = [["Elig","Order","HH_ID","Head Name","Phone","Other ID"]]
        for _, rr in head.iterrows():
            td.append([rr.get('Eligibility',''), rr.get('Sample_Order',''), rr.get('hh_id',''), rr.get('head_name',''), rr.get('phone',''), rr.get('other_id','')])
        t2 = Table(td, hAlign='LEFT', colWidths=[2*cm,2*cm,3*cm,6*cm,3.5*cm,3.5*cm])
        t2.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#D9E1F2')),
            ('GRID',(0,0),(-1,-1), 0.4, colors.grey),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold')
        ]))
        elems.append(Spacer(1,6)); elems.append(Paragraph("Sample preview (first 12)", h2)); elems.append(t2)
    elems.append(Spacer(1,12)); elems.append(Paragraph("Field notes: _______________________________________________", body))
    elems.append(Spacer(1,6)); elems.append(Paragraph("Enumerator: _____________________   Team Lead: _____________________", body))
    doc.build(elems); return buff.getvalue()


def build_bulk_pack_zip(hh_sample: pd.DataFrame, hh_summary: pd.DataFrame, include_pdfs=True, include_csvs=True, folder_by='kebele', app_title='PPS & HH Sampler') -> bytes:
    if hh_sample is None or hh_sample.empty:
        raise ValueError("No household sample available. Run HH sampling first.")
    bytestream = io.BytesIO()
    with zipfile.ZipFile(bytestream, 'w', zipfile.ZIP_DEFLATED) as z:
        villages = hh_sample[["Woreda","Kebele","Village"]].drop_duplicates().values.tolist()
        for w,k,v in villages:
            safe = {'w': _safe_name(w), 'k': _safe_name(k), 'v': _safe_name(v)}
            if folder_by == 'woreda': folder = f"{safe['w']}/{safe['k']}/{safe['v']}"
            elif folder_by == 'kebele': folder = f"{safe['k']}/{safe['v']}"
            else: folder = f"{safe['w']}_{safe['k']}_{safe['v']}"
            sub = hh_sample[(hh_sample['Woreda']==w) & (hh_sample['Kebele']==k) & (hh_sample['Village']==v)]
            if include_csvs:
                z.writestr(f"{folder}/{safe['v']}_HH_Sample.csv", sub.to_csv(index=False).encode('utf-8'))
            if include_pdfs:
                z.writestr(f"{folder}/{safe['v']}_FieldSheet.pdf", _village_pdf_bytes(w,k,v, hh_sample, hh_summary, app_title=app_title))
    bytestream.seek(0); return bytestream.getvalue()

# ----------------- UI -----------------

st.subheader(" Upload single roster (Excel/CSV)")
single_file = st.file_uploader("One file with: Woreda | Kebele | Village | Eligibility | Household Head Name [| HH_ID | Phone | Other ID]", type=["xlsx","xls","csv"]) 

if single_file is not None:
    try:
        roster = read_single_file(single_file)
        st.success(f"Loaded {len(roster):,} rows, across {roster[['woreda','kebele','village']].drop_duplicates().shape[0]} distinct villages.")
        with st.expander("Roster preview (top 25)"):
            st.dataframe(roster.head(25), use_container_width=True)

        st.markdown("---")
        colA, colB = st.columns(2)
        with colA:
            pps_base = st.selectbox("PPS base (for village HHs)", ["All households","Eligible-only","Non-eligible-only"], index=0)
            dedup_names = st.checkbox("Treat duplicate head names within a village as one HH (keep first)", value=False)
        with colB:
            method = st.selectbox("PPS method", ["Systematic","Independent"], index=0)
            use_fixed_m = st.checkbox("Use fixed m for all kebeles", value=False)
            if use_fixed_m:
                fixed_m = st.number_input("Fixed m", min_value=1, max_value=30, value=2, step=1)
                m_default, threshold_n, m_large = 2, 7, 4
            else:
                m_default = st.number_input("Default m", min_value=1, max_value=30, value=2, step=1)
                threshold_n = st.number_input("If kebele has ≥ (villages)", min_value=2, max_value=1000, value=7, step=1)
                m_large = st.number_input("Use m =", min_value=1, max_value=30, value=4, step=1)
                fixed_m = m_default
        seed_base = st.text_input("Random seed (optional)", value="")

        pps_frame = build_pps_frame(roster, pps_base, dedup_names)
        with st.expander("Derived village frame for PPS (from your file)"):
            st.dataframe(pps_frame, use_container_width=True, height=300)

        if st.button(" Run PPS Village Sampling", type="primary"):
            sampled_villages, diag, summary = sample_villages_pps(pps_frame, method, m_default, threshold_n, m_large, use_fixed_m, fixed_m, seed_base)
            st.subheader("✅ Sampled Villages")
            st.dataframe(sampled_villages, use_container_width=True, height=300)
            st.subheader("📋 Kebele Summary")
            st.dataframe(summary, use_container_width=True, height=220)
            st.markdown("### ⬇️ Download (villages)")
            st.download_button("Sampled_Villages.csv", data=sampled_villages.to_csv(index=False).encode('utf-8'), file_name="Sampled_Villages.csv", mime="text/csv")
            st.download_button("Sampled_Villages.xlsx (with Diagnostics)", data=to_excel_bytes(sampled_villages, "Sampled_Villages", diag, "Diagnostics"), file_name="Sampled_Villages.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.markdown("---")
            st.subheader("🏠 Household Systematic Sampling (within sampled villages)")
            # NEW: Mode toggle
            mode = st.radio("Sampling mode", ["Stratified (Eligible / Non-eligible)", "Unstratified (All households)"], index=0)

            if mode.startswith("Stratified"):
                col1, col2 = st.columns(2)
                with col1:
                    default_nE = st.number_input("Eligible per village (default)", min_value=0, max_value=9999, value=15, step=1)
                with col2:
                    default_nNE = st.number_input("Non-eligible per village (default)", min_value=0, max_value=9999, value=15, step=1)
                order_by = st.text_input("Order by column (in your roster)", value="hh_id", help="Used for systematic skip; fallback = head_name → hh_id")
                hh_seed = st.text_input("Random seed (HH sampling, optional)", value=seed_base)
                if st.button("▶️ Run HH Sampling", type="primary"):
                    hh_sample, hh_summary = sample_households_stratified(roster, sampled_villages, default_nE, default_nNE, order_by.strip().lower(), hh_seed)
                    if hh_sample.empty:
                        st.warning("No households selected. Check quotas vs. availability.")
                    st.subheader("✅ Household Sample")
                    st.dataframe(hh_sample, use_container_width=True, height=360)
                    st.subheader("📊 HH Summary / Diagnostics")
                    st.dataframe(hh_summary, use_container_width=True, height=240)
                    st.markdown("### ⬇️ Download (households)")
                    st.download_button("HH_Sample.csv", data=hh_sample.to_csv(index=False).encode('utf-8'), file_name="HH_Sample.csv", mime="text/csv")
                    st.download_button("HH_Sample.xlsx (with HH_Summary)", data=to_excel_bytes(hh_sample, "HH_Sample", hh_summary, "HH_Summary"), file_name="HH_Sample.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                    # Bulk pack
                    st.markdown("---")
                    st.subheader("📦 One‑click Bulk Pack (ZIP)")
                    colx, coly, colz = st.columns(3)
                    with colx: include_pdfs = st.checkbox("Include PDFs", value=True)
                    with coly: include_csvs = st.checkbox("Include CSVs", value=True)
                    with colz: folder_by = st.selectbox("Folder structure", ["kebele","woreda","flat"], index=0)
                    app_title = st.text_input("PDF header title", value="WFP – PPS & Household Sampler")
                    if st.button("📦 Build Bulk Pack ZIP", type="secondary"):
                        try:
                            zip_bytes = build_bulk_pack_zip(hh_sample, hh_summary, include_pdfs=include_pdfs, include_csvs=include_csvs, folder_by=folder_by, app_title=app_title)
                            st.download_button("Download Bulk_Pack.zip", data=zip_bytes, file_name="Bulk_Pack.zip", mime="application/zip")
                            st.success("Bulk pack prepared. Click the button to download.")
                        except Exception as e:
                            st.error(f"Bulk pack error: {e}")

            else:
                # Unstratified mode
                col1, col2 = st.columns(2)
                with col1:
                    n_default = st.number_input("Households per village (n)", min_value=0, max_value=9999, value=30, step=1)
                with col2:
                    order_by = st.text_input("Order by column (in your roster)", value="hh_id", help="Used for systematic skip; fallback = head_name → hh_id")
                hh_seed = st.text_input("Random seed (HH sampling, optional)", value=seed_base)
                if st.button("▶️ Run HH Sampling", type="primary"):
                    hh_sample, hh_summary = sample_households_unstratified(roster, sampled_villages, n_default, order_by.strip().lower(), hh_seed)
                    if hh_sample.empty:
                        st.warning("No households selected. Check 'n' vs. availability.")
                    st.subheader("✅ Household Sample (Unstratified)")
                    st.dataframe(hh_sample, use_container_width=True, height=360)
                    st.subheader("📊 HH Summary / Diagnostics")
                    st.dataframe(hh_summary, use_container_width=True, height=240)
                    st.markdown("### ⬇️ Download (households)")
                    st.download_button("HH_Sample.csv", data=hh_sample.to_csv(index=False).encode('utf-8'), file_name="HH_Sample.csv", mime="text/csv")
                    st.download_button("HH_Sample.xlsx (with HH_Summary)", data=to_excel_bytes(hh_sample, "HH_Sample", hh_summary, "HH_Summary"), file_name="HH_Sample.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                    # Bulk pack
                    st.markdown("---")
                    st.subheader("📦 One‑click Bulk Pack (ZIP)")
                    colx, coly, colz = st.columns(3)
                    with colx: include_pdfs = st.checkbox("Include PDFs", value=True)
                    with coly: include_csvs = st.checkbox("Include CSVs", value=True)
                    with colz: folder_by = st.selectbox("Folder structure", ["kebele","woreda","flat"], index=0)
                    app_title = st.text_input("PDF header title", value="WFP – PPS & Household Sampler")
                    if st.button("📦 Build Bulk Pack ZIP", type="secondary"):
                        try:
                            zip_bytes = build_bulk_pack_zip(hh_sample, hh_summary, include_pdfs=include_pdfs, include_csvs=include_csvs, folder_by=folder_by, app_title=app_title)
                            st.download_button("Download Bulk_Pack.zip", data=zip_bytes, file_name="Bulk_Pack.zip", mime="application/zip")
                            st.success("Bulk pack prepared. Click the button to download.")
                        except Exception as e:
                            st.error(f"Bulk pack error: {e}")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a single roster file to begin.")

st.markdown("<hr style='margin-top:2rem;margin-bottom:0.5rem;'><div style='color:gray;font-size:0.9em;'>© WFP Ethiopia – Somali Region (Jijiga AO) | One-file PPS & HH sampling utility</div>", unsafe_allow_html=True)
