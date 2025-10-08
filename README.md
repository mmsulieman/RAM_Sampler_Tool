
# One-file PPS Village & Household Sampler (Streamlit) — v2 (with Unstratified mode)

**Input:** Single Excel/CSV with columns (case-insensitive):
```
Woreda | Kebele | Village | Eligibility | Household Head Name [| HH_ID | Phone | Other ID]
```
**PPS base:** Choose whether village HHs = count of All, Eligible-only, or Non-eligible-only rows.

**Household sampling modes:**
- **Stratified:** separate quotas for Eligible & Non-eligible per village.
- **Unstratified:** one combined quota `n` per village across all households.

**Outputs:**
- `Sampled_Villages.csv` + `Sampled_Villages.xlsx (Diagnostics)`
- `HH_Sample.csv` + `HH_Sample.xlsx (HH_Summary)`
- Optional `Bulk_Pack.zip` (per-village CSV + PDF field sheet)

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
- Push these files to GitHub
- On https://streamlit.io/cloud → **New app** → file = `app.py` → **Deploy**
