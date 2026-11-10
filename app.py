"""
app.py — Matrix Shrinker (Rate Matrix Builder 2)  ·  Streamlit UI

The customer takes the matrix from Rate Matrix Builder 1, applies their own
factor, sorts it on their price and sends it back without prices. This app
shrinks that matrix: per country only the rows that decide something remain,
in the customer's own columns, ready to send back (customer_matrix.py).

Run locally:   streamlit run app.py
"""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import customer_matrix as cmx

st.set_page_config(page_title="Matrix Shrinker", page_icon="✂️", layout="centered")

st.title("✂️ Matrix Shrinker")
st.caption("Upload the combined matrix the customer sent back — all countries, sorted on "
           "their price, without prices. Every country is shrunk to the rows that decide "
           "something. The result has the customer's own columns and can be sent back as is.")

with st.expander("How does it work?"):
    st.markdown(
        "- CargoWrite takes the **first row an order fits**. The customer's file is sorted "
        "on their price, so the row order *is* their price: row 1 is their cheapest.\n"
        "- Per country the app keeps only the rows where the choice of carrier changes "
        "(by number of parcels and weight). Every order still lands on the same carrier "
        "as in the customer's full list.\n"
        "- Rows picked by name — PRIO and customer names (e.g. Audio Partner) — are kept "
        "unchanged and shown in yellow; the PRIO rows at the end of the file. The DHL "
        "catch-all stays the last row of each country.\n"
        "- No rate card is needed.")

uploaded = st.file_uploader("Customer matrix (.xlsx / .xlsb)", type=["xlsx", "xlsb"])

if uploaded is not None and st.button("✂️ Shrink", type='primary', width='stretch'):
    bar = st.progress(0.0, text="Reading the file…")
    df, header = cmx.read_customer_matrix(uploaded)
    rows_in = len(df)

    def _progress(i, n, country):
        bar.progress(i / max(n, 1), text=f"Shrinking {country}…  ({i + 1}/{n})")
    out, stats = cmx.compact_customer_matrix(df, progress=_progress)
    del df                                       # free the full matrix before writing
    bar.progress(1.0, text="Writing the file…")
    path = Path(tempfile.mkdtemp()) / f"{Path(uploaded.name).stem}_compact.xlsx"
    cmx.write_customer_matrix(out, header, path)
    st.session_state['result'] = (path.name, path.read_bytes(), stats, rows_in, len(out))
    bar.empty()

if st.session_state.get('result'):
    name, data, stats, rows_in, rows_out = st.session_state['result']
    st.success(f"{rows_in:,} rows → **{rows_out:,} rows** ({len(stats)} countries).")
    st.download_button(f"📥 Download {name}", data=data, file_name=name, type='primary',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                       width='stretch')
    st.dataframe(pd.DataFrame([{'Country': c, 'Rows in': f"{v['rows_in']:,}",
                                'Rows out': f"{v['rows_out']:,}", 'Check': v['mismatch']}
                               for c, v in stats.items()]).set_index('Country'),
                 width='stretch')
    st.caption("**Check** = test orders exactly on a weight boundary (within a gram) that "
               "land on another carrier than in the customer's full list — rounding of the "
               "parcel weights; 0 or a handful per country is normal.")
