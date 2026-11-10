"""
customer_matrix.py — shrink a matrix the customer sends back.

The customer takes our matrix, applies their own factor, sorts it on THEIR
price (cheapest first) and returns it WITHOUT prices. CargoWrite takes the
first row an order fits, so the row order alone carries their prices: row 1 is
their cheapest. Here every row gets its position as price and the routing
table (routing.py) keeps only the rows that decide something — the result
routes every order like their full list does, in their own columns.

Rows picked by name, not by price, are kept unchanged in their order: a
customer name or service code in USER_DEF_TYPE_1 first at its country, the
PRIO rows together at the very end of the file (as the customer has them).
"""
import re

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import PatternFill

import routing

# Old layout: USER_DEF_TYPE_4 is what is now USER_DEF_NUM_1.
COLUMN_ALIASES = {'USER_DEF_TYPE_4': 'USER_DEF_NUM_1'}
NUMERIC_COLS = ['MIN_VOLUME', 'MAX_VOLUME', 'MIN_WEIGHT', 'MAX_WEIGHT',
                'MIN_PARCEL', 'MAX_PARCEL', 'EACH_WEIGHT', 'EACH_VOLUME']
# The DHL catch-all (only if nothing else fits) stays the last row of a country.
SENTINEL_MIN_KG = 24000
# Prio orders are picked by USER_DEF_TYPE_1 'PRIO'.
PRIO = 'PRIO'
# Fixed service-code rows carry this marker in USER_DEF_TYPE_1.
SERVICE_CODE_MARK = 1e-12

_CELL = re.compile(r'([A-Z]{1,3})(\d+)')


def _col_index(letters):
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def _eval_formulas(df, raw):
    """Evaluate the simple cell formulas of the file (=L2*M2, =J2/167 ...).
    `raw` holds the cell values as read; formulas are strings starting '='."""
    cols = list(df.columns)
    for c in cols:
        s = raw[c]
        is_f = s.map(lambda v: isinstance(v, str) and v.startswith('='))
        if not is_f.any():
            continue
        out = s.where(~is_f, np.nan).astype(object)
        # group by the formula with row numbers removed: one vectorised
        # evaluation per pattern
        pattern = s[is_f].map(lambda v: _CELL.sub(lambda m: m.group(1) + '#', v))
        for pat, idx in pattern.groupby(pattern).groups.items():
            expr = pat[1:]
            refs = {}

            def _ref(m):
                name = cols[_col_index(m.group(1))]
                key = f'c{len(refs)}'
                refs[key] = pd.to_numeric(df.loc[idx, name], errors='coerce').to_numpy(float)
                return key
            expr = re.sub(r'([A-Z]{1,3})#', _ref, expr)
            if not re.fullmatch(r'[\sc0-9.+\-*/()]*', expr):
                continue                               # not plain arithmetic
            try:
                out.loc[idx] = eval(expr, {'__builtins__': {}}, refs)
            except Exception:
                pass
        df[c] = out
    return df


def read_customer_matrix(path):
    """Read the customer's sorted matrix. Returns (frame, header): the frame
    in the file's row order with numeric values, `header` its own columns."""
    name = str(getattr(path, 'name', path)).lower()
    if name.endswith('.xlsb'):
        # Excel binary workbook: values only (no formulas to evaluate)
        if hasattr(path, 'seek'):
            path.seek(0)
        raw = pd.read_excel(path, sheet_name=0, engine='calamine', dtype=object)
        raw = raw.loc[:, [c for c in raw.columns if not str(c).startswith('Unnamed')]]
        raw = raw.dropna(how='all').reset_index(drop=True)
        raw = raw.astype(object).where(raw.notna(), None)
        header = list(raw.columns)
    else:
        wb = load_workbook(path, read_only=True)
        ws = wb.worksheets[0]
        rows = ws.iter_rows(values_only=True)
        header = [h for h in next(rows)]
        while header and header[-1] is None:
            header.pop()
        data = [list(r[:len(header)]) for r in rows if any(v is not None for v in r)]
        wb.close()
        raw = pd.DataFrame(data, columns=header)
    df = raw.copy()
    for _ in range(3):                  # formulas that use other formulas
        df = _eval_formulas(df, raw)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'POSTCODE' in df.columns:
        df['POSTCODE'] = df['POSTCODE'].map(
            lambda v: None if v is None or (isinstance(v, float) and np.isnan(v))
            else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)))
    df = df.rename(columns=COLUMN_ALIASES)
    return df, header


def _by_name(df):
    """Rows CargoWrite picks by name / code, not by price."""
    u1 = df['USER_DEF_TYPE_1'] if 'USER_DEF_TYPE_1' in df.columns else pd.Series(None, index=df.index)
    num = pd.to_numeric(u1.map(lambda v: str(v).replace(',', '.') if v is not None else None),
                        errors='coerce')
    text = u1.notna() & num.isna() & (u1.astype(str).str.strip() != '')
    code = num.notna() & (num.abs() < SERVICE_CODE_MARK)
    return text | code


def compact_customer_matrix(df, progress=None):
    """Shrink the customer's matrix per country. Returns (frame, stats per
    country). `progress(i, n, country)` is called before each country."""
    countries = list(dict.fromkeys(df['COUNTRYISO2']))
    prio = (df['USER_DEF_TYPE_1'].map(str) == PRIO) if 'USER_DEF_TYPE_1' in df.columns \
        else pd.Series(False, index=df.index)
    parts, stats = [], {}
    for i, country in enumerate(countries):
        if progress:
            progress(i, len(countries), country)
        g = df[df['COUNTRYISO2'] == country].copy()
        g['TOTAL_PRICE'] = np.arange(1, len(g) + 1, dtype=float)   # their order
        sentinel = ((g['CARRIER_ID'] == 'DHL-FENDER') & g['POSTCODE'].isna()
                    & (g['MAX_WEIGHT'] >= SENTINEL_MIN_KG))
        g.loc[sentinel, 'TOTAL_PRICE'] = 999999.0
        named = _by_name(g)
        fixed = g[named & ~prio[g.index]]
        routed, st = routing.build_routing(g[~named], blank_postcode=False, generic=False)
        stats[country] = {'rows_in': len(g),
                          'rows_out': int(named.sum()) + len(routed),
                          'mismatch': st['mismatch']}
        parts += [fixed, routed]
    parts.append(df[prio])                     # PRIO rows: all at the end
    out = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True)
    return out, stats


def write_customer_matrix(df, header, path):
    """Write the shrunk matrix with the customer's own columns (no prices
    unless their file had them), values only. Rows picked by name (PRIO,
    customer) are yellow, as in our own matrix."""
    named = _by_name(df).to_numpy()
    back = {v: k for k, v in COLUMN_ALIASES.items() if k in header}
    df = df.rename(columns=back)
    wb = Workbook(write_only=True)
    ws = wb.create_sheet('Matrix')
    ws.append(header)
    fill = PatternFill('solid', fgColor='FFF2CC')
    cols = [c if c in df.columns else None for c in header]
    for i, row in enumerate(df.itertuples(index=False, name=None)):
        rec = dict(zip(df.columns, row))
        vals = []
        for c in cols:
            v = rec.get(c) if c else None
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = None
            elif isinstance(v, np.generic):
                v = v.item()
            if named[i]:
                v = WriteOnlyCell(ws, value=v)
                v.fill = fill
            vals.append(v)
        ws.append(vals)
    wb.save(path)
    return path
