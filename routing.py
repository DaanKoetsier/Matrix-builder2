"""
routing.py — turn a minimal matrix into a carrier routing table.

Per postcode the table walks the number of parcels upward (1, 2, 3 …) and,
for each count, lists which carrier is the cheapest as the total weight goes
up. Parcel counts with exactly the same choices are merged into one block
(MIN_PARCEL – MAX_PARCEL). Rows come out in that order: parcel count up, then
weight up; CargoWrite takes the first row an order fits, so no prices are
needed and the price columns are left empty.

Assumptions (agreed with the user):
  * the parcels of an order weigh the same — how the weight is spread does
    not matter, only that no parcel is heavier than the carrier takes
    (EACH_WEIGHT = the carrier's real per-parcel maximum);
  * volume is not used to pick the carrier; MAX/EACH_VOLUME follow from the
    weights with the carrier's own volume divisor;
  * weights are rounded to whole grams (70.000002 kg → 70 kg).
"""
import numpy as np
import pandas as pd

INF = np.inf

# Real per-parcel maximum of the UPS rate tables (pipeline.UPS_MAX_PARCEL_KG).
# Other carriers: the highest EACH_WEIGHT they have in the country's matrix.
# Carriers whose EACH_WEIGHT marks a separate tariff (DPD small 3 kg vs normal 31.5 kg).
TARIFF_BY_EACH_WEIGHT = {'DPD'}

CARRIER_EACH_CAP = {'UPDE': 70.0, 'UPSNL': 70.0, 'UPSGB': 70.0}

OUTCOME_COLS = ['CARRIER_ID', 'SERVICE_LEVEL']
# Rows that only apply to particular orders (customer, oversize, hazmat …)
SPECIAL_COLS = ['USER_DEF_TYPE_1', 'USER_DEF_TYPE_2', 'USER_DEF_TYPE_3',
                'USER_DEF_NUM_1', 'HAZMAT', 'AWKWARD']
PRICE_COLS = ['RATE_BASE', 'RATE_BASE2.0', 'RATE_EXTRA', 'MOBILITY', 'FUEL', 'MAUT',
              'Linehaul UPSDE', 'TOLL', 'ADMIN', 'TOTAL_PRICE']
EPS = 1e-6


def _num(s, default):
    v = pd.to_numeric(s, errors='coerce')
    return v.fillna(default).to_numpy(float)


def _blank(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ('', 'nan', 'None')


def _segments(n, nlo, nhi, wlo, whi, each, price, out):
    """Cheapest outcome along the total weight for n equal parcels:
    [(outcome, lo, hi, row)], consecutive equal outcomes merged."""
    m = (nlo <= n) & (nhi >= n)
    if not m.any():
        return []
    idx = np.flatnonzero(m)
    bps = np.concatenate([whi[idx], wlo[idx], each[idx] * n])
    bps = np.unique(np.round(bps[np.isfinite(bps) & (bps > 0)], 6))
    segs, prev, gap = [], 0.0, False
    for b in bps:
        fit = idx[(wlo[idx] < b - EPS / 2) & (whi[idx] >= b - EPS / 2)
                  & (each[idx] * n >= b - EPS / 2)]
        if len(fit):
            k = fit[np.argmin(price[fit])]
            o = out[k]
            # same carrier on both sides of a weight gap no row covers (e.g.
            # the DHL band 22000-22500 kg missing from the pallet file):
            # one row straight through
            if segs and segs[-1][0] == o and (abs(segs[-1][2] - prev) < EPS or gap):
                segs[-1][2] = b
            else:
                segs.append([o, prev, b, k])
            gap = False
        elif segs:
            gap = True
        prev = b
    return segs


def _first_match(rows, n, w):
    """Outcome of the first row an order of n equal parcels / w kg fits."""
    for o, pmin, pmax, wmin, wmax, e in rows:
        if pmin <= n <= pmax and wmin < w <= wmax + EPS / 2 and w / n <= e + EPS / 2:
            return o
    return -1


def _route_group(g, out_codes, caps, vratio):
    nlo = np.maximum(_num(g.get('MIN_PARCEL'), 1), 1)
    nhi = _num(g.get('MAX_PARCEL'), INF)
    wlo = np.round(_num(g.get('MIN_WEIGHT'), 0), 3)
    whi = _num(g.get('MAX_WEIGHT'), INF)
    whi = np.where(np.isfinite(whi), np.round(whi, 3), INF)
    each = _num(g.get('EACH_WEIGHT'), INF)
    each = np.where(np.isfinite(each), np.ceil(each * 1e4 - 1e-6) / 1e4, INF)
    price = _num(g.get('TOTAL_PRICE'), INF)
    out = out_codes

    finite_n = nhi[np.isfinite(nhi)]
    nmax = int(finite_n.max()) if len(finite_n) else 0
    open_end = bool(np.isinf(nhi).any())
    last_n = nmax + 1 if open_end else nmax

    # per parcel count: segments. Consecutive counts merge into one block when
    # every segment has the same carrier and lower limit, and its upper limit
    # is either the same total weight or the carrier's per-parcel maximum × n
    # (31.5 kg × n) for every count in the block.
    per_n = []
    for n in range(1, last_n + 1):
        segs = _segments(n, nlo, nhi, wlo, whi, each, price, out)
        items, prev_hi = [], 0.0
        for o, lo, hi, k in segs:
            lo_r = None if abs(lo - prev_hi) < EPS else round(lo, 3)
            items.append({'o': o, 'lo': lo_r, 'hi': round(hi, 3), 'k': k,
                          'is_cap': abs(hi - caps.get(o, INF) * n) < 1e-3})
            prev_hi = hi
        per_n.append((n, items))

    # Pallets (rows without a parcel limit): the carrier that takes over above
    # the tipping point of every parcel count gets ONE row at the end of the
    # postcode instead of one in every parcel block. Read top-down, an order
    # only reaches it once no parcel row above fits. Pallet stretches in the
    # middle of a block (cheaper between two parcel rows) stay where they are.
    trimmed = False
    if open_end and len(per_n) > 1 and per_n[-1][1]:
        tail = [(t['o'], t['lo'], t['hi']) for t in per_n[-1][1]]
        k = len(tail)
        for i in range(len(per_n) - 1):
            n, items = per_n[i]
            if len(items) >= k and [(x['o'], x['lo'], x['hi']) for x in items[-k:]] == tail:
                per_n[i] = (n, items[:-k])
                trimmed = True

    blocks = []
    for n, items in per_n:
        b = blocks[-1] if blocks else None
        if (b is not None and len(b['items']) == len(items)
                and all(x['o'] == y['o'] and x['lo'] == y['lo']
                        and ((x['num'] and x['hi'] == y['hi']) or (x['cap'] and y['is_cap']))
                        for x, y in zip(b['items'], items))):
            for x, y in zip(b['items'], items):
                x['num'] = x['num'] and x['hi'] == y['hi']
                x['cap'] = x['cap'] and y['is_cap']
            b['n_to'] = n
        else:
            blocks.append({'n_from': n, 'n_to': n,
                           'items': [dict(y, num=True, cap=y['is_cap']) for y in items]})

    if trimmed:
        blocks[-1]['n_from'] = 1                  # the pallet tail: any parcel count
    # The pallet carrier (it has the last word for every parcel count) also
    # starts at 1 parcel on its rows in the middle of a block: an order with
    # fewer parcels that gets that far would land on the pallet row at the end
    # anyway. MAX_PARCEL stays, so heavier-count orders still reach the parcel
    # rows below.
    tail_outs = {it['o'] for it in blocks[-1]['items']} if trimmed else set()
    rows_out, sim = [], []
    for b in blocks:
        n_to_open = open_end and b['n_to'] == last_n
        n_from = b['n_from']
        for it in b['items']:
            o, lo_r, k = it['o'], it['lo'], it['k']
            cap = caps.get(o, INF)
            if it['num']:
                max_w = it['hi']
            else:
                max_w = None if n_to_open else round(cap * b['n_to'], 3)
            r = g.iloc[k].copy()
            p_from = 1 if it['o'] in tail_outs else n_from
            r['MIN_PARCEL'] = p_from
            r['MAX_PARCEL'] = None if n_to_open else b['n_to']
            r['MIN_WEIGHT'] = lo_r
            r['MAX_WEIGHT'] = max_w
            r['EACH_WEIGHT'] = None if not np.isfinite(cap) else cap
            vr = vratio.get(o)
            r['MIN_VOLUME'] = None
            r['MAX_VOLUME'] = None if (vr is None or max_w is None) else max_w * vr
            r['EACH_VOLUME'] = None if (vr is None or not np.isfinite(cap)) else cap * vr
            for c in PRICE_COLS:
                if c in r.index:
                    r[c] = None
            rows_out.append(r)
            sim.append((o, p_from, INF if n_to_open else b['n_to'],
                        -INF if lo_r is None else lo_r,
                        INF if max_w is None else max_w,
                        INF if not np.isfinite(cap) else cap))

    # Catch-all: the last parcel block once more without MAX_PARCEL, so orders
    # with more parcels than the matrix goes to still get a carrier.
    if blocks and not (open_end and blocks[-1]['n_to'] == last_n):
        n_last = len(blocks[-1]['items'])
        for r, sm in zip(rows_out[-n_last:], sim[-n_last:]):
            c = r.copy()
            c['MIN_PARCEL'] = blocks[-1]['n_to'] + 1      # starts where the block ends
            c['MAX_PARCEL'] = None
            rows_out.append(c)
            sim.append((sm[0], blocks[-1]['n_to'] + 1, INF) + sm[3:])

    # check: every equal-parcel order lands on the same carrier as the
    # cheapest row of the minimal matrix
    mismatch = 0
    for n in range(1, last_n + 1):
        m = (nlo <= n) & (nhi >= n)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        bps = np.concatenate([whi[idx], wlo[idx], each[idx] * n])
        bps = np.unique(bps[np.isfinite(bps) & (bps > 0)])
        tests = np.unique(np.concatenate([bps, bps + 1e-3, bps - 1e-3,
                                          np.linspace(0.001, bps.max() * 1.05, 200)]))
        for w in tests[tests > 0]:
            fit = idx[(wlo[idx] < w - EPS / 2) & (whi[idx] >= w - EPS / 2)
                      & (each[idx] * n >= w - EPS / 2)]
            if not len(fit):
                continue          # a gap, bridged when the same carrier is on both sides
            if _first_match(sim, n, w) != out[fit[np.argmin(price[fit])]]:
                mismatch += 1
    return rows_out, mismatch


def build_routing(df, blank_postcode=True):
    """Routing table for one country's minimal frame. Returns (frame, stats)."""
    if df is None or df.empty:
        return df, {'rows_in': 0, 'rows_out': 0, 'mismatch': 0}
    df = df.reset_index(drop=True)
    price = pd.to_numeric(df.get('TOTAL_PRICE'), errors='coerce')
    # Rows for particular orders only (a customer name in USER_DEF, or an
    # AWKWARD / HAZMAT flag) stay as they are. A number in USER_DEF (e.g. the
    # 1.5 m oversize limit stamped on UPSGB rows) is a limit of the carrier:
    # it travels with the row as part of the outcome.
    spec_cols = [c for c in SPECIAL_COLS if c in df.columns]
    flag_cols = [c for c in ('HAZMAT', 'AWKWARD') if c in df.columns]
    udf_cols = [c for c in spec_cols if c not in flag_cols]

    def _is_text(v):
        if _blank(v):
            return False
        try:
            float(str(v).replace(',', '.'))
            return False
        except ValueError:
            return True
    # Only a name in USER_DEF_TYPE_1 (a customer, e.g. Audio Partner) marks a
    # row for particular orders. Other text — UPDE's 'single' / 'multi' tariff
    # label in USER_DEF_TYPE_2 — travels with the row like the carrier does.
    cust_cols = [c for c in ('USER_DEF_TYPE_1',) if c in udf_cols]
    text_udf = (df[cust_cols].apply(lambda c: c.map(_is_text)).any(axis=1)
                if cust_cols else pd.Series(False, index=df.index))
    flagged = (df[flag_cols].apply(lambda c: ~c.map(_blank)).any(axis=1)
               if flag_cols else pd.Series(False, index=df.index))
    # customer / service-code rows stay as they are and go first; flagged rows
    # (oversize buckets: AWKWARD) get their own routing table below the normal
    # one, so an order only falls through to them when no normal row fits
    # Sentinel catch-alls (TOTAL_PRICE 999999, e.g. DHL up to 24000 kg: "only
    # if nothing else fits") go to the very bottom, below the oversize rows.
    sentinel = price >= 999999
    keep_as_is = df[price.isna()]
    customers = df[price.notna() & text_udf]
    normal = df[price.notna() & ~text_udf & ~flagged & ~sentinel]
    buckets = df[price.notna() & ~text_udf & flagged & ~sentinel]
    tail_rows = df[sentinel & ~text_udf].copy()
    for c in PRICE_COLS:
        if c in tail_rows.columns:
            tail_rows[c] = None

    per_norm, mm1, gr_norm = _route_frame(normal, OUTCOME_COLS + udf_cols)

    # A size limit in USER_DEF_TYPE_1 (UPSGB 1.5 m) sends a bigger parcel to
    # the oversize rows (AWKWARD) further down. At a postcode with such a
    # limit, the rows without one (the pallet) get the same limit, so a bigger
    # parcel falls through them too instead of stopping at the pallet.
    if len(buckets) and 'USER_DEF_TYPE_1' in normal.columns:
        def _stamp(f):
            lim = pd.to_numeric(f['USER_DEF_TYPE_1'].map(
                lambda v: str(v).replace(',', '.')), errors='coerce')
            if lim.notna().any():
                f = f.copy()
                top = f.loc[lim.idxmax(), 'USER_DEF_TYPE_1']
                f.loc[f['USER_DEF_TYPE_1'].map(_blank), 'USER_DEF_TYPE_1'] = top
            return f
        per_norm = {p: _stamp(f) for p, f in per_norm.items()}
    per_buck, mm2, gr_buck = _route_frame(buckets, OUTCOME_COLS + udf_cols + flag_cols)

    # The largest group of postcodes with identical rows is written once with
    # an empty POSTCODE (= the rest of the country), BELOW the postcodes that
    # differ (e.g. GB islands), so CargoWrite — reading top-down — gives those
    # their own rows first. A country where every postcode is the same ends up
    # as one set of rows without a postcode.
    def _split(per, groups):
        # standard matrix: rates differ per postcode, so every postcode keeps
        # its own rows (blank_postcode=False)
        if not groups or not blank_postcode:
            return per, None
        big = max(groups, key=len)
        rest = per[big[0]].assign(POSTCODE=None)
        return {p: f for p, f in per.items() if p not in set(big)}, rest
    per_norm, blank_norm = _split(per_norm, gr_norm)

    # customer rows (a name in USER_DEF_TYPE_1): the same layout, per customer,
    # each postcode keeping its own rows; they go first at their postcode
    per_cust, mm3 = {}, 0
    for cname, sub in (customers.groupby(customers['USER_DEF_TYPE_1'].map(str))
                       if len(customers) else []):
        pc_rows, mm, _ = _route_frame(sub, OUTCOME_COLS + udf_cols + flag_cols)
        mm3 += mm
        for p, f in pc_rows.items():
            per_cust[p] = pd.concat([per_cust[p], f]) if p in per_cust else f
    per_buck, blank_buck = _split(per_buck, gr_buck)

    key = keep_as_is['POSTCODE'].map(lambda v: '' if _blank(v) else str(v))
    parts = [keep_as_is[key == ''], per_cust.get('')]   # service / customer rows first
    for p in sorted((set(per_norm) | set(per_buck) | set(per_cust) | set(key)) - {''}):
        parts += [keep_as_is[key == p], per_cust.get(p), per_norm.get(p), per_buck.get(p)]
    parts += [per_norm.get(''), blank_norm, per_buck.get(''), blank_buck]
    parts.append(tail_rows)                             # sentinel catch-all: last
    res = pd.concat([x for x in parts if x is not None and len(x)], ignore_index=True)
    res = res.drop(columns=[c for c in ('_out', '_pc') if c in res.columns])
    return res, {'rows_in': len(df), 'rows_out': len(res), 'mismatch': mm1 + mm2 + mm3,
                 'postcode_groups': len(gr_norm) + len(gr_buck)}


def _route_frame(work, outcome_cols):
    """Route every postcode of `work`: {postcode: frame}, mismatches, groups."""
    if work is None or work.empty:
        return {}, 0, []
    work = work.copy()
    ocols = [c for c in outcome_cols if c in work.columns]
    okey = work[ocols].apply(lambda c: c.map(str)).agg('\x1f'.join, axis=1)
    # Carriers with separate tariffs by parcel size under one service level
    # (DPD small <= 3 kg / normal <= 31.5 kg): each tariff is its own option,
    # with its own per-parcel maximum.
    split = work['CARRIER_ID'].isin(TARIFF_BY_EACH_WEIGHT)
    if split.any():
        ew_cls = pd.to_numeric(work['EACH_WEIGHT'], errors='coerce').map(
            lambda v: '' if pd.isna(v) else f'{v:g}')
        okey = okey.where(~split, okey + '\x1f' + ew_cls)
    codes, _ = pd.factorize(okey)
    work['_out'] = codes
    work['_pc'] = work['POSTCODE'].map(lambda v: '' if _blank(v) else str(v))

    caps, vratio = {}, {}
    ew = pd.to_numeric(work.get('EACH_WEIGHT'), errors='coerce')
    mv = pd.to_numeric(work.get('MAX_VOLUME'), errors='coerce')
    mw = pd.to_numeric(work.get('MAX_WEIGHT'), errors='coerce')
    for o, g in work.groupby('_out'):
        car = g['CARRIER_ID'].iloc[0]
        e = ew[g.index].max()
        cap = CARRIER_EACH_CAP.get(car, e if pd.notna(e) else INF)
        caps[o] = cap
        ratio = (mv[g.index] / mw[g.index]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio):
            vratio[o] = float(ratio.median())

    sigcols = [c for c in work.columns if c not in ('POSTCODE', '_pc')]
    groups = {}
    for p, g in work.groupby('_pc', sort=True):
        h = pd.util.hash_pandas_object(g[sigcols].apply(lambda c: c.map(str)), index=False)
        groups.setdefault(tuple(sorted(h.tolist())), []).append(p)

    per_pc, mismatch = {}, 0
    for pcs in groups.values():
        g = work[work['_pc'] == pcs[0]].reset_index(drop=True)
        rows, mm = _route_group(g, g['_out'].to_numpy(), caps, vratio)
        mismatch += mm
        block = pd.DataFrame(rows)
        for p in pcs:
            per_pc[p] = block.assign(POSTCODE=(None if p == '' else p))
    return per_pc, mismatch, list(groups.values())
