"""
compact.py — shrink a finished matrix to its switch points.

CargoWrite picks, for an order, the cheapest row whose limits the order fits.
Only the CARRIER (and service level / user-defined columns) that this choice
lands on matters; this module removes and merges rows so that every possible
order still lands on exactly the same carrier as in the full matrix, and every
order that had a rate still has one (and none that had none gets one).

Each row is treated as a box in five dimensions, every one an interval
(lo, hi] with a blank lower limit meaning 0 and a blank upper limit meaning
"no limit":

    parcels        (MIN_PARCEL-1, MAX_PARCEL]
    total weight   (MIN_WEIGHT,   MAX_WEIGHT]
    each weight    (0,            EACH_WEIGHT]
    total volume   (MIN_VOLUME,   MAX_VOLUME]
    each volume    (0,            EACH_VOLUME]

Two operations, both checked exactly with box geometry (no sampling):

  * drop a row that never wins anywhere (every point of it is covered by a
    cheaper row);
  * merge two rows with the same outcome into their bounding box at a price
    that keeps every other carrier's winning area unchanged.

The price on a merged row is one of the two original prices where possible,
so TOTAL_PRICE stays a real rate of that carrier, but it is no longer the
exact price for every order the row covers — only the carrier choice is kept.
"""
import numpy as np
import pandas as pd

INF = np.inf
DIMS = 5

# Columns that decide WHAT an order gets (kept identical) — everything that is
# neither a limit, a price nor the postcode.
OUTCOME_COLS = ['CARRIER_ID', 'SERVICE_LEVEL', 'USER_DEF_TYPE_1', 'USER_DEF_TYPE_2',
                'USER_DEF_TYPE_3', 'USER_DEF_NUM_1', 'HAZMAT', 'AWKWARD']
# Real per-parcel maximum of the UPS rate tables (pipeline.UPS_MAX_PARCEL_KG).
CARRIER_EACH_CAP = {'UPDE': 70.0, 'UPSNL': 70.0, 'UPSGB': 70.0}

LIMIT_COLS = ['MIN_PARCEL', 'MAX_PARCEL', 'MIN_WEIGHT', 'MAX_WEIGHT', 'EACH_WEIGHT',
              'MIN_VOLUME', 'MAX_VOLUME', 'EACH_VOLUME']


def _num(v, default):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(f) else f


def _row_box(r):
    lo = np.array([max(_num(r.get('MIN_PARCEL'), 1) - 1, 0),
                   _num(r.get('MIN_WEIGHT'), 0), 0.0,
                   _num(r.get('MIN_VOLUME'), 0), 0.0])
    hi = np.array([_num(r.get('MAX_PARCEL'), INF), _num(r.get('MAX_WEIGHT'), INF),
                   _num(r.get('EACH_WEIGHT'), INF), _num(r.get('MAX_VOLUME'), INF),
                   _num(r.get('EACH_VOLUME'), INF)])
    return lo, hi


# Equal-parcels mode: an order is n parcels of the same weight (heaviest =
# total / n); volume is ignored. Set by compact_frame(equal_parcels=True).
_MODE = {'equal': False, 'nmax': 40}


def _physical(lo, hi):
    if _MODE['equal']:
        n_lo = int(np.floor(lo[0])) + 1
        n_hi = int(min(hi[0], _MODE['nmax']))
        if n_hi < max(n_lo, 1) or not lo[1] < hi[1]:
            return False
        ns = np.arange(max(n_lo, 1), n_hi + 1, dtype=float)
        return bool(np.any(np.maximum(lo[1], ns * lo[2]) < np.minimum(hi[1], ns * hi[2])))
    return _physical_real(lo, hi)


def _physical_real(lo, hi):
    """Does box (lo, hi] hold at least one real order? A real order has n >= 1
    parcels whose heaviest parcel weighs at least the average and at most the
    total (same for volume): total/n <= each <= total."""
    if not np.all(lo < hi) or hi[0] < 1:
        return False
    n = hi[0]
    for t, e in ((1, 2), (3, 4)):
        if not max(lo[e], lo[t] / n) < min(hi[e], hi[t]):
            return False
    return True


def _covered(alo, ahi, LO, HI):
    """True when every real order in box (alo, ahi] is covered by the union of
    boxes LO/HI (see _physical)."""
    if not _physical(alo, ahi):
        return True                                   # no real order in it
    if len(LO) == 0:
        return False
    hit = np.all((LO < ahi) & (HI > alo), axis=1)
    LO, HI = LO[hit], HI[hit]
    if len(LO) == 0:
        return False
    stack = [(alo, ahi)]
    budget = 200000
    while stack:
        plo, phi = stack.pop()
        if not _physical(plo, phi):
            continue
        budget -= 1
        if budget < 0:
            return False                              # give up = "not covered" (safe)
        inter = np.all((LO < phi) & (HI > plo), axis=1)
        idx = np.flatnonzero(inter)
        if len(idx) == 0:
            return False
        full = idx[np.all((LO[idx] <= plo) & (HI[idx] >= phi), axis=1)]
        if len(full):
            continue
        # subtract the box that overlaps the piece most (log-volume heuristic)
        olo = np.maximum(LO[idx], plo); ohi = np.minimum(HI[idx], phi)
        span = np.where(np.isinf(ohi - olo), 1e9, ohi - olo)
        b = idx[np.argmax(np.sum(np.log(np.maximum(span, 1e-12)), axis=1))]
        blo, bhi = LO[b], HI[b]
        clo, chi = plo.copy(), phi.copy()
        for d in range(DIMS):
            if blo[d] > clo[d]:
                nlo, nhi = clo.copy(), chi.copy(); nhi[d] = blo[d]
                stack.append((nlo, nhi)); clo[d] = blo[d]
            if bhi[d] < chi[d]:
                nlo, nhi = clo.copy(), chi.copy(); nlo[d] = bhi[d]
                stack.append((nlo, nhi)); chi[d] = bhi[d]
    return True


class _Group:
    """The rows of one postcode (all rows CargoWrite compares for one order)."""

    def __init__(self, lo, hi, price, outcome, fill_gaps=False):
        self.fill_gaps = fill_gaps
        self.lo, self.hi = lo, hi
        self.price = price
        self.out = outcome
        self.alive = np.ones(len(price), bool)
        self.src = [[i] for i in range(len(price))]   # original rows behind each row
        self.orig_price = price.copy()

    def _act(self):
        return np.flatnonzero(self.alive)

    def wins_in(self, t, alo, ahi):
        """Does row t win anywhere inside box A?"""
        lo = np.maximum(self.lo[t], alo); hi = np.minimum(self.hi[t], ahi)
        if not _physical(lo, hi):
            return False
        a = self._act()
        cheaper = a[(self.price[a] < self.price[t]) & (a != t)]
        return not _covered(lo, hi, self.lo[cheaper], self.hi[cheaper])

    def drop_dead(self):
        changed = True
        while changed:
            changed = False
            for t in sorted(self._act(), key=lambda i: -self.price[i]):
                if not self.wins_in(t, self.lo[t], self.hi[t]):
                    self.alive[t] = False
                    changed = True

    def try_merge(self, r, s):
        mlo = np.minimum(self.lo[r], self.lo[s]); mhi = np.maximum(self.hi[r], self.hi[s])
        a = self._act()
        # 1. merged box may not reach orders that currently get no rate
        if not self.fill_gaps and not _covered(mlo, mhi, self.lo[a], self.hi[a]):
            return False
        other = a[self.out[a] != self.out[r]]
        # L: the dearest other carrier that wins somewhere inside the box — the
        #    merged row must stay dearer than it there.
        inter_m = other[np.all((self.lo[other] < mhi) & (self.hi[other] > mlo), axis=1)]
        L = -INF
        for t in inter_m[np.argsort(-self.price[inter_m])]:
            if self.wins_in(t, mlo, mhi):
                L = self.price[t]; break
        # U: the cheapest other carrier that reaches a spot r or s wins today —
        #    the merged row must stay cheaper than it.
        U = INF
        for t in other[np.argsort(self.price[other])]:
            for q in (r, s):
                ilo = np.maximum(self.lo[t], self.lo[q]); ihi = np.minimum(self.hi[t], self.hi[q])
                if not _physical(ilo, ihi):
                    continue
                cheaper_q = a[self.price[a] < self.price[q]]
                if not _covered(ilo, ihi, self.lo[cheaper_q], self.hi[cheaper_q]):
                    U = self.price[t]; break
            if U < INF:
                break
        if not L < U:
            return False
        # The merged row must carry a real rate of one of its rows: the Excel
        # writer rebuilds TOTAL_PRICE from that row's surcharge columns.
        real = [self.orig_price[i] for i in self.src[r] + self.src[s]
                if L < self.orig_price[i] < U]
        if not real:
            return False
        p = min(real)
        self.lo[r], self.hi[r], self.price[r] = mlo, mhi, p
        self.alive[s] = False
        self.src[r] = self.src[r] + self.src[s]
        return True

    def compact(self, rounds=8, all_pairs_below=40):
        self.drop_dead()
        # Chain merges between neighbours, alternating the sort order so rows
        # merge first along the weight axis, then across parcel counts.
        keys = [lambda i: (self.hi[i][0], self.hi[i][1], self.hi[i][2]),
                lambda i: (self.hi[i][1], self.hi[i][0], self.hi[i][2])]
        for rnd in range(rounds):
            changed = False
            for o in np.unique(self.out[self._act()]):
                rows = sorted((i for i in self._act() if self.out[i] == o), key=keys[rnd % 2])
                if len(rows) <= all_pairs_below:
                    for x in range(len(rows)):
                        for y in range(x + 1, len(rows)):
                            r, s = rows[x], rows[y]
                            if self.alive[r] and self.alive[s] and self.try_merge(r, s):
                                changed = True
                    continue
                acc = rows[0]
                for s in rows[1:]:
                    if self.try_merge(acc, s):
                        changed = True
                    else:
                        acc = s
            self.drop_dead()
            if not changed and rnd % 2 == 1:
                break


def _winner(group_lo, group_hi, price, out, pts):
    """Outcome of the cheapest row containing each point (-1 = no rate)."""
    res = np.full(len(pts), -1)
    best = np.full(len(pts), INF)
    for k in range(len(price)):
        inside = np.all((pts > group_lo[k]) & (pts <= group_hi[k]), axis=1)
        better = inside & (price[k] < best)
        best[better] = price[k]; res[better] = out[k]
    return res


def _equal_points(lo, hi, n_pts=20000, seed=0):
    """Equal-parcel test orders: random ones plus every weight limit (and a
    hair either side) for every parcel count."""
    rng = np.random.default_rng(seed)
    cap = np.where(np.isinf(hi), np.where(np.isinf(lo), 1.0, lo * 2 + 1), hi)
    wmax = cap[:, 1].max() * 1.1
    n = rng.integers(1, _MODE['nmax'] + 1, n_pts).astype(float)
    w = rng.random(n_pts) * wmax
    limits = np.unique(np.concatenate([hi[:, 1][np.isfinite(hi[:, 1])], lo[:, 1]]))
    each = np.unique(hi[:, 2][np.isfinite(hi[:, 2])])
    ns, ws = [n], [w]
    for n_ in range(1, _MODE['nmax'] + 1):
        cand = np.unique(np.concatenate([limits, each * n_]))
        cand = cand[cand > 0]
        for d in (0.0, 1e-6, -1e-6):
            ws.append(cand + d); ns.append(np.full(len(cand), float(n_)))
    n = np.concatenate(ns); w = np.concatenate(ws)
    pts = np.zeros((len(n), DIMS))
    pts[:, 0] = n; pts[:, 1] = w; pts[:, 2] = w / n
    return pts


def _check(full, comp, out_full, out_comp, n_pts=20000, seed=0, fill_gaps=False):
    """Spot check: random points + every row corner of the full matrix."""
    lo, hi, pr = full
    clo, chi, cpr = comp
    rng = np.random.default_rng(seed)
    cap = np.where(np.isinf(hi), np.where(np.isinf(lo), 1.0, lo * 2 + 1), hi)
    capmax = cap.max(axis=0) * 1.1
    pts = rng.random((n_pts, DIMS)) * capmax
    pts[:, 0] = np.ceil(pts[:, 0])
    corners = np.vstack([np.where(np.isinf(hi), capmax, hi),
                         np.where(np.isinf(hi), capmax, hi) - 1e-6,
                         lo + 1e-6])
    corners[:, 0] = np.ceil(corners[:, 0])
    pts = np.vstack([pts, corners])
    if _MODE['equal']:
        n = rng.integers(1, _MODE['nmax'] + 1, len(pts)).astype(float)
        pts[:, 0] = n; pts[:, 2] = pts[:, 1] / n; pts[:, 3:] = 0
        extra = []
        for k in range(len(lo)):
            for n_ in range(1, _MODE['nmax'] + 1):
                for w in (hi[k, 1], hi[k, 2] * n_, lo[k, 1]):
                    if np.isfinite(w) and w > 0:
                        for ww in (w, w + 1e-6, w - 1e-6):
                            extra.append([n_, ww, ww / n_, 0, 0])
        pts = np.vstack([pts, np.array(extra)]) if extra else pts
        a = _winner(lo, hi, pr, out_full, pts)
        b = _winner(clo, chi, cpr, out_comp, pts)
        diff = a != b
        if fill_gaps:
            diff &= a != -1
        return int(np.sum(diff))
    # only real orders: total/n <= each <= total (weight and volume)
    n = np.maximum(pts[:, 0], 1); pts[:, 0] = n
    for t, e in ((1, 2), (3, 4)):
        pts[:, e] = pts[:, t] / n + rng.random(len(pts)) * (pts[:, t] - pts[:, t] / n)
    # plus points exactly on the each-weight limits of the full matrix
    extra = []
    for k in range(len(lo)):
        if np.isfinite(hi[k, 2]) and np.isfinite(hi[k, 1]):
            for n_ in range(1, int(min(hi[k, 0], 60)) + 1):
                w = min(hi[k, 1], hi[k, 2] * n_)
                if w / n_ <= hi[k, 2] <= w:
                    v = min(hi[k, 3], w / 167) if np.isfinite(hi[k, 3]) else w / 167
                    extra.append([n_, w, hi[k, 2], v, v / n_])
    if extra:
        pts = np.vstack([pts, np.array(extra)])
    a = _winner(lo, hi, pr, out_full, pts)
    b = _winner(clo, chi, cpr, out_comp, pts)
    diff = a != b
    if fill_gaps:
        diff &= a != -1          # orders without a rate before may now get one
    return int(np.sum(diff))


def _as_str(frame):
    return frame.apply(lambda col: col.map(str))


def compact_frame(df, verify=True, fill_gaps=False, equal_parcels=False):
    """Compact one country's matrix frame. Returns (compacted_df, stats)."""
    if df is None or df.empty:
        return df, {'rows_in': 0, 'rows_out': 0, 'mismatch': 0}
    df = df.reset_index(drop=True)
    price = pd.to_numeric(df.get('TOTAL_PRICE'), errors='coerce')
    fixed = df[price.isna()]                          # service-code rows: untouched
    work = df[price.notna()].copy()
    work['_price'] = price[price.notna()]
    ocols = [c for c in OUTCOME_COLS if c in work.columns]
    okey = _as_str(work[ocols]).agg('\x1f'.join, axis=1) if ocols else pd.Series('', index=work.index)
    codes, uniq = pd.factorize(okey)
    work['_out'] = codes
    pc = work['POSTCODE'].fillna('').astype(str) if 'POSTCODE' in work else pd.Series('', index=work.index)
    work['_pc'] = pc

    # each carrier's highest per-parcel weight anywhere in this country
    ew = pd.to_numeric(work.get('EACH_WEIGHT'), errors='coerce')
    country_each_cap = ew.groupby(work['CARRIER_ID']).max().dropna().to_dict()

    # postcodes with identical row sets are compacted once
    sigcols = [c for c in work.columns if c not in ('POSTCODE', '_pc')]
    sig = {}
    for p, g in work.groupby('_pc', sort=False):
        h = pd.util.hash_pandas_object(_as_str(g[sigcols]), index=False)
        sig.setdefault(tuple(sorted(h.tolist())), []).append(p)

    out_parts, mismatches = [], 0
    for pcs in sig.values():
        g = work[work['_pc'] == pcs[0]].reset_index(drop=True)
        boxes = [_row_box(r) for r in g.to_dict('records')]
        lo = np.array([b[0] for b in boxes]); hi = np.array([b[1] for b in boxes])
        _MODE['equal'] = equal_parcels
        if equal_parcels:
            lo[:, 3:] = 0.0; hi[:, 3:] = INF          # volume ignored
            # snap rounding noise (70.000002 kg, 5.000004 kg) to whole grams so
            # slivers like 70 → 70.000002 kg don't need rows of their own;
            # per-parcel limits are rounded UP so equal parcels still fit
            lo[:, 1] = np.round(lo[:, 1], 3)
            hi[:, 1] = np.where(np.isfinite(hi[:, 1]), np.round(hi[:, 1], 3), INF)
            hi[:, 2] = np.where(np.isfinite(hi[:, 2]), np.ceil(hi[:, 2] * 1e4 - 1e-6) / 1e4, INF)
            finite = hi[:, 0][np.isfinite(hi[:, 0])]
            _MODE['nmax'] = int(finite.max()) + 1 if len(finite) else 1
        pr = g['_price'].to_numpy(float); oc = g['_out'].to_numpy()
        G = _Group(lo.copy(), hi.copy(), pr.copy(), oc.copy(), fill_gaps)
        G.compact()
        keep = G._act()
        if equal_parcels:
            # EACH_WEIGHT becomes the carrier's real per-parcel maximum (how
            # the weight is spread over the parcels doesn't matter, only that
            # no parcel exceeds what the carrier takes). Kept only where the
            # check shows every equal-parcel order keeps its carrier.
            cap = {}
            for i in range(len(oc)):
                car = g['CARRIER_ID'].iloc[i]
                c_max = CARRIER_EACH_CAP.get(car, country_each_cap.get(car, 0))
                if np.isfinite(hi[i, 2]) and c_max:
                    cap[oc[i]] = c_max
            pts = _equal_points(lo, hi)
            ref = _winner(lo, hi, pr, oc, pts)

            def _diff():
                b = _winner(G.lo[keep], G.hi[keep], G.price[keep], oc[keep], pts)
                d = ref != b
                if fill_gaps:
                    d &= ref != -1
                return int(np.sum(d))
            base = _diff()
            for k in sorted(keep, key=lambda i: G.price[i]):
                if oc[k] in cap and G.hi[k][2] < cap[oc[k]]:
                    old_e = G.hi[k][2]
                    G.hi[k][2] = cap[oc[k]]
                    if _diff() > base:
                        G.hi[k][2] = old_e
        if verify and equal_parcels:
            mismatches += _diff()
        elif verify:
            mismatches += _check((lo, hi, pr), (G.lo[keep], G.hi[keep], G.price[keep]), oc, oc[keep],
                                 fill_gaps=fill_gaps)
        rows = []
        for k in keep:
            r = g.iloc[k].copy()
            l, h = G.lo[k], G.hi[k]
            r['MIN_PARCEL'] = None if l[0] <= 0 else int(l[0]) + 1
            r['MAX_PARCEL'] = None if np.isinf(h[0]) else int(h[0])
            r['MIN_WEIGHT'] = None if l[1] <= 0 else l[1]
            r['MAX_WEIGHT'] = None if np.isinf(h[1]) else h[1]
            r['EACH_WEIGHT'] = None if np.isinf(h[2]) else h[2]
            if equal_parcels:          # volume not modelled: widest of the merged rows
                vol = g.iloc[G.src[k]]
                for c in ('MIN_VOLUME', 'MAX_VOLUME', 'EACH_VOLUME'):
                    if c in vol:
                        v = pd.to_numeric(vol[c], errors='coerce')
                        r[c] = None if v.isna().any() else (v.min() if c == 'MIN_VOLUME' else v.max())
            else:
                r['MIN_VOLUME'] = None if l[3] <= 0 else l[3]
                r['MAX_VOLUME'] = None if np.isinf(h[3]) else h[3]
                r['EACH_VOLUME'] = None if np.isinf(h[4]) else h[4]
            if len(G.src[k]) > 1 or G.price[k] != pr[k]:
                src = g.iloc[[i for i in G.src[k] if pr[i] == G.price[k]][0]]
                for c in ('RATE_BASE', 'RATE_BASE2.0', 'RATE_EXTRA', 'FUEL', 'MAUT',
                          'Linehaul UPSDE', 'TOTAL_PRICE'):
                    if c in r.index:
                        r[c] = src[c]
            rows.append(r)
        comp = pd.DataFrame(rows)
        rep = pd.concat([comp.assign(POSTCODE=p) for p in pcs], ignore_index=True)
        out_parts.append(rep)
    res = pd.concat(out_parts + [fixed], ignore_index=True)
    res = res.drop(columns=[c for c in ('_price', '_out', '_pc') if c in res.columns])
    return res, {'rows_in': len(df), 'rows_out': len(res), 'mismatch': mismatches,
                 'postcode_groups': len(sig)}
