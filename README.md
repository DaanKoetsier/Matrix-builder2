# Matrix Shrinker (Rate Matrix Builder 2)

A small Streamlit app that **shrinks the matrix the customer sends back**.

## The workflow

1. **Rate Matrix Builder 1** builds the full matrix with our prices.
2. The **customer** applies their own factor, sorts the matrix on *their* price
   (cheapest first) and sends it back **without prices**.
3. **This app** shrinks it: per country only the rows that decide something
   remain. The result has the customer's own columns and can be sent back as is.

## How to use it

1. Open the app.
2. Upload the customer's combined matrix (all countries) — `.xlsx` or `.xlsb`.
3. Click **✂️ Shrink**. A full matrix (~740,000 rows, 29 countries) takes about
   6 minutes; you see the progress per country.
4. Download the shrunk file. The table shows per country the rows in / out and
   the check.

## How it works

CargoWrite takes the **first row an order fits**. The customer's file is sorted on
their price, so the row order *is* their price. Every row gets its position as
price, and `routing.py` builds per country a routing table: the number of parcels
goes up, and per count the rows list which carrier is chosen as the weight goes
up. Only the rows where the choice changes remain; every order still lands on the
same carrier as in the customer's full list.

- Rows picked by name — `PRIO` and customer names in `USER_DEF_TYPE_1`, and the
  fixed service-code rows — are kept unchanged (yellow). The PRIO rows go to the
  end of the file, in the customer's order.
- The DHL catch-all (DHL-FENDER, no postcode, 24,000 kg) stays the last row of
  each country.
- Both column layouts are read (`USER_DEF_TYPE_4` / `USER_DEF_NUM_1`); cell
  formulas in `.xlsx` (e.g. `MAX_WEIGHT = MAX_PARCEL × EACH_WEIGHT`) are evaluated.

## Files

| File | What it does |
|------|--------------|
| `app.py` | The Streamlit page: upload, shrink, download. |
| `customer_matrix.py` | Reads the customer's file, shrinks it per country, writes it back. |
| `routing.py` | Builds the routing table of one country. |

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

The versions in `requirements.txt` are pinned to what runs on Streamlit Cloud
(Python 3.14). Update them deliberately and test first.
