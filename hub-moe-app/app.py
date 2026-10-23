"""Hub Model Adoption Advisor  -  run with:  streamlit run app.py"""
import datetime as dt
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                      # make data/ and artifacts/ paths work from any folder
sys.path.insert(0, str(HERE))       # make the hubmoe package importable

st.set_page_config(page_title="Hub Model Adoption Advisor", page_icon="🧠", layout="wide")


@st.cache_data
def load():
    return pd.read_csv("data/models.csv"), pd.read_csv("data/models_by_month.csv")


@st.cache_resource
def get_predictor():
    from hubmoe.predictor import Predictor
    return Predictor("artifacts")


models, monthly = load()
st.title("🧠 Hub Model Adoption Advisor")
st.caption("Mixture-of-Experts model trained on 12,000 Hugging Face models (snapshot 2026-10-11).")

tab_over, tab_pred, tab_find, tab_about = st.tabs(["📊 Overview", "🔮 Predict", "🔎 Model finder", "ℹ️ About"])

# ---------------------------------------------------------------- Overview
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models", f"{len(models):,}")
    c2.metric("Monthly downloads", f"{models.downloads.sum()/1e9:.2f} B")
    c3.metric("Top-100 share", f"{models.downloads.nlargest(100).sum()/models.downloads.sum():.0%}")
    c4.metric("Median downloads", f"{int(models.downloads.median()):,}")
    l, r = st.columns(2)
    with l:
        st.subheader("Downloads by task (M)")
        st.bar_chart(models.groupby("pipeline_tag").downloads.sum().nlargest(12) / 1e6)
    with r:
        st.subheader("Downloads by library (M)")
        st.bar_chart(models.groupby("library").downloads.sum().nlargest(12) / 1e6)
    st.subheader("New models per month")
    st.line_chart(monthly.iloc[1:].set_index("month"))
    st.caption("First month (2022-03) excluded: the dataset floors all older creation dates to 2022-03-02.")

# ---------------------------------------------------------------- Predict
with tab_pred:
    art_ok = Path("artifacts/moe.pt").exists() and Path("artifacts/meta.json").exists()
    if not art_ok:
        st.error("No trained model found. Run `python train.py` first.")
    else:
        P = get_predictor()
        tasks = [t for t in P.vocab["pipeline_tag"] if t != "__other__"]
        libs = [t for t in P.vocab["library"] if t != "__other__"]
        authors = P.vocab["author"]
        with st.form("pred"):
            a, b, c = st.columns(3)
            task = a.selectbox("Task (pipeline_tag)", tasks, index=tasks.index("text-generation") if "text-generation" in tasks else 0)
            lib = b.selectbox("Library", libs, index=libs.index("transformers") if "transformers" in libs else 0)
            author = c.selectbox("Author / org (__other__ = small or unlisted)", authors)
            d, e = st.columns(2)
            name = d.text_input("Model name", "MyModel-7B-Instruct")
            created = e.date_input("Creation date", value=dt.date(2026, 6, 1),
                                   min_value=dt.date(2022, 3, 2), max_value=dt.date(2026, 10, 11))
            go = st.form_submit_button("Predict adoption")
        if go:
            res = P.predict(task, lib, author, name, created)
            pct = (models.downloads < res["downloads"]).mean()
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted downloads / month", f"{res['downloads']:,.0f}")
            m2.metric("90% interval", f"{res['low']:,.0f} – {res['high']:,.0f}")
            m3.metric("Percentile among Top-12K", f"{pct:.0%}")
            st.subheader("Which experts handled this model?")
            st.bar_chart(pd.Series(res["expert_weights"], index=[f"Expert {i}" for i in range(len(res["expert_weights"]))]))
            st.warning("The training set only contains models that already reached the Top-12K "
                       "(≥ 4,791 downloads/month). Treat the number as 'expected downloads **if** the "
                       "model makes it into the Top-12K', not as an absolute forecast.")

# ---------------------------------------------------------------- Finder
with tab_find:
    f1, f2, f3 = st.columns(3)
    t = f1.multiselect("Task", sorted(models.pipeline_tag.unique()))
    lb = f2.multiselect("Library", sorted(models.library.unique()))
    q = f3.text_input("Name contains")
    mind = st.slider("Min downloads / month", 0, 1_000_000, 10_000, step=5_000)
    v = models[models.downloads >= mind]
    if t: v = v[v.pipeline_tag.isin(t)]
    if lb: v = v[v.library.isin(lb)]
    if q: v = v[v.model_id.str.contains(q, case=False, regex=False)]
    st.write(f"{len(v):,} models")
    st.dataframe(v.sort_values("downloads", ascending=False)
                  [["model_id", "pipeline_tag", "library", "downloads", "likes", "created_at"]],
                 use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- About
with tab_about:
    st.markdown("### Model")
    st.markdown("Sparse **Mixture of Experts**: embeddings → shared trunk → gate → top-k of N expert MLPs → regression head. "
                "Target = log10(monthly downloads).")
    if Path("artifacts/meta.json").exists():
        import json
        st.json(json.loads(Path("artifacts/meta.json").read_text())["metrics"])
        rt = Path("artifacts/routing_by_task.csv")
        if rt.exists():
            st.markdown("**Expert routing by task** (share of models of each task sent to each expert)")
            st.dataframe(pd.read_csv(rt).set_index("pipeline_tag"))
    st.markdown("### Known dataset limitations\n"
                "- `license` is `unknown` for all 12,000 rows → license analysis is impossible.\n"
                "- `created_at` is floored at 2022-03-02 (915 rows).\n"
                "- Only the top-12K by downloads: selection bias.\n"
                "- Metadata only: no text/images/time-series to learn from.")
