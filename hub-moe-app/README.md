# Hub Model Adoption Advisor (Mixture of Experts)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python eda.py            # dataset analysis
python train.py          # trains MoE + baseline -> artifacts/
streamlit run app.py     # http://localhost:8501
```
Docker: `docker build -t hub-moe . && docker run -p 8501:8501 hub-moe`
(run `python train.py` first so `artifacts/` is copied in).
