import streamlit as st
import pickle
import glob
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from model import TabularMLP
import json


@st.cache_data
def load_client_datasets(path="client_datasets.pkl"):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_drift_reports(folder="drift_reports"):
    files = glob.glob(os.path.join(folder, "*.csv"))
    reports = []
    for p in files:
        try:
            df = pd.read_csv(p, index_col=0)
            # extract round and client from filename if possible
            fname = os.path.basename(p)
            parts = fname.replace(".csv", "").split("_")
            rnd = None
            cid = None
            for i, part in enumerate(parts):
                if part == "round" and i + 1 < len(parts):
                    rnd = int(parts[i + 1])
                if part == "client" and i + 1 < len(parts):
                    cid = int(parts[i + 1])
            reports.append({"path": p, "round": rnd, "client": cid, "df": df})
        except Exception:
            continue
    return reports


@st.cache_data
def list_models(folder="models"):
    files = glob.glob(os.path.join(folder, "*.pt"))
    models = []
    for p in files:
        try:
            fname = os.path.basename(p)
            models.append({"path": p, "name": fname})
        except Exception:
            continue
    models = sorted(models, key=lambda x: x["name"])
    return models


@st.cache_data
def load_metrics_history(path="models/metrics_history.json"):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for entry in data:
            rnd = entry.get("round")
            metrics = entry.get("metrics", {}) or {}
            loss = metrics.get("loss") if isinstance(metrics.get("loss"), (int, float)) else None
            acc = metrics.get("accuracy") if isinstance(metrics.get("accuracy"), (int, float)) else None
            if acc is None:
                acc = metrics.get("acc") if isinstance(metrics.get("acc"), (int, float)) else None
            auc = metrics.get("auc") if isinstance(metrics.get("auc"), (int, float)) else None
            rows.append({"round": rnd, "loss": loss, "accuracy": acc, "auc": auc})
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        if df[["loss","accuracy","auc"]].notna().sum(axis=1).sum() == 0:
            return None
        return df.sort_values("round").reset_index(drop=True)
    except Exception:
        return None


@st.cache_data
def build_global_scaler(client_datasets):
    # Concatenate aligned raw DataFrames to compute a global scaler
    frames = []
    for cid, ds in client_datasets.items():
        df = ds.get("X_train_raw")
        if isinstance(df, pd.DataFrame):
            frames.append(df)
    if not frames:
        return None, None
    G = pd.concat(frames, axis=0)
    scaler = StandardScaler()
    scaler.fit(G.values)
    col_means = G.mean(axis=0)
    return scaler, list(G.columns)


@st.cache_data
def compute_feature_explanations(client_datasets):
    """Return a DataFrame describing each feature and what 'small'/'big' mean."""
    frames = []
    for cid, ds in client_datasets.items():
        df = ds.get("X_train_raw")
        if isinstance(df, pd.DataFrame):
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["feature", "type", "explanation", "median"])

    G = pd.concat(frames, axis=0)
    rows = []

    # Common numeric feature descriptions for Adult dataset
    numeric_desc = {
        "age": "Age of the individual (years). 'Small' = younger than typical (below median), 'Big' = older than typical (at/above median).",
        "hours-per-week": "Hours worked per week. 'Small' = works fewer hours than typical, 'Big' = works more hours than typical.",
        "capital-gain": "Capital gains (dollars). 'Small' = little or no gains, 'Big' = higher capital gains.",
        "capital-loss": "Capital losses (dollars). 'Small' = little or no losses, 'Big' = higher capital losses.",
        "education-num": "Education level as an integer (years). 'Small' = fewer years of education, 'Big' = more years.",
        "fnlwgt": "Final weight (sampling weight). Used for census weighting — not directly interpretable as income. 'Small'/'Big' are relative to dataset median.",
    }

    for col in G.columns:
        vals = G[col].dropna()
        median = float(vals.median()) if not vals.empty else None

        # If exact match for known numeric
        if col in numeric_desc:
            explanation = numeric_desc[col]
            rows.append({"feature": col, "type": "numeric", "explanation": explanation, "median": median})
            continue

        # Detect one-hot encoded categorical like 'education_Bachelors' or 'sex_Male'
        if "_" in col:
            feat, cat = col.split("_", 1)
            feat_clean = feat.replace('-', ' ').replace('.', ' ')
            explanation = (
                f"Indicator for {feat_clean} == '{cat}'. Value 1 means the individual belongs to the category '{cat}' for '{feat_clean}', 0 means they do not."
            )
            rows.append({"feature": col, "type": "binary", "explanation": explanation, "median": median})
            continue

        # If column looks binary (0/1) despite not having underscore
        unique = np.unique(vals)
        is_binary = False
        try:
            if len(unique) <= 3 and set(np.round(unique).astype(int)).issubset({0, 1}):
                is_binary = True
        except Exception:
            is_binary = False

        if is_binary:
            explanation = "Binary indicator column. Value 1 means presence/True, 0 means absence/False."
            rows.append({"feature": col, "type": "binary", "explanation": explanation, "median": median})
            continue

        # Fallback: generic numeric explanation
        explanation = (
            f"Feature derived from original data column '{col}'. Median={median:.3f} across clients. 'Small' = below median, 'Big' = at/above median."
        )
        rows.append({"feature": col, "type": "numeric", "explanation": explanation, "median": median})

    return pd.DataFrame(rows)


@st.cache_data
def load_latest_model(model_folder="models"):
    models = list_models(model_folder)
    if not models:
        return None, None
    latest = models[-1]["path"]
    # try to infer input dim and load state_dict
    try:
        state = torch.load(latest, map_location="cpu")
        # create a model by inferring first Linear weight if possible
        first_shape = None
        for v in state.values():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                first_shape = v.shape
                break
        if first_shape is None:
            return latest, None
        input_dim = first_shape[1]
        model = TabularMLP(input_dim=input_dim)
        model.load_state_dict(state)
        model.eval()
        return latest, model
    except Exception:
        return latest, None


def predict_with_model(model, X_np):
    if model is None:
        return None
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_np.astype(np.float32))
        preds = model(xt).cpu().numpy()
    return preds


def main():
    st.set_page_config(page_title="FL Monitor & Predictor", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose page", ["Overview", "Clients", "Drift Monitor", "Models", "User Predictor"])

    client_datasets = load_client_datasets()
    drift_reports = load_drift_reports()
    models = list_models()
    latest_model_path, latest_model = load_latest_model()
    global_scaler, feature_columns = (None, None)
    if client_datasets:
        global_scaler, feature_columns = build_global_scaler(client_datasets)

    if page == "Overview":
        st.title("Federated Learning Overview")
        cols = st.columns(4)
        cols[0].metric("Clients", len(client_datasets) if client_datasets else 0)
        cols[1].metric("Saved models", len(models))
        cols[2].metric("Drift reports", len(drift_reports))
        cols[3].write("\n")

        st.markdown("#### Latest model")
        if latest_model_path:
            st.write(latest_model_path)
        else:
            st.info("No model saved in `models/` yet.")

        # Plot training/eval metrics vs round if available
        metrics_df = load_metrics_history()
        if metrics_df is not None:
            st.markdown("#### Training metrics vs round")
            plot_df = metrics_df.set_index('round')
            c1, c2 = st.columns(2)
            with c1:
                if 'loss' not in plot_df.columns or plot_df['loss'].dropna().empty:
                    st.info("No loss values available in metrics history.")
                else:
                    st.line_chart(plot_df['loss'])
                    st.caption("Loss vs round")
            with c2:
                if (('accuracy' not in plot_df.columns or plot_df['accuracy'].dropna().empty)
                        and ('auc' not in plot_df.columns or plot_df['auc'].dropna().empty)):
                    st.info("No accuracy/AUC values available in metrics history.")
                else:
                    if 'accuracy' in plot_df.columns and not plot_df['accuracy'].dropna().empty:
                        st.line_chart(plot_df['accuracy'])
                        st.caption("Accuracy vs round")
                    else:
                        st.line_chart(plot_df['auc'])
                        st.caption("AUC vs round")

    elif page == "Clients":
        st.title("Clients")
        if not client_datasets:
            st.info("No `client_datasets.pkl` found. Run `save_clients.py` first.")
            return

        cid = st.sidebar.selectbox("Select client", sorted(client_datasets.keys()))
        ds = client_datasets[cid]
        st.subheader(f"Client {cid} summary")
        st.write(f"Train size: {len(ds['y_train'])} | Test size: {len(ds['y_test'])}")

        with st.expander("Show first rows of raw training features"):
            st.dataframe(ds['X_train_raw'].head())

        st.markdown("#### Class balance (train)")
        try:
            vc = pd.Series(ds['y_train']).value_counts().sort_index()
            st.bar_chart(vc)
        except Exception:
            pass

    elif page == "Drift Monitor":
        st.title("Drift Monitor")
        if not drift_reports:
            st.info("No drift CSVs found in `drift_reports/`.")
            return

        # Build simple index of reports
        idx = pd.DataFrame([{"path": r['path'], "round": r['round'], "client": r['client']} for r in drift_reports])
        st.dataframe(idx)

        sel = st.selectbox("Select report", list(range(len(drift_reports))))
        report = drift_reports[sel]
        st.subheader(f"Drift report: round {report['round']} client {report['client']}")
        df = report['df']
        # style drift_flag column if present
        if 'drift_flag' in df.columns:
            def highlight_flag(val):
                color = 'red' if val else 'white'
                return f'background-color: {color}'
            st.dataframe(df.style.applymap(lambda v: 'background-color: #faa' if str(v) in ('True','1') else '', subset=['drift_flag']))
        else:
            st.dataframe(df)

        st.markdown("#### PSI per feature (bar)")
        if 'psi' in df.columns:
            st.bar_chart(df['psi'])

    elif page == "Models":
        st.title("Models")
        if not models:
            st.info("No models saved in `models/`.")
            return
        for m in models:
            st.write(m['name'])
        st.markdown("#### Latest model preview")
        if latest_model is None and latest_model_path:
            st.warning("Latest model found but failed to load as a PyTorch TabularMLP instance.")
        elif latest_model is not None:
            st.write(latest_model)

    elif page == "User Predictor":
        st.title("User Predictor — will predict >50k (Y/N)")
        st.markdown("Provide a row of raw features. You may upload a CSV with aligned columns or use the manual editor.")

        if feature_columns is None:
            st.info("No aligned feature columns available. Run `save_clients.py` to create `client_datasets.pkl`.")
            return

        mode = st.radio("Input mode", ["Upload CSV", "Manual entry (quick)"])

        user_df = None

        if mode == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV with feature columns matching aligned features", type=['csv'])
            if uploaded is not None:
                try:
                    df = pd.read_csv(uploaded)
                    # allow both raw aligned names or subset; reindex
                    row = df.iloc[0:1]
                    row_reindexed = row.reindex(columns=feature_columns, fill_value=0)
                    user_df = row_reindexed
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")

        else:
            st.markdown("Select a client to prefill defaults (means). You can then edit a small set of features.")
            pref_client = st.selectbox("Prefill from client (means)", [None] + sorted(client_datasets.keys()))
            defaults = pd.Series(0, index=feature_columns)
            if pref_client is not None:
                cdf = client_datasets[pref_client]['X_train_raw']
                defaults = cdf.mean(axis=0)

            # let user choose up to 12 features to edit
            chosen = st.multiselect("Choose features to edit (up to 12)", feature_columns, default=feature_columns[:8], max_selections=12)
            inputs = {}
            for f in chosen:
                v = st.number_input(f, value=float(defaults.get(f, 0.0)), format="%f")
                inputs[f] = v

            # build one-row df
            row = pd.DataFrame([defaults.values], columns=feature_columns)
            for k, v in inputs.items():
                row.loc[0, k] = v
            user_df = row

        if user_df is not None:
            st.markdown("#### Preview input (first row)")
            st.dataframe(user_df.head(1))

            # Feature explanations: show what each column means and what small/big imply
            if client_datasets:
                feat_exp_df = compute_feature_explanations(client_datasets)
            else:
                feat_exp_df = pd.DataFrame()

            with st.expander("Feature explanations (what each column means and small/big guidance)"):
                if feat_exp_df.empty:
                    st.write("No feature information available. Ensure `client_datasets.pkl` exists.")
                else:
                    # show a compact table
                    st.dataframe(feat_exp_df.rename(columns={"feature": "Column", "type": "Type", "explanation": "Explanation", "median": "Median"}))
                    sel_feat = st.selectbox("Show detail for feature", feat_exp_df['feature'].tolist())
                    detail = feat_exp_df[feat_exp_df['feature'] == sel_feat].iloc[0]
                    st.markdown(f"**{sel_feat}** — *{detail['type']}*")
                    st.write(detail['explanation'])
                    if pd.notna(detail['median']):
                        st.write(f"Median value across training data: {detail['median']:.4f}")

            if global_scaler is None:
                st.error("Global scaler not available; cannot normalize input for model.")
            else:
                X_scaled = global_scaler.transform(user_df.values)
                if latest_model is None:
                    st.error("No model available to make predictions. Save a model in `models/` first.")
                else:
                    preds = predict_with_model(latest_model, X_scaled)
                    prob = float(preds[0])
                    cls = "<=50k" if prob < 0.5 else ">50k"
                    st.metric("Predicted probability of >50k", f"{prob:.4f}", delta=None)
                    st.success(f"Predicted class: {cls}")


if __name__ == '__main__':
    main()
