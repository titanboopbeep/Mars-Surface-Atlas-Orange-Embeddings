import io
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Mars Surface Atlas", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{color-scheme:light!important;--ink:#24241f;--muted:#625f57;--paper:#f6f4ef;--panel:#fffefa;--line:#ded9cf;--rust:#a94f37}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;color:var(--ink)!important;color-scheme:light!important}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stMainBlockContainer"]{background:var(--paper)!important;color:var(--ink)!important}
[data-testid="stHeader"],[data-testid="stToolbar"]{background:var(--paper)!important}
[data-testid="stSidebar"]{background:#efede6!important;border-right:1px solid var(--line)!important}
[data-testid="stSidebar"] *{color:var(--ink)!important}
h1,h2,h3,h4,p,li,label,[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] *,
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] *,[data-testid="stCaptionContainer"] *,
[data-testid="stMetricLabel"],[data-testid="stMetricValue"]{color:var(--ink)!important}
[data-testid="stCaptionContainer"] *{color:var(--muted)!important}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em}
.block-container{max-width:1260px;padding-top:2.8rem!important;padding-bottom:3rem}
div[data-testid="stMetric"]{background:var(--panel)!important;border:1px solid var(--line)!important;padding:16px 18px!important;border-radius:12px!important}
div[data-testid="stMetric"] *{color:var(--ink)!important}
[data-testid="stFileUploader"]{background:var(--panel)!important;border:1px dashed #bdb6a9!important;border-radius:12px!important;padding:10px!important}
[data-testid="stFileUploader"] *{color:var(--ink)!important}
[data-testid="stFileUploader"] section{background:#f1efe9!important;border:1px solid #d5cfc4!important;border-radius:8px!important}
[data-testid="stFileUploader"] button,[data-testid="stFileUploader"] button *{background:#fffefa!important;color:var(--ink)!important;border:1px solid #cfc8bc!important;opacity:1!important}
.stButton>button,.stDownloadButton>button{color:var(--ink)!important;background:#fffefa!important;border:1px solid #cfc8bc!important;border-radius:8px!important;font-weight:600!important;min-height:2.7rem!important}
.stButton>button[kind="primary"]{color:#fff!important;background:var(--rust)!important;border-color:var(--rust)!important}
.stButton>button[kind="primary"]:hover{background:#8f402d!important;color:#fff!important}
input,textarea,[data-baseweb="select"]>div,[data-baseweb="input"]>div{background:#fffefa!important;color:var(--ink)!important;border-color:#cfc8bc!important}
[data-baseweb="select"] *,[data-baseweb="input"] *{color:var(--ink)!important}
[data-baseweb="popover"],[role="listbox"],[role="option"]{background:#fffefa!important;color:var(--ink)!important}
[data-testid="stExpander"]{background:#fffefa!important;border:1px solid var(--line)!important;border-radius:10px!important}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{color:var(--ink)!important;background:#f0eee8!important}
[data-testid="stAlert"],[data-testid="stAlert"]>div,[data-testid="stAlert"] *{background:#eeece5!important;color:var(--ink)!important;border-color:var(--line)!important}
[data-testid="stDataFrame"],[data-testid="stTable"]{background:#fffefa!important;color:var(--ink)!important}
hr{border-color:var(--line)!important}
</style>""", unsafe_allow_html=True)

TARGET="category"
LABELS=["rocky","smooth"]

def read_orange_csv(file):
    raw=file.getvalue()
    try:
        df=pd.read_csv(io.BytesIO(raw),header=0,skiprows=[1,2])
        if TARGET in df.columns and any(isinstance(c,str) and c.startswith("n") and c[1:].isdigit() for c in df.columns):
            return df,"Orange CSV (metadata rows)"
    except Exception:
        pass
    return pd.read_csv(io.BytesIO(raw)),"Standard CSV"

def feature_columns(df):
    cols=[c for c in df.columns if isinstance(c,str) and c.startswith("n") and c[1:].isdigit()]
    cols.sort(key=lambda c:int(c[1:]))
    if not cols:
        raise ValueError("No n0...n999 embedding columns found. Export the Image Embedding (SqueezeNet) output from Orange.")
    return cols

def get_features(df,cols):
    missing=[c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} required embedding columns (examples: {missing[:6]}). Use the same Orange SqueezeNet workflow.")
    values=df[cols].apply(pd.to_numeric,errors="coerce")
    arr=values.to_numpy(dtype=float)
    bad=values.isna().any(axis=1) | ~np.isfinite(arr).all(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows have missing or invalid embedding values. Re-export or remove those rows in Orange.")
    return arr.astype(np.float32)

def image_names(df):
    keys={"image","images","filename","file name","filepath","file path","path","file","name","source","image name","image path"}
    for col in df.columns:
        if str(col).strip().lower() in keys:
            vals=df[col].fillna("").astype(str).str.strip().tolist()
            if any(vals): return vals
    return [f"CSV row {i+1}" for i in range(len(df))]

def train_model(df,test_size):
    if TARGET not in df.columns:
        raise ValueError("Training CSV must contain a 'category' column with rocky/smooth labels.")
    cols=feature_columns(df)
    data=df.dropna(subset=[TARGET]).copy()
    data[TARGET]=data[TARGET].astype(str).str.strip().str.lower()
    data=data[data[TARGET].isin(LABELS)].copy()
    if data.empty or set(data[TARGET].unique())!=set(LABELS):
        raise ValueError("Both 'rocky' and 'smooth' labels must be present in category.")
    numeric=data[cols].apply(pd.to_numeric,errors="coerce")
    arr=numeric.to_numpy(dtype=float)
    valid=numeric.notna().all(axis=1) & np.isfinite(arr).all(axis=1)
    excluded=int((~valid).sum())
    data=data.loc[valid]
    if data.empty or set(data[TARGET].unique())!=set(LABELS):
        raise ValueError("Both classes need valid embedding rows after invalid rows are excluded.")
    counts=data[TARGET].value_counts()
    if counts.min()<2: raise ValueError("Need at least two valid rows in each class.")
    X=data[cols].to_numpy(dtype=np.float32); y=data[TARGET].to_numpy()
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=42,stratify=y)
    evaluator=RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced",n_jobs=-1)
    evaluator.fit(Xtr,ytr); pred=evaluator.predict(Xte)
    metrics={"accuracy":accuracy_score(yte,pred),
             "precision":precision_score(yte,pred,labels=LABELS,average="macro",zero_division=0),
             "recall":recall_score(yte,pred,labels=LABELS,average="macro",zero_division=0),
             "f1":f1_score(yte,pred,labels=LABELS,average="macro",zero_division=0),
             "cm":confusion_matrix(yte,pred,labels=LABELS),
             "report":classification_report(yte,pred,labels=LABELS,output_dict=True,zero_division=0),
             "n_train":len(ytr),"n_test":len(yte)}
    final=RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced",n_jobs=-1)
    final.fit(X,y)
    bundle={"model":final,"feature_columns":cols,"classes":LABELS,
            "feature_source":"Orange Image Embedding — SqueezeNet","n_training_rows":len(y)}
    return bundle,metrics,counts.to_dict(),excluded

with st.sidebar:
    st.markdown("## MARS / ATLAS")
    st.caption("Surface classification workspace")
    st.divider()
    st.markdown("**WORKFLOW**")
    st.markdown("01  Train from Orange embeddings")
    st.markdown("02  Predict from Orange embeddings")
    st.divider()
    st.markdown("**FEATURE EXTRACTOR**")
    st.write("Orange Image Embedding")
    st.write("SqueezeNet · 1,000 features")
    st.divider()
    st.caption("Raw images must be embedded in Orange before upload.")

st.title("Mars Surface Atlas")
st.markdown("Train and evaluate the rocky/smooth classifier using the original Orange SqueezeNet embeddings.")
train_tab,predict_tab,method_tab=st.tabs(["01 · Train model","02 · Classify images","03 · Method"])

with train_tab:
    st.subheader("Train from labelled Orange embeddings")
    st.write("In Orange: **Import Images → Image Embedding (SqueezeNet) → Save Data**. Upload the labelled CSV with `category` and `n0`–`n999`.")
    training_file=st.file_uploader("Training embeddings CSV",type=["csv"],key="training_csv")
    test_size=st.selectbox("Holdout test set",[0.20,0.25,0.30],index=1,format_func=lambda x:f"{int(x*100)}%")
    if training_file:
        try:
            df,mode=read_orange_csv(training_file)
            cols=feature_columns(df)
            st.caption(f"Detected {mode}; {len(cols)} embedding features.")
            if TARGET not in df.columns:
                st.error("The CSV has no 'category' column. Keep the labels when exporting from Orange.")
            else:
                labs=df[TARGET].dropna().astype(str).str.strip().str.lower()
                labs=labs[labs.isin(LABELS)].value_counts()
                a,b,c=st.columns(3)
                a.metric("CSV rows",len(df)); b.metric("Rocky labels",int(labs.get("rocky",0))); c.metric("Smooth labels",int(labs.get("smooth",0)))
                if st.button("Train Random Forest",type="primary",use_container_width=True):
                    with st.spinner("Training and evaluating..."):
                        bundle,metrics,counts,excluded=train_model(df,test_size)
                    st.session_state["orange_bundle"]=bundle
                    st.session_state["orange_metrics"]=metrics
                    st.session_state["orange_counts"]=counts
                    st.session_state["orange_excluded"]=excluded
                    st.success(f"Trained on {sum(counts.values())} valid labelled images.")
        except Exception as e:
            st.error(f"CSV validation failed: {e}")
    if "orange_metrics" in st.session_state:
        m=st.session_state["orange_metrics"]
        st.divider(); st.subheader("Held-out evaluation")
        st.caption(f"{m['n_train']} training rows · {m['n_test']} held-out rows. Final downloadable model is refitted on all valid labelled rows.")
        a,b,c,d=st.columns(4)
        a.metric("Accuracy",f"{m['accuracy']:.1%}"); b.metric("Precision",f"{m['precision']:.1%}")
        c.metric("Recall",f"{m['recall']:.1%}"); d.metric("F1",f"{m['f1']:.1%}")
        st.markdown("**Confusion matrix**")
        st.dataframe(pd.DataFrame(m["cm"],index=["Actual rocky","Actual smooth"],columns=["Predicted rocky","Predicted smooth"]),use_container_width=True)
        st.markdown("**Per-class report**")
        st.dataframe(pd.DataFrame(m["report"]).T.round(3),use_container_width=True)
        if st.session_state.get("orange_excluded",0):
            st.warning(f"{st.session_state['orange_excluded']} invalid embedding rows were excluded.")
        model_buffer=io.BytesIO()
        joblib.dump(st.session_state["orange_bundle"],model_buffer)
        st.download_button("Download trained model (.joblib)",model_buffer.getvalue(),"mars_orange_random_forest.joblib","application/octet-stream",use_container_width=True)

with predict_tab:
    st.subheader("Predict from new Orange embeddings")
    st.write("Embed each new image in Orange using the same SqueezeNet widget, then export its embedding CSV. Do not upload raw JPG/PNG files here.")
    model_upload=st.file_uploader("Load trained model (.joblib)",type=["joblib"],key="model_upload")
    if model_upload:
        try:
            loaded=joblib.load(io.BytesIO(model_upload.getvalue()))
            if not isinstance(loaded,dict) or "model" not in loaded or "feature_columns" not in loaded:
                st.error("This is not a compatible Orange-embeddings model.")
            else:
                st.session_state["orange_bundle"]=loaded
                st.success("Model loaded.")
        except Exception as e: st.error(f"Model loading failed: {e}")
    bundle=st.session_state.get("orange_bundle")
    if bundle:
        st.caption(f"Active model · {bundle.get('feature_source','Orange embeddings')} · {bundle.get('n_training_rows','?')} training rows")
        pred_file=st.file_uploader("New image embeddings CSV",type=["csv"],key="prediction_csv")
        if pred_file and st.button("Classify embedded images",type="primary",use_container_width=True):
            try:
                df,mode=read_orange_csv(pred_file)
                X=get_features(df,bundle["feature_columns"])
                names=image_names(df)
                model=bundle["model"]; preds=model.predict(X); probs=model.predict_proba(X)
                rows=[]
                for name,pred,prob in zip(names,preds,probs):
                    scores=dict(zip(model.classes_,prob))
                    rows.append({"Image":Path(str(name).replace("\\","/")).name or str(name),
                                 "Prediction":str(pred).upper(),
                                 "Rocky score":float(scores.get("rocky",0)),
                                 "Smooth score":float(scores.get("smooth",0)),
                                 "Model score":float(scores[pred])})
                result=pd.DataFrame(rows)
                st.session_state["orange_results"]=result
                st.success(f"Classified {len(result)} embedded images · {mode}.")
            except Exception as e: st.error(f"Prediction failed: {e}")
        if "orange_results" in st.session_state:
            result=st.session_state["orange_results"]
            shown=result.copy()
            for col in ["Rocky score","Smooth score","Model score"]: shown[col]=shown[col].map(lambda x:f"{x:.1%}")
            st.dataframe(shown,use_container_width=True,hide_index=True)
            st.download_button("Download predictions CSV",result.to_csv(index=False).encode("utf-8"),"mars_surface_predictions.csv","text/csv")

with method_tab:
    st.subheader("Method and limitations")
    st.markdown("""
**Feature extraction:** Orange's Image Embedding widget creates the 1,000 SqueezeNet features (`n0`–`n999`). This app consumes those features directly; it does not substitute Torchvision's implementation.

**Classifier:** Random Forest with 300 trees, `random_state=42`, `class_weight="balanced"`, and a stratified 75/25 holdout by default. After evaluation, the downloadable model is refitted on all valid labelled rows, matching the original Python workflow.

**Prediction:** New images must be embedded in Orange with the same SqueezeNet widget and exported as CSV. The `category` column is not required for prediction.

**Evaluation caution:** The earlier 100% score was one 58-image holdout result, not a guarantee on new scenes. Keep near-duplicate rover frames out of both sides of the same split.
""")
