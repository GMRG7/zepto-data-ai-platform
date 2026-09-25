"""Titanic profiling and exploratory analysis; charts are saved under analytics/outputs."""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; OUT=HERE/"outputs"
OUT.mkdir(exist_ok=True)
def load_data():
    local=HERE/"titanic.csv"
    if local.exists(): return pd.read_csv(local)
    df=sns.load_dataset("titanic"); df.to_csv(local,index=False); return df
def profile(df):
    print("shape:",df.shape); df.info(); print(df.describe(include="all").T)
    missing=(df.isna().mean()*100).sort_values(ascending=False)
    print("Missing %:\n",missing.to_string()); missing.to_csv(OUT/"missing_percent.csv",header=["missing_percent"])
    # Transparent threshold policy: drop columns with >50% missing; median/mode imputation is handled in modeling pipeline.
    dropped=missing[missing>50].index.tolist()
    print("Columns exceeding 50% missingness (drop from analysis):",dropped)
    return missing
def _save(fig,name):
    fig.tight_layout(); fig.savefig(OUT/name,dpi=150,bbox_inches="tight"); plt.close(fig)
def eda(df):
    for col in ["age","fare"]:
        fig,axes=plt.subplots(1,2,figsize=(10,4)); sns.histplot(data=df,x=col,kde=True,ax=axes[0]); sns.boxplot(data=df,x=col,ax=axes[1]); _save(fig,f"{col}_distribution.png")
        s=df[col].dropna(); q1,q3=s.quantile([.25,.75]); iqr=q3-q1; n=int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum()); print(f"{col} IQR outlier count: {n}")
    fig,ax=plt.subplots(figsize=(6,4)); sns.countplot(data=df,x="survived",ax=ax); ax.set_title("Survival count (0=No, 1=Yes)"); _save(fig,"survival_count.png")
    fig,ax=plt.subplots(figsize=(7,4)); sns.barplot(data=df,x="sex",y="survived",errorbar=None,ax=ax); ax.set_title("Survival rate by sex"); _save(fig,"survival_by_sex.png")
    fig,ax=plt.subplots(figsize=(7,4)); sns.barplot(data=df,x="class",y="survived",errorbar=None,ax=ax); ax.set_title("Survival rate by passenger class"); _save(fig,"survival_by_class.png")
    fig,ax=plt.subplots(figsize=(7,4)); sns.barplot(data=df,x="embarked",y="survived",errorbar=None,ax=ax); ax.set_title("Survival rate by embarkation port"); _save(fig,"survival_by_embarked.png")
    for col in ["sex","class","embarked","alone"]:
        if col in df: df.groupby(col,dropna=False)["survived"].agg(survival_rate="mean",passengers="size").to_csv(OUT/f"survival_by_{col}.csv")
    cols=["survived","pclass","age","sibsp","parch","fare"]
    corr=df[cols].corr(numeric_only=True)
    fig,ax=plt.subplots(figsize=(8,6)); sns.heatmap(corr,annot=True,cmap="coolwarm",center=0,ax=ax); ax.set_title("Correlation matrix (six specified columns)"); _save(fig,"correlation_heatmap.png")
    # Standardization sanity check: verify transformed age/fare have approx zero mean and unit SD.
    from sklearn.preprocessing import StandardScaler
    vals=df[["age","fare"]].copy(); vals=vals.fillna(vals.median()); scaled=StandardScaler().fit_transform(vals)
    sanity=pd.DataFrame({"feature":["age","fare"],"before_mean":vals.mean().values,"before_std":vals.std(ddof=0).values,"after_mean":scaled.mean(axis=0),"after_std":scaled.std(axis=0)})
    sanity.to_csv(OUT/"standardization_sanity.csv",index=False); print(sanity.to_string(index=False))
    print("Saved EDA charts/tables to",OUT)
if __name__=="__main__":
    data=load_data(); profile(data); eda(data)
