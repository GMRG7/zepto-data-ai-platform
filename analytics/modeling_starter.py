"""Leakage-aware Titanic classification and fare regression experiments."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix,roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,mean_absolute_error,mean_squared_error,r2_score)
from sklearn.base import clone
import joblib
HERE=Path(__file__).resolve().parent; OUT=HERE/"outputs"; OUT.mkdir(exist_ok=True)
NUMERIC=["age","sibsp","parch","fare"]; CATEGORICAL=["sex","embarked","class","alone"]
FEATURES=NUMERIC+CATEGORICAL

def _prep():
    num=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num",num,NUMERIC),("cat",cat,CATEGORICAL)])
def _metrics(model,X,y):
    pred=model.predict(X); prob=model.predict_proba(X)[:,1] if hasattr(model,"predict_proba") else None
    return {"accuracy":accuracy_score(y,pred),"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),"roc_auc":roc_auc_score(y,prob) if prob is not None else np.nan,"confusion_matrix":confusion_matrix(y,pred).tolist()}
def main():
    df=pd.read_csv(HERE/"titanic.csv") if (HERE/"titanic.csv").exists() else pd.read_csv("titanic.csv")
    # Remove columns with high missingness and identifiers/leakage; split BEFORE fitting imputers/encoders/scalers.
    df=df.drop(columns=[c for c in ["deck","alive","who","adult_male","class","embark_town","embarked"] if c in df.columns and c in ["deck","alive","who","adult_male","embark_town"]],errors="ignore")
    # Map original class and embarked if present; preserve class categorical feature and port.
    if "class" not in df and "pclass" in df: df["class"]=df["pclass"].map({1:"First",2:"Second",3:"Third"})
    if "embarked" not in df: df["embarked"]=np.nan
    for c in CATEGORICAL:
        if c not in df: df[c]=np.nan
    for c in NUMERIC:
        if c not in df: df[c]=np.nan
    X=df[FEATURES].copy(); y=df["survived"].astype(int)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    models={"LogisticRegression":LogisticRegression(max_iter=1500,random_state=42),"DecisionTree":DecisionTreeClassifier(random_state=42),"RandomForest":RandomForestClassifier(n_estimators=250,random_state=42,n_jobs=-1)}
    results=[]; fitted={}
    for name,est in models.items():
        pipe=Pipeline([("preprocess",_prep()),("model",est)]); pipe.fit(X_train,y_train); fitted[name]=pipe
        m=_metrics(pipe,X_test,y_test); results.append({"model":name,**{k:v for k,v in m.items() if k!="confusion_matrix"},"confusion_matrix":str(m["confusion_matrix"])})
        print(name,m)
    pd.DataFrame(results).to_csv(OUT/"classifier_metrics.csv",index=False)
    # Class-weight balancing comparison on the identical train/test split.
    balanced=Pipeline([("preprocess",_prep()),("model",RandomForestClassifier(n_estimators=250,class_weight="balanced",random_state=42,n_jobs=-1))]); balanced.fit(X_train,y_train)
    imbalance=[{"method":"baseline",**{k:v for k,v in _metrics(fitted["RandomForest"],X_test,y_test).items() if k!="confusion_matrix"}}, {"method":"class_weight_balanced",**{k:v for k,v in _metrics(balanced,X_test,y_test).items() if k!="confusion_matrix"}}]
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        smote_pipe=ImbPipeline([("preprocess",_prep()),("smote",SMOTE(random_state=42)),("model",RandomForestClassifier(n_estimators=250,random_state=42,n_jobs=-1))])
        smote_pipe.fit(X_train,y_train)
        imbalance.append({"method":"SMOTE_train_only",**{k:v for k,v in _metrics(smote_pipe,X_test,y_test).items() if k!="confusion_matrix"}})
    except Exception as exc: warnings.warn(f"SMOTE comparison skipped: {exc}")
    pd.DataFrame(imbalance).to_csv(OUT/"imbalance_comparison.csv",index=False)
    rf=Pipeline([("preprocess",_prep()),("model",RandomForestClassifier(n_estimators=250,bootstrap=True,oob_score=True,random_state=42,n_jobs=-1))])
    search=GridSearchCV(rf,{"model__max_depth":[None,5,10,20],"model__max_features":["sqrt",0.5]},cv=5,scoring="f1",n_jobs=-1,refit=True); search.fit(X_train,y_train)
    best=search.best_estimator_; model=best.named_steps["model"]
    (OUT/"random_forest_tuning.txt").write_text(f"Best parameters: {search.best_params_}\nBest CV F1: {search.best_score_:.4f}\nOOB score: {model.oob_score_:.4f}\n",encoding="utf-8")
    joblib.dump(best,OUT/"titanic_classifier_pipeline.joblib")
    # Fare regression uses a separate split and train-only preprocessing.
    regdf=df.dropna(subset=["fare"]).copy(); yfare=regdf["fare"].astype(float)
    regfeatures=[c for c in ["age","sibsp","parch","pclass","sex","embarked"] if c in regdf.columns]
    XR=regdf[regfeatures]; Xr_train,Xr_test,yr_train,yr_test=train_test_split(XR,yfare,test_size=.2,random_state=42)
    rn=[c for c in regfeatures if pd.api.types.is_numeric_dtype(XR[c])]; rc=[c for c in regfeatures if c not in rn]
    trans=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),rn),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),rc)])
    reg=Pipeline([("preprocess",trans),("model",LinearRegression())]); reg.fit(Xr_train,yr_train); yp=reg.predict(Xr_test)
    rmse=float(np.sqrt(mean_squared_error(yr_test,yp))); r2=float(r2_score(yr_test,yp)); n=len(yr_test); p=max(1,len(regfeatures)); adj=1-(1-r2)*(n-1)/(n-p-1) if n>p+1 else np.nan
    residual=yr_test.to_numpy()-yp
    regmetrics={"MAE":mean_absolute_error(yr_test,yp),"RMSE":rmse,"R2":r2,"adjusted_R2":adj,"residual_fitted_correlation":float(np.corrcoef(residual,yp)[0,1]) if np.std(residual)>0 and np.std(yp)>0 else np.nan}
    pd.DataFrame([regmetrics]).to_csv(OUT/"fare_regression_metrics.csv",index=False)
    joblib.dump(reg,OUT/"fare_regression_pipeline.joblib")
    print("Fare regression:",regmetrics)
    print("Outputs saved to",OUT)
if __name__=="__main__":main()
