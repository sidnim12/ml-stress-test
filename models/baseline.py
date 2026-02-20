
# Data handling
import pandas as pd

#Data splitting
from sklearn.model_selection import train_test_split

# Preprocessing building blocks
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Pipeline (end-to-end workflow)
from sklearn.pipeline import Pipeline

# Model
from sklearn.linear_model import LogisticRegression

# Evaluation / Metrics
from sklearn.metrics import accuracy_score


#------------------------------------------------------------------------------------------


def train_baseline_model(df: pd.DataFrame, target_col: str, seed: int = 42):
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categoric_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    
    
    
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])   
    
    categoric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols ),
        ("cat", categoric_pipeline, categoric_cols)
    ])
    
    
    model = Pipeline([
        ("preprocessor",preprocessor),
         ("logreg", LogisticRegression(max_iter=1000))
    ])
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=42,
                                                        stratify=y if y.nunique() <= 20 else None
                                                        )
    model.fit(X_train, y_train)
    
    y_preds = model.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_preds)
    
    return model, accuracy
    
    