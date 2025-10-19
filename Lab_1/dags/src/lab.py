import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64
import numpy as np

# ---------- Helpers ----------
def _project_path(*parts):
    return os.path.join(os.path.dirname(__file__), *parts)

def _select_numeric_features(df: pd.DataFrame):
    """Pick numeric columns and drop id-like columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # filter out obvious ID columns
    drop_like = {"customerid", "id"}
    features = [c for c in num_cols if c.lower() not in drop_like]
    if not features:
        raise ValueError("No usable numeric feature columns found.")
    return features

# ---------- Tasks ----------
def load_data():
    """
    Loads training data from mall_customers.csv, serializes it, and returns base64 for XCom.
    Returns: str (base64-encoded pickle of the raw DataFrame)
    """
    df = pd.read_csv(_project_path("../data/mall_customers.csv"))
    serialized = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized).decode("ascii")  # JSON-safe string


def data_preprocessing(data_b64: str):
    """
    Deserializes raw DataFrame, selects numeric features (auto), fits MinMax scaler,
    transforms train data, and returns base64 of a dict: {X, scaler, features}.
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    features = _select_numeric_features(df)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features])

    payload = {
        "X": X,                    # numpy array
        "scaler": scaler,          # fitted scaler
        "features": features       # list of feature names to reuse on test
    }
    return base64.b64encode(pickle.dumps(payload)).decode("ascii")


def build_save_model(preproc_b64: str, filename: str):
    """
    Builds a KMeans model on preprocessed data (uses elbow to pick k),
    saves model+scaler+feature list to ../model/<filename>, and returns the SSE list.
    """
    # decode -> dict
    payload = pickle.loads(base64.b64decode(preproc_b64))
    X = payload["X"]
    scaler = payload["scaler"]
    features = payload["features"]

    # compute SSE for k=1..49
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    ks = range(1, 50)
    sse = []
    for k in ks:
        km = KMeans(n_clusters=k, **kmeans_kwargs)
        km.fit(X)
        sse.append(km.inertia_)

    # pick elbow if possible, else fallback to 3
    try:
        kl = KneeLocator(list(ks), sse, curve="convex", direction="decreasing")
        k_star = kl.elbow if kl.elbow is not None else 3
    except Exception:
        k_star = 3

    model = KMeans(n_clusters=int(k_star), **kmeans_kwargs).fit(X)

    # save model bundle
    output_dir = _project_path("../model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(
            {"model": model, "scaler": scaler, "features": features, "k_star": int(k_star)},
            f
        )

    return sse  # JSON-serializable list


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model bundle and predicts the first row of mall_customers_test.csv
    using the SAME features and scaler. Returns an int cluster label.
    """
    # load bundle
    model_path = _project_path("../model", filename)
    bundle = pickle.load(open(model_path, "rb"))
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle["features"]

    # elbow log (optional)
    try:
        kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
        print(f"Optimal no. of clusters (elbow): {kl.elbow}")
    except Exception:
        pass

    # read and preprocess test using same columns + scaler
    test_df = pd.read_csv(_project_path("../data/mall_customers_test.csv")).dropna()

    # ensure features exist in test
    missing = [c for c in features if c not in test_df.columns]
    if missing:
        raise ValueError(f"Test data is missing required feature columns: {missing}")

    Xtest = scaler.transform(test_df[features].values)
    pred = model.predict(Xtest)[0]

    return int(pred)
