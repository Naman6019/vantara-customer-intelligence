from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("segmentation")

SEGMENTATION_FEATURES = [
    "recency",
    "frequency",
    "monetary_total",
    "return_rate",
    "engagement_score",
]


def evaluate_kmeans_k_range(
    X_scaled: np.ndarray, k_min: int = 2, k_max: int = 8, seed: int = 42
) -> Tuple[int, pd.DataFrame]:
    """
    Evaluates K-Means across a range of k using Inertia (Elbow method)
    and Silhouette scores to empirically determine the optimal number of clusters.
    """
    results = []
    best_k = 4
    best_silhouette = -1.0

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        inertia = km.inertia_

        results.append({
            "k": k,
            "inertia": round(inertia, 2),
            "silhouette": round(sil, 4),
            "davies_bouldin": round(db, 4),
        })

        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k

    metrics_df = pd.DataFrame(results)
    logger.info(f"Optimal K selection: recommended k={best_k} (Silhouette: {best_silhouette:.4f})")
    return best_k, metrics_df


def assign_business_personas(profiles: pd.DataFrame) -> Dict[int, str]:
    """
    Translates raw cluster metrics into actionable marketing personas based on
    Recency, Frequency, Monetary value, and Engagement score.
    """
    personas = {}
    for idx, row in profiles.iterrows():
        cluster_id = int(row["cluster"])
        r = row["recency"]
        f = row["frequency"]
        m = row["monetary_total"]
        eng = row["engagement_score"]

        if eng >= 75 or (f >= 8 and m >= 2500):
            personas[cluster_id] = "High-Value Champions (VIP)"
        elif r <= 60 and f >= 3:
            personas[cluster_id] = "Loyal & Active Frequent Buyers"
        elif r > 180 and m >= 1000:
            personas[cluster_id] = "At-Risk / Lapsed High-Value"
        elif r <= 90 and f <= 2:
            personas[cluster_id] = "Recent Onboarding / New Prospects"
        elif r > 180 and f <= 2:
            personas[cluster_id] = "Dormant / Low-Engagement"
        else:
            personas[cluster_id] = f"Standard Active Segment {cluster_id}"

    return personas


def run_customer_segmentation(
    df: Optional[pd.DataFrame] = None, save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    Executes K-Means and Gaussian Mixture Model clustering on customer RFM & engagement features.
    Computes cluster quality metrics, assigns business personas, and exports results.
    """
    root = get_project_root()
    config = load_config()

    if df is None:
        proc_path = root / config.get("paths", {}).get("processed_dir", "data/processed") / "customer_features.parquet"
        df = pd.read_parquet(proc_path)

    seed = config.get("project", {}).get("random_seed", 42)
    features_df = df[SEGMENTATION_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    # 1. Optimal K analysis
    best_k, k_metrics = evaluate_kmeans_k_range(X_scaled, k_min=3, k_max=7, seed=seed)
    # Default to 4 clusters for distinct marketing segmentation if best_k is too granular
    target_k = 4 if best_k < 3 or best_k > 6 else best_k

    # 2. Fit K-Means
    logger.info(f"Fitting production K-Means model with k={target_k}...")
    kmeans = KMeans(n_clusters=target_k, random_state=seed, n_init=15)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # 3. Fit Gaussian Mixture Model (PRD Section 9 requirement)
    logger.info(f"Fitting Gaussian Mixture Model with n_components={target_k}...")
    gmm = GaussianMixture(n_components=target_k, random_state=seed, covariance_type="full")
    gmm_labels = gmm.fit_predict(X_scaled)

    # Calculate validation scores
    km_sil = silhouette_score(X_scaled, kmeans_labels)
    km_db = davies_bouldin_score(X_scaled, kmeans_labels)
    gmm_sil = silhouette_score(X_scaled, gmm_labels)
    gmm_db = davies_bouldin_score(X_scaled, gmm_labels)

    logger.info(f"K-Means Quality: Silhouette={km_sil:.4f}, Davies-Bouldin={km_db:.4f}")
    logger.info(f"GMM Quality: Silhouette={gmm_sil:.4f}, Davies-Bouldin={gmm_db:.4f}")

    # 4. Profile clusters and assign personas
    df_segmented = df.copy()
    df_segmented["kmeans_cluster"] = kmeans_labels
    df_segmented["gmm_cluster"] = gmm_labels

    cluster_summary = (
        df_segmented.groupby("kmeans_cluster")[SEGMENTATION_FEATURES + ["is_churned", "future_clv"]]
        .agg(["mean", "median", "count"])
    )

    profiles = (
        df_segmented.groupby("kmeans_cluster")[SEGMENTATION_FEATURES]
        .mean()
        .reset_index()
        .rename(columns={"kmeans_cluster": "cluster"})
    )
    personas = assign_business_personas(profiles)
    df_segmented["segment_persona"] = df_segmented["kmeans_cluster"].map(personas)

    # 5. Persist artifacts
    if save_artifacts:
        art_dir = root / config.get("paths", {}).get("models_artifacts_dir", "models_artifacts")
        art_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(kmeans, art_dir / "kmeans_model.joblib")
        joblib.dump(gmm, art_dir / "gmm_model.joblib")
        joblib.dump(scaler, art_dir / "segmentation_scaler.joblib")
        joblib.dump(personas, art_dir / "segment_personas.joblib")

        # Save segmented customer dataset
        proc_dir = root / config.get("paths", {}).get("processed_dir", "data/processed")
        df_segmented.to_parquet(proc_dir / "customer_segments.parquet", index=False)
        k_metrics.to_csv(art_dir / "kmeans_k_metrics.csv", index=False)
        logger.info("Segmentation artifacts and enriched dataset saved successfully.")

    return {
        "kmeans_model": kmeans,
        "gmm_model": gmm,
        "k_metrics": k_metrics,
        "personas": personas,
        "kmeans_silhouette": km_sil,
        "kmeans_davies_bouldin": km_db,
        "gmm_silhouette": gmm_sil,
        "gmm_davies_bouldin": gmm_db,
        "segmented_df": df_segmented,
    }


if __name__ == "__main__":
    results = run_customer_segmentation()
    print("\n--- K-Means vs GMM Cluster Evaluation ---")
    print(f"K-Means: Silhouette = {results['kmeans_silhouette']:.4f}, Davies-Bouldin = {results['kmeans_davies_bouldin']:.4f}")
    print(f"GMM:     Silhouette = {results['gmm_silhouette']:.4f}, Davies-Bouldin = {results['gmm_davies_bouldin']:.4f}")
    print("\n--- Assigned Business Personas ---")
    for cluster, persona in results["personas"].items():
        print(f"Cluster {cluster}: {persona}")
    print("\nSegment Distribution:")
    print(results["segmented_df"]["segment_persona"].value_counts())
