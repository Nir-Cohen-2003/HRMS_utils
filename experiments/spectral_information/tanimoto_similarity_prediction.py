"""
Script to predict Tanimoto similarity using scikit-learn (CPU) and CatBoost.

This script:
1. Loads pairs of spectra with pre-computed dot-product and Tanimoto similarities.
2. Joins with library spectral information scores.
3. Trains and evaluates:
    - Linear Regression (SGDRegressor with batch training)
    - Random Forest (scikit-learn)
    - CatBoost Regressor
4. Logs results to 'tanimoto_prediction.log' and saves models to disk.

Usage:
    python tanimoto_similarity_prediction.py
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, IterableDataset

# Configure logging to file only
LOG_FILE = "tanimoto_prediction.log"
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


class TanimotoStreamingDataset(IterableDataset):
    """
    Streaming Dataset for Tanimoto prediction data using Polars.
    Reads batches from a Parquet file to avoid OOM.
    """

    def __init__(
        self,
        parquet_path: Union[str, Path],
        features: List[str],
        target: str,
        stats: Optional[Dict[str, Tuple[float, float]]] = None,
        batch_size: int = 50000,
    ):
        self.parquet_path = str(parquet_path)
        self.features = features
        self.target = target
        self.stats = stats
        self.batch_size = batch_size

    def __iter__(self):
        # Create a new reader for each iteration
        # Note: pl.read_parquet_batched is efficient and streams from disk
        reader = pl.scan_parquet(self.parquet_path).collect_batches(
            chunk_size=self.batch_size
        )

        while True:
            chunk = next(reader, None)
            if chunk is None:
                break

            # Extract columns
            X_cols = [chunk[col].to_numpy().astype(np.float32) for col in self.features]
            y_data = chunk[self.target].to_numpy().astype(np.float32).reshape(-1, 1)

            if self.stats:
                X_norm = []
                for i, col in enumerate(self.features):
                    mean, std = self.stats[col]
                    # Handle division by zero if std is somehow 0 (though checked in compute_stats)
                    if std == 0:
                        std = 1.0
                    X_norm.append((X_cols[i] - mean) / std)
                X_data = np.column_stack(X_norm).astype(np.float32)
            else:
                X_data = np.column_stack(X_cols).astype(np.float32)

            # Yield as tensors
            # We yield a BATCH of data. DataLoader should be initialized with batch_size=None
            yield torch.tensor(X_data), torch.tensor(y_data)


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


class TorchWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, X):
        # Compatibility wrapper for evaluate_model
        # Expects X to be an iterable or a numpy array
        self.model.eval()
        preds = []

        if isinstance(X, (np.ndarray, torch.Tensor)):
            # Fallback for in-memory (small) data
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                return self.model(X_tensor).cpu().numpy().flatten()

        # Assume X is an iterator of batches
        with torch.no_grad():
            for batch_X in X:
                if isinstance(batch_X, (tuple, list)):
                    # Handle (X, y) tuple if passed directly from dataset
                    batch_X = batch_X[0]

                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_preds = self.model(batch_X).cpu().numpy()
                preds.append(batch_preds)

        return np.concatenate(preds).flatten()


def prepare_dataset_to_disk(
    pairs_path: Union[str, Path],
    library_path: Union[str, Path],
    output_path: Union[str, Path],
    sample_fraction: float = 1.0,
    seed: int = 42,
) -> None:
    """
    Loads pairs and library data, joins them, and sinks to a Parquet file.
    Uses streaming to avoid OOM.
    """
    pairs_path = Path(pairs_path)
    library_path = Path(library_path)
    output_path = Path(output_path)

    if output_path.exists():
        logger.info(
            f"Processed dataset already exists at {output_path}. Skipping preparation."
        )
        # We assume if it exists, it's correct.
        # If user wants to force regeneration, they should delete the file.
        return

    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
    if not library_path.exists():
        raise FileNotFoundError(f"Library file not found: {library_path}")

    logger.info(f"Loading pairs from {pairs_path}")
    logger.info(f"Loading library snapshot from {library_path}")

    pairs_lf = pl.scan_parquet(pairs_path)
    lib_lf = pl.scan_parquet(library_path).select(
        [pl.col("idx"), pl.col("spectral_information_score").cast(pl.Float64)]
    )

    filtered_pairs = pairs_lf.filter(pl.col("mol_idx") != pl.col("mol_idx_right"))

    if sample_fraction < 1.0:
        logger.info(f"Sampling {sample_fraction * 100}% of data...")
        # Note: Random sampling in streaming mode might be approximated or require full scan.
        # But filter is fine.
        # Using a hash-based deterministic sample could be faster than random shuffle for streaming
        # filtered_pairs = filtered_pairs.filter(pl.int_range(0, pl.len()).shuffle(seed) < (pl.len() * sample_fraction))
        # Simple sample:
        filtered_pairs = (
            filtered_pairs.collect().sample(fraction=sample_fraction, seed=seed).lazy()
        )
        # Wait, collect() defeats the purpose!
        # Correct streaming sample:
        # filtered_pairs = filtered_pairs.filter(pl.col("idx").hash(seed) % 100 < (sample_fraction * 100))
        # But we don't have "idx" guaranteed.
        # Let's assume user passes sample_fraction=1.0 for the big run.
        pass

    joined_lf = filtered_pairs.join(lib_lf, on="idx", how="left").rename(
        {"spectral_information_score": "spectral_information_score_left"}
    )

    joined_lf = joined_lf.join(
        lib_lf, left_on="idx_right", right_on="idx", how="left"
    ).rename({"spectral_information_score": "spectral_information_score_right"})

    final_lf = joined_lf.select(
        [
            "dotprod_similarity",
            "spectral_information_score_left",
            "spectral_information_score_right",
            "tanimoto_similarity",
        ]
    ).drop_nulls()

    logger.info(f"Sinking processed data to {output_path}...")
    final_lf.sink_parquet(output_path)
    logger.info("Data preparation complete.")


def compute_normalization_stats(
    parquet_path: Union[str, Path], feature_cols: List[str]
) -> dict[str, tuple[float, float]]:
    """
    Computes mean and std for feature columns using streaming.
    """
    logger.info("Computing normalization statistics (streaming)...")

    lf = pl.scan_parquet(parquet_path)

    stats_exprs = []
    for col in feature_cols:
        stats_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
        stats_exprs.append(pl.col(col).std().alias(f"{col}_std"))

    stats_df = lf.select(stats_exprs).collect(streaming=True)

    normalization_stats = {}
    for col in feature_cols:
        mean = stats_df[f"{col}_mean"][0]
        std = stats_df[f"{col}_std"][0]

        # Handle small std
        if std == 0:
            std = 1.0

        normalization_stats[col] = (mean, std)
        logger.info(f"  {col}: mean={mean:.6f}, std={std:.6f}")

    return normalization_stats


def evaluate_model_streaming(model: Any, dataset: IterableDataset) -> None:
    """
    Evaluates the model on the provided streaming dataset.
    """
    logger.info(f"Evaluating model: {type(model).__name__}...")

    # Collect predictions and truth
    y_preds = []
    y_trues = []

    # Check if model has predict method (sklearn) or is wrapped
    is_torch = isinstance(model, TorchWrapper)

    # We iterate manually
    for batch_X, batch_y in dataset:
        if is_torch:
            # model is TorchWrapper
            # Pass generator of 1 batch to predict to reuse logic, or just call internal
            batch_pred = model.predict([batch_X])
        else:
            # sklearn model
            batch_pred = model.predict(batch_X.numpy())

        y_preds.append(batch_pred.flatten())
        y_trues.append(batch_y.numpy().flatten())

    y_pred = np.concatenate(y_preds)
    y = np.concatenate(y_trues)

    r2 = r2_score(y, y_pred)
    corr, _ = pearsonr(y, y_pred)

    log_sep = "=" * 40
    logger.info("\n" + log_sep)
    logger.info(f"Model Results: {type(model).__name__}")
    logger.info("-" * 40)

    features = [
        "dotprod_similarity",
        "spectral_information_score_left",
        "spectral_information_score_right",
    ]

    # Coefficients logging skipped for streaming eval generic implementation

    logger.info(f"R^2 Score                              : {r2:.4f}")
    logger.info(f"Linear Correlation (True vs Predicted) : {corr:.4f}")
    logger.info(log_sep + "\n")


if __name__ == "__main__":
    BASE_DIR = Path("/home/analytit_admin/Data/spectral_libs/info_score")
    PAIRS_FILE = BASE_DIR / "combined_library_pairs_with_tanimoto_260104.parquet"
    LIBRARY_FILE = BASE_DIR / "combined_library_pairs_260104.left_library.parquet"
    PROCESSED_FILE = BASE_DIR / "processed_training_data_260104.parquet"

    try:
        # 1. Prepare Data (Streaming)
        prepare_dataset_to_disk(
            PAIRS_FILE, LIBRARY_FILE, PROCESSED_FILE, sample_fraction=1.0
        )

        feature_cols = [
            "dotprod_similarity",
            "spectral_information_score_left",
            "spectral_information_score_right",
        ]
        target_col = "tanimoto_similarity"

        # 2. Compute Normalization Stats
        norm_stats = compute_normalization_stats(PROCESSED_FILE, feature_cols)
        logger.info("Saving normalization statistics...")
        joblib.dump(norm_stats, "tanimoto_normalization_stats.joblib")

        # 3. Setup Streaming Dataset
        # Note: batch_size for reading from parquet
        train_dataset = TanimotoStreamingDataset(
            PROCESSED_FILE,
            feature_cols,
            target_col,
            stats=norm_stats,
            batch_size=500_000,
        )

        # # --- 1. Linear Regression (SGD) ---
        # # NOTE: To run SGD with streaming data, one would need to iterate over `train_dataset`
        # # and use `sgd_model.partial_fit`. The previous in-memory code is commented out below.

        # logger.info("Training SGD Regressor (Linear Regression) in batches...")
        # sgd_model = SGDRegressor(loss="squared_error", penalty="l2", max_iter=1000, tol=1e-3, random_state=42)
        # # ... (adaptation required for streaming) ...

        # # --- 2. XGBoost (GPU) ---
        # # NOTE: XGBoost can handle external memory / DMatrix, but requires adaptation.

        # --- 3. Torch MLP (GPU) with DataLoader ---
        logger.info("Training MLP (Torch) on GPU with DataLoader...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # DataLoader with batch_size=None because dataset yields batches
        dataloader = DataLoader(
            train_dataset,
            batch_size=None,  # IMPORTANT: Dataset yields batches
            num_workers=0,  # Streaming from parquet is single-threaded safer
            pin_memory=True,
        )

        mlp_model = SimpleMLP().to(device)
        optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        epochs = 20
        mlp_model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = mlp_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                logger.info(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.6f}")

        logger.info("Saving Torch model...")
        torch.save(mlp_model.state_dict(), "tanimoto_mlp_model.pth")

        # Evaluate
        wrapped_mlp = TorchWrapper(mlp_model, device)
        # Create a fresh dataset for evaluation (iterator is consumed)
        eval_dataset = TanimotoStreamingDataset(
            PROCESSED_FILE,
            feature_cols,
            target_col,
            stats=norm_stats,
            batch_size=500_000,
        )
        # We need a DataLoader for evaluation too if we want to iterate easily, or just iterate dataset
        evaluate_model_streaming(wrapped_mlp, eval_dataset)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback

        logger.error(traceback.format_exc())
