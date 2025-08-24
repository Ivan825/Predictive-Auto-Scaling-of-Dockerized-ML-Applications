from prophet import Prophet
import pandas as pd
from typing import Iterable, Optional

def _auto_regressors(df: pd.DataFrame, candidate_cols: Iterable[str]) -> list:
    """Select regressor columns that actually exist in df."""
    return [c for c in candidate_cols if c in df.columns]

def train_prophet(
    df: pd.DataFrame,
    regressor_cols: Optional[Iterable[str]] = None,
) -> Prophet:
    """
    Train Prophet with any regressors you want to keep (e.g., ["is_event", "hour", ...]).
    Regressors MUST be present in df during training.

    Parameters
    ----------
    df : DataFrame with columns ["timestamp", "req_per_min", ...regressors]
    regressor_cols : iterable of column names to add as Prophet regressors.
                     If None, we auto-pick ["is_event"] if present.

    Returns
    -------
    Trained Prophet model
    """
    if regressor_cols is None:
        regressor_cols = _auto_regressors(df, ["is_event"])

    pdf = df.rename(columns={"timestamp": "ds", "req_per_min": "y"}).copy()

    m = Prophet()
    for col in regressor_cols:
        m.add_regressor(col)

    m.fit(pdf)
    return m


def forecast_prophet(
    model: Prophet,
    df: Optional[pd.DataFrame] = None,
    horizon_minutes: int = 0,
    freq: str = "min",
    future_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Make predictions with a model that may have regressors.

    Parameters
    ----------
    model : trained Prophet model
    df : training dataframe used for in-sample prediction and to infer regressor names.
         Must have ["timestamp", "req_per_min", ...regressors] columns.
    horizon_minutes : 0 => in-sample prediction on df; >0 => future forecast
    freq : pandas offset alias for step (e.g., "min")
    future_df : optional DataFrame providing future dates and regressor values. 
                If provided, `horizon_minutes` and `df` are ignored for creating future dates.
                Must include a datetime column named "timestamp" or "ds".

    Returns
    -------
    DataFrame with columns ["ds","yhat","yhat_lower","yhat_upper"]
    """
    # Determine which regressors the model expects
    regressor_cols = list(getattr(model, "extra_regressors", {}).keys())

    if future_df is not None:
        future = future_df.copy()
        if "ds" not in future.columns and "timestamp" in future.columns:
            future = future.rename(columns={"timestamp": "ds"})
    elif horizon_minutes > 0:
        future = model.make_future_dataframe(periods=horizon_minutes, freq=freq)
    else: # In-sample
        if df is None:
            raise ValueError("df must be provided for in-sample forecasting")
        future = df.rename(columns={"timestamp": "ds", "req_per_min": "y"}).copy()

    # Ensure regressors are present and populated
    if regressor_cols:
        # If a dataframe was passed for context, use it to populate regressors
        source_df = df if df is not None else pd.DataFrame()
        if "ds" not in source_df.columns and "timestamp" in source_df.columns:
            source_df = source_df.rename(columns={"timestamp": "ds"})

        # Merge source data to fill known regressor values
        if not source_df.empty:
             # Keep only needed columns from source
            keep_cols = ["ds"] + [c for c in regressor_cols if c in source_df.columns]
            future = future.merge(source_df[keep_cols], on="ds", how="left")

        # Fill any remaining missing regressor values with 0
        for col in regressor_cols:
            if col not in future.columns:
                future[col] = 0
            future[col] = future[col].fillna(0)

    fc = model.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]
