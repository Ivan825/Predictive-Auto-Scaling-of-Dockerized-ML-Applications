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
    df: pd.DataFrame,
    horizon_minutes: int = 0,
    freq: str = "min",
    future_regressors: Optional[pd.DataFrame] = None,
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
    future_regressors : optional DataFrame providing regressor values for the
                        future horizon. Must include a datetime column named "timestamp"
                        (or "ds") plus the required regressor columns. If omitted,
                        we will fill required future regressors with 0.

    Returns
    -------
    DataFrame with columns ["ds","yhat","yhat_lower","yhat_upper"]
    """
    # Determine which regressors the model expects
    regressor_cols = list(getattr(model, "extra_regressors", {}).keys())

    if horizon_minutes == 0:
        # In-sample: pass the same regressors used in training
        pdf = df.rename(columns={"timestamp": "ds", "req_per_min": "y"}).copy()
        # Ensure required regressors are present (Prophet will error if missing)
        for col in regressor_cols:
            if col not in pdf.columns:
                # If a regressor was added at training time, it MUST be present here.
                # Default to 0 if you truly want no effect for those rows.
                pdf[col] = 0
        fc = model.predict(pdf)
        return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    # Future forecast
    future = model.make_future_dataframe(periods=horizon_minutes, freq=freq)

    if regressor_cols:
        # Prepare a future regressor frame aligned on "ds"
        if future_regressors is not None:
            fr = future_regressors.copy()
            # Normalize column names
            if "ds" not in fr.columns and "timestamp" in fr.columns:
                fr = fr.rename(columns={"timestamp": "ds"})
            # Keep only needed columns
            keep = ["ds"] + [c for c in regressor_cols if c in fr.columns]
            fr = fr[keep].copy()
        else:
            # No values provided: default all future regressor values to 0
            fr = pd.DataFrame({"ds": future["ds"]})
            for col in regressor_cols:
                fr[col] = 0

        # Merge regressors into the future frame
        future = future.merge(fr, on="ds", how="left")
        # If any required regressor missing post-merge, fill with 0
        for col in regressor_cols:
            if col not in future.columns:
                future[col] = 0
            future[col] = future[col].fillna(0)

    fc = model.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]
