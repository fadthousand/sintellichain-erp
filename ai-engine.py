"""
S-IntelliChain ERP - Fully Dynamic Multi-Product Demand Forecasting & Dynamic ROP Engine
========================================================================================
Decoupled Artificial Intelligence Module for FMCG Frozen Food Inventory Optimization.
Implementation: Multiple Linear Regression & Gradient Descent from Scratch.

"""

import ast
import json
import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Central Configuration for Database, Model Hyperparameters, and ERP Inventory Defaults."""

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "sintellichain_db"
    db_user: str = "odoo"
    db_password: str = "odoo_password"

    # ML Hyperparameters
    learning_rate: float = 0.05
    epochs: int = 1000
    random_seed: int = 42

    # Inventory Policy Defaults
    default_lead_time_days: int = 3
    default_safety_stock: float = 20.0

    # Generic Fallback Products if Database is Completely Empty / Unreachable
    fallback_generic_products: list[str] = field(
        default_factory=lambda: ["Frozen Food Item A", "Frozen Food Item B"]
    )

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL SQLAlchemy Connection String with psycopg2 driver."""
        return f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


def parse_product_name(val: Any) -> str:
    """Extract clean string product name from Odoo 17 JSONB or dict structure."""
    if isinstance(val, dict):
        return str(val.get("en_US") or val.get("en_GB") or next(iter(val.values()), str(val)))

    if isinstance(val, str):
        val_str = val.strip()
        if val_str.startswith("{") and val_str.endswith("}"):
            try:
                parsed = ast.literal_eval(val_str)
                if isinstance(parsed, dict):
                    return str(parsed.get("en_US") or next(iter(parsed.values()), val_str))
            except (ValueError, SyntaxError):
                pass
            try:
                parsed = json.loads(val_str)
                if isinstance(parsed, dict):
                    return str(parsed.get("en_US") or next(iter(parsed.values()), val_str))
            except json.JSONDecodeError:
                pass
        return val_str

    return str(val)


def fetch_active_odoo_products(engine: Engine) -> list[str]:
    """Query active product names from Odoo product_template table."""
    query = "SELECT name FROM product_template WHERE active = TRUE ORDER BY id ASC;"
    try:
        df_tmpl = pd.read_sql_query(query, engine)
        if not df_tmpl.empty:
            clean_names = [
                parse_product_name(raw)
                for raw in df_tmpl["name"]
                if parse_product_name(raw) not in ("Tips", "Discount")
            ]
            if clean_names:
                return clean_names
    except SQLAlchemyError as err:
        logger.debug(f"Could not fetch product_template records: {err}")
    return []


def generate_synthetic_sales_dataset(product_names: list[str], days: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate generic, realistic FMCG sales history dynamically for any given list of product names."""
    logger.info(f"Generating dynamic synthetic FMCG sales dataset ({days} days) for products: {product_names}")
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.now().floor("D"), periods=days, freq="D")
    day_of_week_arr = np.array([d.dayofweek for d in dates], dtype=np.int64)

    dataframes: list[pd.DataFrame] = []

    for idx, prod in enumerate(product_names):
        # Generate dynamic baseline sales demand & trend per product
        base_demand = 30.0 + (idx * 15.0)
        trend = np.sin(np.linspace(0, 2 * np.pi, days)) * 10.0
        noise = np.random.normal(0, 3.0, days)
        weekend_spike = np.where(day_of_week_arr >= 5, 8.0, 0.0)

        sales_qty = np.maximum(1.0, np.round(base_demand + trend + noise + weekend_spike))

        df_prod = pd.DataFrame({
            "sales_date": dates,
            "product_name": prod,
            "total_quantity_sold": sales_qty,
        })
        dataframes.append(df_prod)

    return pd.concat(dataframes, ignore_index=True)


def fetch_pos_sales_data(config: EngineConfig) -> pd.DataFrame:
    """Fetch historical POS daily sales aggregation from PostgreSQL Odoo.

    Fallback Hierarchy:
    1. Real POS transaction data from pos_order_line.
    2. Active master products in Odoo product_template -> Generate synthetic initial history for actual products.
    3. Generic FMCG product list -> Fully decoupled synthetic fallback.
    """
    logger.info("Connecting to Odoo PostgreSQL database...")
    query_pos = """
    SELECT
        DATE(po.date_order) AS sales_date,
        pt.name AS raw_product_name,
        SUM(pol.qty) AS total_quantity_sold
    FROM pos_order_line pol
    JOIN pos_order po ON pol.order_id = po.id
    JOIN product_product pp ON pol.product_id = pp.id
    JOIN product_template pt ON pp.product_tmpl_id = pt.id
    GROUP BY DATE(po.date_order), pt.name
    ORDER BY sales_date ASC;
    """

    engine: Engine | None = None
    try:
        engine = create_engine(config.database_url)
        df = pd.read_sql_query(query_pos, engine)
        if not df.empty:
            logger.info(f"Successfully retrieved {len(df)} transaction records from Odoo database.")
            df["product_name"] = df["raw_product_name"].apply(parse_product_name)
            return df

        logger.info("POS transaction table is empty. Querying active product master records from Odoo...")
        active_products = fetch_active_odoo_products(engine)
        if active_products:
            logger.info(f"Discovered {len(active_products)} active product(s) in Odoo: {active_products}")
            return generate_synthetic_sales_dataset(active_products, days=30, seed=config.random_seed)

    except SQLAlchemyError as err:
        logger.warning(f"Database query unfulfilled ({err}). Activating decoupled synthetic engine.")

    # Final fallback if DB is unreachable or product_template is empty
    return generate_synthetic_sales_dataset(
        config.fallback_generic_products, days=30, seed=config.random_seed
    )


class LinearRegressionGradientDescent:
    """Multiple Linear Regression trained via Gradient Descent from scratch.

    Hypothesis: Y_pred = X * w + b
    Loss Function: Mean Squared Error (MSE) = (1 / 2n) * sum((Y_pred - Y)^2)
    """

    def __init__(self, learning_rate: float = 0.05, epochs: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: np.ndarray = np.array([], dtype=np.float64)
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model parameters (weights & bias) using Gradient Descent."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = float(np.mean(y)) if n_samples > 0 else 0.0
        self.loss_history = []

        if n_samples == 0:
            return

        for epoch in range(1, self.epochs + 1):
            # 1. Forward Pass
            y_predicted = np.dot(X, self.weights) + self.bias

            # 2. Compute Loss (MSE)
            loss = float((1 / (2 * n_samples)) * np.sum((y_predicted - y) ** 2))
            self.loss_history.append(loss)

            # 3. Compute Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = float((1 / n_samples) * np.sum(y_predicted - y))

            # 4. Update Parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % 200 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:4d}/{self.epochs} - MSE Loss: {loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for feature matrix X."""
        return np.asarray(np.dot(X, self.weights) + self.bias, dtype=np.float64)


def calculate_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Compute Evaluation Metrics:

    1. RMSE (Root Mean Squared Error)
    2. MAPE (Mean Absolute Percentage Error)
    """
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    epsilon = 1e-8
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)
    return rmse, mape


def calculate_reorder_point(
    predicted_daily_demand: float, lead_time_days: int, safety_stock: float
) -> float:
    """Calculate Dynamic Reorder Point (ROP):

    ROP = (Predicted Daily Demand * Lead Time) + Safety Stock
    """
    rop = (predicted_daily_demand * lead_time_days) + safety_stock
    return float(rop)


def forecast_product_demand(df_prod_input: pd.DataFrame, product_name: str, config: EngineConfig) -> None:
    """Run preprocessing, model training, evaluation, and ROP calculation for a single product."""
    print("\n" + "=" * 60)
    print(f" FORECASTING ENGINE FOR PRODUCT: {product_name}")
    print("=" * 60)

    # Sort chronology and ensure clean DataFrame instance
    df_prod: pd.DataFrame = cast(pd.DataFrame, df_prod_input.sort_values("sales_date")).copy()

    # Feature Engineering
    df_prod["sales_date"] = pd.to_datetime(df_prod["sales_date"])
    df_prod["day_index"] = np.arange(1, len(df_prod) + 1, dtype=np.int64)
    df_prod["day_of_week"] = df_prod["sales_date"].dt.dayofweek

    print(f"\n[Data Pipeline] Processed Sales History ({len(df_prod)} record(s)):\n")
    print(df_prod[["sales_date", "product_name", "total_quantity_sold"]].tail(5).to_string(index=False))

    X_raw: np.ndarray = df_prod[["day_index", "day_of_week"]].to_numpy(dtype=np.float64)
    y: np.ndarray = df_prod["total_quantity_sold"].to_numpy(dtype=np.float64)

    # Feature Scaling (Z-score Normalization) with Zero-Variance Guard
    mean_X: np.ndarray = np.asarray(np.mean(X_raw, axis=0), dtype=np.float64)
    std_X: np.ndarray = np.asarray(np.std(X_raw, axis=0), dtype=np.float64)
    std_X = np.where(std_X == 0, 1.0, std_X)  # Prevent division by zero if std == 0

    X_scaled: np.ndarray = np.asarray((X_raw - mean_X) / std_X, dtype=np.float64)

    # Step 3: Train Model via Gradient Descent
    logger.info(f"Training Gradient Descent model for '{product_name}'...")
    model = LinearRegressionGradientDescent(
        learning_rate=config.learning_rate, epochs=config.epochs
    )
    model.fit(X_scaled, y)

    # Step 4: Model Evaluation
    y_pred: np.ndarray = model.predict(X_scaled)
    rmse, mape = calculate_evaluation_metrics(y, y_pred)

    print("\n" + "-" * 60)
    print(" MODEL PERFORMANCE EVALUATION METRICS")
    print("-" * 60)
    print(f" Root Mean Squared Error (RMSE) : {rmse:.2f} units")
    print(f" Mean Absolute Percentage Error (MAPE) : {mape:.2f}%")
    print(f" Optimized Weights (w)          : {model.weights}")
    print(f" Optimized Bias (b)             : {model.bias:.2f}")

    # Step 5: Dynamic Reorder Point (ROP) Recommendation
    next_day_index = len(df_prod) + 1
    max_date = df_prod["sales_date"].max()
    next_date = max_date + pd.Timedelta(days=1)
    next_day_of_week = next_date.dayofweek

    next_X_raw: np.ndarray = np.array([[next_day_index, next_day_of_week]], dtype=np.float64)
    next_X_scaled: np.ndarray = np.asarray((next_X_raw - mean_X) / std_X, dtype=np.float64)

    predicted_tomorrow_sales = max(0.0, float(model.predict(next_X_scaled)[0]))
    rop_recommendation = calculate_reorder_point(
        predicted_tomorrow_sales,
        lead_time_days=config.default_lead_time_days,
        safety_stock=config.default_safety_stock,
    )

    print("\n" + "-" * 60)
    print(" INVENTORY REORDER POINT (ROP) DECISION SUPPORT")
    print("-" * 60)
    print(f" Target Product                 : {product_name}")
    print(f" Predicted Demand ({next_date.strftime('%Y-%m-%d')}) : {predicted_tomorrow_sales:.2f} units")
    print(f" Supplier Lead Time             : {config.default_lead_time_days} Days")
    print(f" Safety Stock Level             : {config.default_safety_stock:.0f} Units")
    print(f" Dynamic Reorder Point (ROP)    : {rop_recommendation:.2f} units")
    print(
        f" ACTION THRESHOLD               : Trigger Purchase Order if Stock <= {int(np.ceil(rop_recommendation))} units"
    )
    print("=" * 60)


def main() -> None:
    print("=" * 60)
    print(" S-INTELLICHAIN ERP - DYNAMIC MULTI-PRODUCT FORECASTING ENGINE")
    print(" Algorithmic Paradigm: Linear Regression & Gradient Descent")
    print("=" * 60)

    # Initialize Engine Configuration
    config = EngineConfig()

    # Step 1: Fetch Data via Resilient Fallback Pipeline
    df_all = fetch_pos_sales_data(config)

    if df_all.empty:
        logger.error("No sales data available to perform forecasting.")
        return

    # Identify all unique products in dataset dynamically
    unique_products = df_all["product_name"].unique()
    logger.info(f"Identified {len(unique_products)} distinct product(s) in sales dataset: {list(unique_products)}")

    # Loop forecasting engine for each product dynamically
    for prod in unique_products:
        sub_df = df_all[df_all["product_name"] == prod]
        if isinstance(sub_df, pd.DataFrame):
            df_prod: pd.DataFrame = sub_df.copy()
            forecast_product_demand(df_prod, str(prod), config)


if __name__ == "__main__":
    main()
