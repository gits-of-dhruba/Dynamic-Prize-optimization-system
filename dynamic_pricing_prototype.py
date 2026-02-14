"""
Dynamic Pricing Optimization System - Prototype
===============================================
Automatically adjusts product prices based on real-time data to maximize revenue.
Uses Walmart sales dataset for: demand elasticity, seasonal trends, revenue optimization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def load_data(csv_path: str = None) -> pd.DataFrame:
    """Load Walmart sales dataset."""
    if csv_path is None:
        csv_path = Path(__file__).parent / "archive (1)" / "Walmart.csv"

    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    return df


def add_price_column(df: pd.DataFrame, method: str = 'cpi_fuel') -> pd.DataFrame:
    """
    Add synthetic price column (dataset doesn't have actual prices).
    CPI and Fuel_Price correlate with retail price levels - used as market proxy.
    """
    df = df.copy()
    cpi_norm = (df['CPI'] - df['CPI'].min()) / (df['CPI'].max() - df['CPI'].min() + 1e-6)
    fuel_norm = (df['Fuel_Price'] - df['Fuel_Price'].min()) / (df['Fuel_Price'].max() - df['Fuel_Price'].min() + 1e-6)
    df['Price'] = 3 + 5 * (0.7 * cpi_norm + 0.3 * fuel_norm) + np.random.normal(0, 0.1, len(df))
    df['Price'] = df['Price'].clip(1, 12)
    return df


def estimate_price_elasticity(df: pd.DataFrame, store_id: int = None) -> dict:
    """
    Estimate price elasticity of demand using log-log regression.
    Log(Q) = β0 + β1*Log(P) + controls  =>  elasticity ≈ β1
    """
    df = df.copy()
    df['Quantity'] = df['Weekly_Sales'] / df['Price']

    if store_id is not None:
        df = df[df['Store'] == store_id]

    df_clean = df.dropna(subset=['Quantity', 'Price', 'Weekly_Sales'])
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['Price'] > 0)]
    df_clean['log_quantity'] = np.log(df_clean['Quantity'])
    df_clean['log_price'] = np.log(df_clean['Price'])

    X = df_clean[['log_price', 'Holiday_Flag', 'Temperature', 'Unemployment']]
    y = df_clean['log_quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elasticity = model.coef_[0]

    return {
        'elasticity': elasticity,
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'model': model,
        'interpretation': 'elastic' if elasticity < -1 else 'inelastic'
    }


def optimal_price(elasticity: float, cost_per_unit: float) -> float:
    """
    Profit-maximizing price: P* = C * |ε| / (|ε| - 1).
    Returns NaN if demand is inelastic (elasticity >= -1).
    """
    if elasticity >= -1:
        return np.nan
    return cost_per_unit * abs(elasticity) / (abs(elasticity) - 1)


def predict_revenue_at_price(price: float, base_sales: float, base_price: float, elasticity: float) -> float:
    """Predict revenue at a given price using elasticity model."""
    q_base = base_sales / base_price
    q_pred = q_base * (price / base_price) ** elasticity
    return price * q_pred


def revenue_curve(base_sales: float, base_price: float, elasticity: float,
                  cost: float, price_range: tuple = (1, 15)) -> pd.DataFrame:
    """Generate revenue and profit curves over price range."""
    prices = np.linspace(price_range[0], price_range[1], 100)
    revenues = [predict_revenue_at_price(p, base_sales, base_price, elasticity) for p in prices]
    q_base = base_sales / base_price
    quantities = [q_base * (p / base_price) ** elasticity for p in prices]
    profits = [(p - cost) * q for p, q in zip(prices, quantities)]

    return pd.DataFrame({
        'Price': prices,
        'Revenue': revenues,
        'Profit': profits,
        'Quantity': quantities
    })


def seasonal_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly seasonal factors."""
    monthly = df.groupby('Month')['Weekly_Sales'].agg(['mean', 'std', 'count'])
    monthly['seasonal_factor'] = monthly['mean'] / df['Weekly_Sales'].mean()
    return monthly


def store_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Store-level summary statistics."""
    return df.groupby('Store').agg({
        'Weekly_Sales': ['mean', 'std', 'sum'],
        'Price': 'mean',
        'Holiday_Flag': 'sum'
    }).round(2)


def run_prototype():
    """Run the full Dynamic Pricing prototype."""
    print("=" * 60)
    print("  DYNAMIC PRICING OPTIMIZATION SYSTEM - PROTOTYPE")
    print("=" * 60)

    df = load_data()
    df = add_price_column(df)
    print(f"\n[1] Loaded {len(df):,} rows | Stores: {df['Store'].nunique()} | Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    result = estimate_price_elasticity(df)
    print(f"\n[2] Price Elasticity: {result['elasticity']:.3f} ({result['interpretation']}) | R²: {result['r2']:.3f}")

    base_sales = df['Weekly_Sales'].median()
    base_price = df['Price'].median()
    cost = 2.5
    p_opt = optimal_price(result['elasticity'], cost)

    print(f"\n[3] Current median price: ${base_price:.2f} | Base weekly sales: ${base_sales:,.0f}")
    if not np.isnan(p_opt):
        rev_current = predict_revenue_at_price(base_price, base_sales, base_price, result['elasticity'])
        rev_opt = predict_revenue_at_price(p_opt, base_sales, base_price, result['elasticity'])
        print(f"    Optimal price: ${p_opt:.2f} (cost=${cost})")
        print(f"    Revenue: ${rev_current:,.0f} -> ${rev_opt:,.0f} ({100*(rev_opt/rev_current-1):.1f}% change)")
    else:
        print("    Demand is inelastic - consider premium pricing.")

    seasonal = seasonal_trends(df).sort_values('seasonal_factor', ascending=False)
    print("\n[4] Top seasonal months:", list(seasonal.head(3).index))

    return {'data': df, 'elasticity_result': result, 'optimal_price': p_opt, 'seasonal': seasonal}


if __name__ == "__main__":
    run_prototype()
