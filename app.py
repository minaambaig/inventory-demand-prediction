from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "inventory_demand_model.joblib"
DATA_PATH = BASE_DIR / "data" / "processed_inventory_data.csv"


# ---------------------------------------------------------
# DISPLAY NAMES
# ---------------------------------------------------------

STORE_NAME_MAP = {
    "S001": "Hyderabad Central Store",
    "S002": "Gachibowli Retail Store",
    "S003": "Banjara Hills Store",
    "S004": "Secunderabad Retail Store",
    "S005": "Madhapur Super Store",
}


PRODUCT_NAME_MAP = {
    "P0001": "Wireless Headphones",
    "P0002": "Smartphone",
    "P0003": "Laptop",
    "P0004": "Smart Television",
    "P0005": "Bluetooth Speaker",
    "P0013": "Refrigerator",
    "P0014": "Washing Machine",
    "P0015": "Air Conditioner",
    "P0020": "Smart Watch",
}


# Default prices shown automatically when a product is selected
PRODUCT_PRICE_MAP = {
    "P0001": 3000.0,     # Wireless Headphones
    "P0002": 30000.0,    # Smartphone
    "P0003": 60000.0,    # Laptop
    "P0004": 45000.0,    # Smart Television
    "P0005": 5000.0,     # Bluetooth Speaker
    "P0013": 55000.0,    # Refrigerator
    "P0014": 35000.0,    # Washing Machine
    "P0015": 40000.0,    # Air Conditioner
    "P0020": 8000.0,     # Smart Watch
}


# Only these products will appear in the website
ELECTRONICS_PRODUCT_IDS = [
    "P0001",
    "P0002",
    "P0003",
    "P0004",
    "P0005",
    "P0013",
    "P0014",
    "P0015",
    "P0020",
]


st.set_page_config(
    page_title="Inventory Demand Prediction",
    page_icon="📦",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"])
    data["Store ID"] = data["Store ID"].astype(str)
    data["Product ID"] = data["Product ID"].astype(str)

    return data


try:
    model_package = load_model()
    model = model_package["model"]
    feature_columns = model_package["features"]
    df = load_data()

except Exception as error:
    st.error(str(error))
    st.stop()


st.title("📦 Inventory Demand Prediction System")

st.write(
    "Predict demand for electronic products and receive "
    "an inventory reorder recommendation."
)


# ---------------------------------------------------------
# FILTER ELECTRONICS DATA
# ---------------------------------------------------------

electronics_df = df[
    (df["Category"].astype(str).str.strip().str.lower() == "electronics")
    & (df["Product ID"].isin(ELECTRONICS_PRODUCT_IDS))
].copy()


if electronics_df.empty:
    st.error(
        "No Electronics records were found in the processed dataset."
    )
    st.stop()


# ---------------------------------------------------------
# GET IDS AND OPTIONS
# ---------------------------------------------------------

store_ids = sorted(df["Store ID"].unique())

available_product_ids = set(
    electronics_df["Product ID"].unique()
)

product_ids = [
    product_id
    for product_id in ELECTRONICS_PRODUCT_IDS
    if product_id in available_product_ids
]

weather_options = sorted(
    electronics_df["Weather Condition"].dropna().unique()
)

season_options = sorted(
    electronics_df["Seasonality"].dropna().unique()
)

discount_options = sorted(
    electronics_df["Discount"].dropna().unique()
)


# ---------------------------------------------------------
# CREATE STORE DISPLAY MAP
# ---------------------------------------------------------

store_display_map = {}

for index, store_id_value in enumerate(store_ids, start=1):
    display_name = STORE_NAME_MAP.get(
        store_id_value,
        f"Retail Store {index}",
    )

    store_display_map[display_name] = store_id_value


# ---------------------------------------------------------
# CREATE ELECTRONICS PRODUCT DISPLAY MAP
# ---------------------------------------------------------

product_display_map = {}

for index, product_id_value in enumerate(product_ids, start=1):
    display_name = PRODUCT_NAME_MAP.get(
        product_id_value,
        f"Electronic Product {index}",
    )

    product_display_map[display_name] = product_id_value


if not product_display_map:
    st.error(
        "No mapped Electronics products are available."
    )
    st.stop()


# ---------------------------------------------------------
# WEBSITE LAYOUT
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:
    selected_store_name = st.selectbox(
        "Store",
        options=list(store_display_map.keys()),
    )

    store_id = store_display_map[selected_store_name]

    selected_product_name = st.selectbox(
        "Electronic Product",
        options=list(product_display_map.keys()),
    )

    product_id = product_display_map[selected_product_name]


# ---------------------------------------------------------
# AUTOMATIC DEFAULT PRICES
# ---------------------------------------------------------

default_price = PRODUCT_PRICE_MAP.get(
    product_id,
    5000.0,
)

default_competitor_price = round(
    default_price * 1.05,
    2,
)


# Category is fixed internally
category = "Electronics"


# Region remains hidden but is still supplied to the model
store_records = electronics_df[
    electronics_df["Store ID"] == store_id
]

if not store_records.empty:
    region = store_records["Region"].mode().iloc[0]
else:
    region = df["Region"].mode().iloc[0]


with col2:
    prediction_date = st.date_input(
        "Prediction Date"
    )

    inventory_level = st.number_input(
        "Current Inventory Level",
        min_value=0,
        value=200,
        step=1,
    )

    price = st.number_input(
        "Product Price (₹)",
        min_value=0.0,
        value=float(default_price),
        step=100.0,
        key=f"price_{product_id}",
    )

    discount = st.selectbox(
        "Discount (%)",
        discount_options,
    )


with col3:
    competitor_price = st.number_input(
        "Competitor Price (₹)",
        min_value=0.0,
        value=float(default_competitor_price),
        step=100.0,
        key=f"competitor_price_{product_id}",
    )

    weather = st.selectbox(
        "Weather Condition",
        weather_options,
    )

    holiday = st.selectbox(
        "Holiday",
        options=[0, 1],
        format_func=lambda value: (
            "Yes" if value == 1 else "No"
        ),
    )

    promotion = st.selectbox(
        "Promotion",
        options=[0, 1],
        format_func=lambda value: (
            "Yes" if value == 1 else "No"
        ),
    )

    seasonality = st.selectbox(
        "Season",
        season_options,
    )


# ---------------------------------------------------------
# COMBINE HOLIDAY AND PROMOTION
# The current model expects one Holiday/Promotion feature.
# ---------------------------------------------------------

holiday_or_promotion = (
    1 if holiday == 1 or promotion == 1 else 0
)


# ---------------------------------------------------------
# GET HISTORICAL ELECTRONICS RECORDS
# ---------------------------------------------------------

historical_rows = electronics_df[
    (electronics_df["Store ID"] == store_id)
    & (electronics_df["Product ID"] == product_id)
].sort_values("Date")


if historical_rows.empty:
    st.error(
        "No Electronics historical records were found for "
        "this store and product combination."
    )
    st.stop()


latest = historical_rows.iloc[-1]
prediction_date = pd.Timestamp(prediction_date)


# ---------------------------------------------------------
# CREATE MODEL INPUT
# ---------------------------------------------------------

input_data = pd.DataFrame({
    "Store ID": [store_id],
    "Product ID": [product_id],
    "Category": [category],
    "Region": [region],
    "Inventory Level": [inventory_level],
    "Price": [price],
    "Discount": [discount],
    "Weather Condition": [weather],
    "Holiday/Promotion": [holiday_or_promotion],
    "Competitor Pricing": [competitor_price],
    "Seasonality": [seasonality],
    "Year": [prediction_date.year],
    "Month": [prediction_date.month],
    "Day": [prediction_date.day],
    "DayOfWeek": [prediction_date.dayofweek],
    "WeekOfYear": [
        int(prediction_date.isocalendar().week)
    ],
    "Quarter": [prediction_date.quarter],
    "IsWeekend": [
        1 if prediction_date.dayofweek in [5, 6] else 0
    ],
    "PriceDifference": [
        competitor_price - price
    ],
    "DiscountedPrice": [
        price * (1 - discount / 100)
    ],
    "Lag_1": [latest["Lag_1"]],
    "Lag_7": [latest["Lag_7"]],
    "Lag_14": [latest["Lag_14"]],
    "Lag_30": [latest["Lag_30"]],
    "RollingMean_7": [latest["RollingMean_7"]],
    "RollingMean_30": [latest["RollingMean_30"]],
})


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if st.button(
    "Predict Inventory Demand",
    type="primary",
):
    try:
        predicted_demand = model.predict(
            input_data[feature_columns]
        )[0]

        predicted_demand = max(
            0,
            round(predicted_demand),
        )

        safety_stock = round(
            predicted_demand * 0.15
        )

        required_stock = (
            predicted_demand + safety_stock
        )

        reorder_quantity = max(
            0,
            required_stock - inventory_level,
        )

        remaining_stock = max(
            0,
            inventory_level - predicted_demand,
        )

        if predicted_demand < 100:
            demand_level = "Low"

        elif predicted_demand < 250:
            demand_level = "Medium"

        else:
            demand_level = "High"


        st.divider()
        st.subheader("Prediction Result")

        st.write(
            f"**Store:** {selected_store_name}"
        )

        st.write(
            f"**Electronic Product:** {selected_product_name}"
        )

        st.write(
            f"**Product Price:** ₹{price:,.2f}"
        )

        st.write(
            f"**Competitor Price:** ₹{competitor_price:,.2f}"
        )


        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric(
            "Predicted Demand",
            f"{predicted_demand} units",
        )

        metric2.metric(
            "Demand Level",
            demand_level,
        )

        metric3.metric(
            "Safety Stock",
            f"{safety_stock} units",
        )

        metric4.metric(
            "Recommended Order",
            f"{reorder_quantity} units",
        )


        metric5, metric6 = st.columns(2)

        metric5.metric(
            "Current Inventory",
            f"{inventory_level} units",
        )

        metric6.metric(
            "Expected Remaining Stock",
            f"{remaining_stock} units",
        )


        if reorder_quantity > 0:
            st.warning(
                f"Stock may be insufficient for "
                f"{selected_product_name}. Order approximately "
                f"{reorder_quantity} additional units."
            )

        else:
            st.success(
                f"Current inventory for "
                f"{selected_product_name} should be sufficient "
                f"for the predicted demand."
            )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )
        