"""
prepare_dataset.py
------------------
Membuat `dataset.csv` (data yang benar-benar dipakai dalam eksperimen TB1) dari
berkas mentah Kaggle "E-commerce Delivery & Shipping Data 2026" (lisensi CC0 1.0).

Langkah yang dilakukan:
1. Membaca CSV mentah dari data/raw/.
2. Memilih kolom identitas (order_id), 14 fitur masukan, dan target late_delivery.
3. Membuang kolom yang bocor (baru diketahui setelah/selama pengiriman):
   actual_delivery_days, delivery_delay_days, delivery_status, tracking_status,
   return_requested, return_reason, customer_rating, delivery_attempts.
   Kolom identitas/lokasi lain (customer_id, kota, negara, warehouse, order_date)
   juga tidak dipakai sebagai fitur.
4. TIDAK melakukan transformasi nilai apa pun (encoding, imputasi, standardisasi
   dilakukan di notebook dan hanya dipelajari dari data latih).

Jalankan:  python prepare_dataset.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "E-commerce_Delivery_Shipping_Data_2026.csv"
OUT_PATH = ROOT / "dataset.csv"

ID_COL = "order_id"
NUMERIC_FEATURES = [
    "product_weight_kg",
    "order_value_usd",
    "distance_km",
    "promised_delivery_days",
    "shipping_cost_usd",
    "warehouse_processing_hours",
]
CATEGORICAL_FEATURES = [
    "customer_segment",
    "product_category",
    "shipping_method",
    "carrier",
    "package_size",
    "payment_method",
    "order_priority",
    "weather_condition",
]
TARGET = "late_delivery"


def main() -> None:
    raw = pd.read_csv(RAW_PATH)
    cols = [ID_COL] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing_cols = set(cols) - set(raw.columns)
    if missing_cols:
        raise KeyError(f"Kolom tidak ditemukan pada data mentah: {missing_cols}")

    data = raw[cols].copy()
    data.to_csv(OUT_PATH, index=False)

    print(f"Data mentah   : {raw.shape[0]} baris x {raw.shape[1]} kolom")
    print(f"dataset.csv   : {data.shape[0]} baris x {data.shape[1]} kolom")
    print(f"Disimpan ke   : {OUT_PATH}")
    print(data[TARGET].value_counts().to_string())


if __name__ == "__main__":
    main()
