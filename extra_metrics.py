import time
import logging
import re
import pandas as pd
import requests
import matplotlib.pyplot as plt
from cachetools import cached, TTLCache
from geopy.exc import GeocoderTimedOut
from rapidfuzz import fuzz

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Google Maps API key
google_api_key = "" # your google api key

# Setup a cache for geolocation lookups
geocode_cache = TTLCache(maxsize=1000, ttl=86400)

@cached(geocode_cache)
def validate_with_google_cached(address, google_api_key, retries=3):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": google_api_key}
    for attempt in range(retries):
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data['status'] == "OK":
                result = data['results'][0]
                standardized_address = result.get("formatted_address")
                location = result.get("geometry", {}).get("location", {})
                lat = location.get("lat")
                lng = location.get("lng")
                return lat, lng, standardized_address
            else:
                logging.warning(f"Google API response: {data['status']} for address: {address}")
        except Exception as e:
            logging.warning(f"Google API attempt {attempt+1} failed: {e}")
            time.sleep(1)
    return None, None, None

# ------------------------------------------------------
# Stage 3: Geolocation Validation
# ------------------------------------------------------
geo_start = time.time()
anomaly_df = pd.read_csv("llm_corrected_anomalies.csv")

validated_latitudes = []
validated_longitudes = []
standardized_addresses = []
geo_success_count = 0

logging.info("Starting Google Maps geolocation validation...")

for index, row in anomaly_df.iterrows():
    raw_address = row.get("LLM_Corrected_Address") or row.get("Incident Address")
    address = "" if raw_address is None else str(raw_address)
    if address and len(address.strip()) > 5:
        lat, lng, std_addr = validate_with_google_cached(address, google_api_key)
        validated_latitudes.append(lat)
        validated_longitudes.append(lng)
        standardized_addresses.append(std_addr)
        if lat is not None and lng is not None:
            geo_success_count += 1
        logging.info(f"Record {index+1}: Google validated lat: {lat}, lng: {lng}, standardized address: {std_addr}")
    else:
        validated_latitudes.append(None)
        validated_longitudes.append(None)
        standardized_addresses.append(None)
        logging.info(f"Record {index+1}: No valid address for geolocation lookup.")
    time.sleep(1)

anomaly_df["Validated_Latitude"] = validated_latitudes
anomaly_df["Validated_Longitude"] = validated_longitudes
anomaly_df["Standardized_Address"] = standardized_addresses

# ------------------------------------------------------
# Address Similarity
# ------------------------------------------------------
def compute_similarity(llm_addr, std_addr):
    if pd.isna(llm_addr) or pd.isna(std_addr):
        return None
    return fuzz.ratio(str(llm_addr).strip(), str(std_addr).strip())

anomaly_df["Address_Similarity"] = anomaly_df.apply(
    lambda row: compute_similarity(row.get("LLM_Corrected_Address"), row.get("Standardized_Address")),
    axis=1
)

valid_similarities = anomaly_df["Address_Similarity"].dropna().astype(float).tolist()
similarity_threshold = 50
if valid_similarities:
    avg_similarity = sum(valid_similarities) / len(valid_similarities)
    low_similarity_count = sum(1 for s in valid_similarities if s < similarity_threshold)
else:
    avg_similarity = None
    low_similarity_count = 0

anomaly_df.to_csv("final_corrected_anomalies.csv", index=False)
geo_end = time.time()
geo_time = geo_end - geo_start

# ------------------------------------------------------
# Stage 4: Metrics + Visualizations
# ------------------------------------------------------
df = pd.read_csv("projectdata.csv")
clean_df = pd.read_csv("clean_records.csv")

total_records = df.shape[0]
clean_count = clean_df.shape[0]
anomaly_count = anomaly_df.shape[0]
llm_success_count = anomaly_df["LLM_Corrected_Address"].notna().sum()
llm_success_rate = (llm_success_count / anomaly_count) * 100 if anomaly_count else 0
geo_success_rate = (geo_success_count / anomaly_count) * 100 if anomaly_count else 0

print("\n========== Performance Metrics ==========")
print(f"Total records: {total_records}")
print(f"Clean records: {clean_count} ({(clean_count/total_records)*100:.1f}%)")
print(f"Anomalies detected: {anomaly_count} ({(anomaly_count/total_records)*100:.1f}%)")
print(f"LLM Correction Success: {llm_success_count} / {anomaly_count} anomalies corrected ({llm_success_rate:.1f}%)")
print(f"Google Geolocation Success: {geo_success_count} / {anomaly_count} anomalies validated ({geo_success_rate:.1f}%)")
print(f"Average Address Similarity: {avg_similarity:.1f}" if avg_similarity is not None else "Average Address Similarity: N/A")
print(f"Records with similarity below {similarity_threshold}: {low_similarity_count}")
print(f"Time for Geolocation Validation: {geo_time:.2f} seconds")
print("=========================================\n")

# Pie Chart
labels = ['Clean Records', 'Anomalies']
sizes = [clean_count, anomaly_count]
colors = ['#66b3ff', '#ff9999']
explode = (0.05, 0)

plt.figure(figsize=(6,6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Distribution of Clean vs. Anomalous Records')
plt.axis('equal')
plt.show()

# Scatter Plot
valid_geo = anomaly_df.dropna(subset=["Validated_Latitude", "Validated_Longitude"])
plt.figure(figsize=(8,6))
plt.scatter(valid_geo["Validated_Longitude"], valid_geo["Validated_Latitude"], alpha=0.7, s=10, c='green')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Scatter Plot of Validated Geolocations (Anomalous Records)")
plt.grid(True)
plt.show()
