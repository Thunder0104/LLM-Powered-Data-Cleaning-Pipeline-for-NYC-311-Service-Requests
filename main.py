import os
import time
import logging
import re
import pandas as pd
import requests
import matplotlib.pyplot as plt
from rapidfuzz import fuzz, process
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from cachetools import cached, TTLCache

# ======================================================
# Setup Logging
# ======================================================
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ======================================================
# Global Configurations
# ======================================================
DATA_FILE = "projectdata.csv"
REQUIRED_COLUMNS = ['Borough', 'Incident Address', 'Latitude', 'Longitude']

# Specify which LLM engine to use: "groq" or "gpt4"
LLM_ENGINE = "groq"  # using Groq for LLM corrections

# Hardcoded API keys (for testing only—replace with your actual keys)
openai_api_key = "your-hardcoded-openai-api-key"      # if using GPT as fallback
groq_api_key = ""
google_api_key = ""

# ======================================================
# Cache for Geolocation API calls (to reduce duplicate requests)
# ======================================================
geocode_cache = TTLCache(maxsize=1000, ttl=86400)  # Cache up to 1000 addresses for 24 hours

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
                logging.warning(f"Google Geocoding API response: {data['status']} for address: {address}")
        except Exception as e:
            logging.warning(f"Google Geocoding API attempt {attempt+1} failed: {e}")
            time.sleep(1)
    return None, None, None

# ======================================================
# Stage 1: Data Loading and Anomaly Detection
# ======================================================
start_total = time.time()

try:
    df = pd.read_csv(DATA_FILE, low_memory=False)
    logging.info(f"Loaded {len(df)} records from {DATA_FILE}.")
except FileNotFoundError:
    logging.error(f"File {DATA_FILE} not found.")
    raise
except Exception as e:
    logging.error(f"Error reading {DATA_FILE}: {e}")
    raise

for col in REQUIRED_COLUMNS:
    if col not in df.columns:
        logging.error(f"Missing required column: {col}")
        raise Exception("Missing required columns in dataset.")

for col in ['Complaint Type', 'Incident Zip']:
    if col not in df.columns:
        df[col] = ''

valid_boroughs = ['MANHATTAN', 'BRONX', 'BROOKLYN', 'QUEENS', 'STATEN ISLAND']

df['Borough'] = df['Borough'].astype(str).str.upper()
df['Borough_Valid'] = df['Borough'].apply(lambda b: b in valid_boroughs)
df['Address_Valid'] = df['Incident Address'].notna() & (df['Incident Address'].str.strip().str.len() > 5)

def is_valid_coord(lat, lon):
    try:
        lat, lon = float(lat), float(lon)
        return 40 <= lat <= 41 and -75 <= lon <= -73
    except Exception:
        return False

df['Geo_Valid'] = df.apply(lambda row: is_valid_coord(row['Latitude'], row['Longitude']), axis=1)
df['Record_Valid'] = df['Borough_Valid'] & df['Address_Valid'] & df['Geo_Valid']

clean_df = df[df['Record_Valid']].copy()
anomaly_df = df[~df['Record_Valid']].copy()

clean_df.to_csv("clean_records.csv", index=False)
anomaly_df.to_csv("detected_anomalies.csv", index=False)
logging.info(f"Clean records: {len(clean_df)}, Anomalies: {len(anomaly_df)}")

# ======================================================
# Stage 2: LLM-Based Correction Using Groq
# ======================================================
llm_start = time.time()

def generate_prompt(row):
    address = row.get('Incident Address', '')
    borough = row.get('Borough', '')
    lat = row.get('Latitude', '')
    lon = row.get('Longitude', '')
    complaint = row.get('Complaint Type', '')
    zipcode = row.get('Incident Zip', '')
    
    if pd.isna(address) or len(str(address).strip()) < 5 or pd.isna(borough) or (str(borough).strip() not in valid_boroughs):
        prompt = f"""
            The following NYC 311 record has a missing or invalid address and/or Borough.
            Context:
            - Borough: {borough if borough and str(borough).strip() in valid_boroughs else "Not specified or invalid"}
            - Latitude: {lat}
            - Longitude: {lon}
            - Complaint Type: {complaint}
            - Zip Code: {zipcode}

            Based on the above context, please provide the full, standardized address including street number, street name, borough, and zip code.
            Also return the correct Borough.
            ONLY return one line in this exact format:
            Corrected Borough: <value>; Corrected Address: <Street Number> <Street Name>, <Borough>, NY <Zip Code>
            Do NOT include any extra explanation.
            """
    else:
        prompt = f"""
            Correct the following NYC 311 record fields:
            Valid Boroughs: Manhattan, Bronx, Brooklyn, Queens, Staten Island
            Original Borough: "{borough}"
            Original Address: "{address}"
            Please provide corrections in this exact format:
            Corrected Borough: <value>; Corrected Address: <Street Number> <Street Name>, <Borough>, NY <Zip Code>
            """
    return prompt.strip()

def get_groq_correction(prompt, retries=3, delay=2):
    try:
        from groq import Groq
    except ImportError as ie:
        logging.error("Groq package not installed. Please install via pip: pip install groq")
        raise ie
        
    client = Groq(api_key=groq_api_key)
    
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=150,
                top_p=1,
                stream=True,
                stop=None,
            )
            result_text = ""
            for chunk in completion:
                part = chunk.choices[0].delta.content or ""
                print(">>", part, flush=True)
                result_text += part
            return result_text.strip()
        except Exception as e:
            logging.warning(f"Groq API call attempt {attempt+1}/{retries} failed: {e}")
            time.sleep(delay)
    logging.error("Groq API call failed after multiple attempts.")
    return None

def get_gpt_correction(prompt, retries=3, delay=2):
    for attempt in range(retries):
        try:
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-3.5",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.warning(f"GPT API call attempt {attempt+1}/{retries} failed: {e}")
            time.sleep(delay)
    logging.error("GPT API call failed after multiple attempts.")
    return None

def get_llm_correction(prompt, retries=3, delay=2):
    if LLM_ENGINE == "gpt":
        return get_groq_correction(prompt, retries, delay)
    else:
        return get_gpt_correction(prompt, retries, delay)

corrected_boroughs = []
corrected_addresses = []
llm_success_count = 0

logging.info("Starting LLM-based corrections using engine: " + LLM_ENGINE)
for index, row in anomaly_df.iterrows():
    prompt = generate_prompt(row)
    correction = get_llm_correction(prompt)
    logging.info(f"RAW LLM Response for Record {index+1}:\n{correction}")
    
    # Use regex to capture the full corrected address and borough.
    boro = None
    addr = None
    if correction:
        match_boro = re.search(r"Corrected Borough:\s*(.*?)(;|$)", correction, re.IGNORECASE|re.DOTALL)
        if match_boro:
            boro = match_boro.group(1).strip()
        match_addr = re.search(r"Corrected Address:\s*(.*)", correction, re.IGNORECASE|re.DOTALL)
        if match_addr:
            # Remove any trailing semicolons or extra text.
            addr = match_addr.group(1).split(";")[0].strip()
    else:
        logging.error(f"Record {index+1}: No correction obtained.")
    
    if not boro:
        boro = row['Borough']
    
    if addr is not None and addr != "":
        llm_success_count += 1
    
    corrected_boroughs.append(boro)
    corrected_addresses.append(addr)
    logging.info(f"Record {index+1}: Corrected Borough: {boro}, Corrected Address: {addr}")
    time.sleep(1)  # Respect API rate limits

anomaly_df["LLM_Corrected_Borough"] = corrected_boroughs
anomaly_df["LLM_Corrected_Address"] = corrected_addresses
anomaly_df.to_csv("llm_corrected_anomalies.csv", index=False)
llm_end = time.time()
llm_time = llm_end - llm_start
logging.info(f"LLM-based corrections complete in {llm_time:.2f} seconds; saved to 'llm_corrected_anomalies.csv'.")

# ======================================================
# Stage 3: External Geolocation Validation with Google Maps
# ======================================================
geo_start = time.time()

validated_latitudes = []
validated_longitudes = []
standardized_addresses = []
geo_success_count = 0

logging.info("Starting Google Maps geolocation validation...")

for index, row in anomaly_df.iterrows():
    address = row.get("LLM_Corrected_Address") or row["Incident Address"]
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
anomaly_df.to_csv("final_corrected_anomalies.csv", index=False)
geo_end = time.time()
geo_time = geo_end - geo_start
logging.info(f"Google Maps geolocation validation complete in {geo_time:.2f} seconds; saved to 'final_corrected_anomalies.csv'.")

end_total = time.time()
total_time = end_total - start_total

# ======================================================
# Stage 4: Enhanced Data Visualization
# ======================================================
labels = ['Clean Records', 'Anomalies']
sizes = [clean_df.shape[0], anomaly_df.shape[0]]
colors = ['#66b3ff', '#ff9999']
explode = (0.05, 0)

plt.figure(figsize=(6,6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Distribution of Clean vs. Anomalous Records')
plt.axis('equal')
plt.show()

valid_geo = anomaly_df.dropna(subset=["Validated_Latitude", "Validated_Longitude"])
plt.figure(figsize=(8,6))
plt.scatter(valid_geo["Validated_Longitude"], valid_geo["Validated_Latitude"], alpha=0.7, s=10, c='green')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Scatter Plot of Validated Geolocations (Anomalous Records)")
plt.grid(True)
plt.show()

total_records = df.shape[0]
clean_count = clean_df.shape[0]
anomaly_count = anomaly_df.shape[0]
llm_success_rate = (llm_success_count / anomaly_count) * 100 if anomaly_count else 0
geo_success_rate = (geo_success_count / anomaly_count) * 100 if anomaly_count else 0

print("\n========== Performance Metrics ==========")
print(f"Total records: {total_records}")
print(f"Clean records: {clean_count} ({(clean_count/total_records)*100:.1f}%)")
print(f"Anomalies detected: {anomaly_count} ({(anomaly_count/total_records)*100:.1f}%)")
print(f"LLM Correction Success: {llm_success_count} / {anomaly_count} anomalies corrected ({llm_success_rate:.1f}%)")
print(f"Google Geolocation Success: {geo_success_count} / {anomaly_count} anomalies validated ({geo_success_rate:.1f}%)")
print(f"Time for LLM Corrections: {llm_time:.2f} seconds")
print(f"Time for Geolocation Validation: {geo_time:.2f} seconds")
print(f"Total Processing Time: {total_time/60:.2f} minutes")
print("=========================================\n")
