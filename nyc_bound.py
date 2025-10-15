import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from geodatasets import get_path  # Using geodatasets to load the NYBB file

# ------------------------------------------------------
# Load Final Corrected Anomalies with Geolocation Data
# ------------------------------------------------------
final_df = pd.read_csv("final_corrected_anomalies.csv")

# Filter out records with missing geolocation data.
final_df = final_df.dropna(subset=["Validated_Latitude", "Validated_Longitude"])

# Convert the latitude and longitude into a geometry column.
geometry = [Point(xy) for xy in zip(final_df["Validated_Longitude"], final_df["Validated_Latitude"])]
geo_df = gpd.GeoDataFrame(final_df, geometry=geometry, crs="EPSG:4326")

# ------------------------------------------------------
# Load NYC Borough Boundaries using geodatasets (NYBB dataset)
# ------------------------------------------------------
nybb_path = get_path('nybb')
nybb = gpd.read_file(nybb_path)

# The NYBB dataset is typically in EPSG:2263; convert to EPSG:4326 for consistency.
nybb = nybb.to_crs(epsg=4326)

# ------------------------------------------------------
# Plotting: Overlay Anomalies on NYC Borough Boundaries
# ------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

# Plot borough boundaries.
nybb.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.5)

# Plot anomaly points on top.
geo_df.plot(ax=ax, markersize=30, color="red", marker="o", label="Anomaly Geolocations")

# Add labels for each borough using the centroid of each polygon.
for idx, row in nybb.iterrows():
    ax.annotate(row['BoroName'],
                xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                horizontalalignment='center',
                fontsize=10,
                fontweight='bold')

# Set title and other aesthetics.
plt.title("Geolocated Anomalies Overlaid on NYC Borough Boundaries", fontsize=14)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.legend()
plt.show()
