import pandas as pd
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def simulate_moisture_and_status(crop_name):
    # Load threshold CSV
    threshold_df = pd.read_csv("data/crop_moisture_thresholds.csv")

    # Match crop (case insensitive)
    matched_crop = threshold_df[threshold_df["Crop"].str.lower() == crop_name.lower()]
    if matched_crop.empty:
        return None, None, None, None, None

    row = matched_crop.iloc[0]
    selected_crop = row["Crop"]
    min_moisture = row["Min_Moisture"]
    max_moisture = row["Max_Moisture"]

    # Choose soil moisture file
    csv_files = ["dry_soil.csv", "balanced_soil.csv", "wet_soil.csv", "random_soil.csv", "soil_moisture_data.csv"]
    selected_csv = random.choice(csv_files)
    df = pd.read_csv(selected_csv)

    latest_moisture = df["Soil_Moisture"].iloc[-1]
    if latest_moisture < min_moisture:
        irrigation_status = "Irrigation: ON (Water Needed)"
        color = "red"
    else:
        irrigation_status = "Irrigation: OFF (No Water Needed)"
        color = "green"

    # Plot and save to static folder
    plt.figure(figsize=(8, 5))
    plt.plot(df["Time"], df["Soil_Moisture"], marker='o', linestyle='-', color='b', label="Soil Moisture")
    plt.axhline(y=min_moisture, color='gray', linestyle='dotted', label=f"Min Threshold ({min_moisture}%)", alpha=0.7)
    plt.axhline(y=max_moisture, color='gray', linestyle='dotted', label=f"Max Threshold ({max_moisture}%)", alpha=0.7)

    plt.grid(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.text(df["Time"].iloc[-2], max_moisture + 5, irrigation_status,
             color=color, fontsize=12, fontweight="bold", ha="center",
             bbox=dict(facecolor='white', alpha=0.6, edgecolor=color))

    plt.xlabel("Time")
    plt.ylabel("Soil Moisture (%)")
    plt.title(f"Soil Moisture Levels for {selected_crop}")
    plt.legend(title=f"Crop: {selected_crop}")
    plt.tight_layout()

    # Save the plot
    image_path = os.path.join("static", "irrigation_plot.png")
    plt.savefig(image_path)
    plt.close()

    return irrigation_status, selected_crop, min_moisture, max_moisture, "irrigation_plot.png"

