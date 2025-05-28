from flask import Flask, render_template, request, redirect, url_for
import os
from simulate_irrigation import simulate_moisture_and_status
from weather_prediction import predict_weather
from predict_crop_disease import predict_disease
import uuid
import pandas as pd

app = Flask(__name__)

# ==== ROUTES ====

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/irrigation", methods=["GET", "POST"])
def irrigation():
    crop_df = pd.read_csv("data/crop_moisture_thresholds.csv")
    crop_df.columns = crop_df.columns.str.strip()
    crop_list = sorted(crop_df["Crop"].dropna().unique().tolist())

    print("🧪 Loaded Crops:", crop_list)

    if request.method == "POST":
        selected_crop = request.form.get("crop")
        # ✅ Redirect to avoid resubmission popup
        return redirect(url_for("irrigation", selected_crop=selected_crop))

    # Handle GET request after redirect (PRG pattern)
    selected_crop = request.args.get("selected_crop")
    if selected_crop:
        status, crop_name, min_moisture, max_moisture, graph_path = simulate_moisture_and_status(selected_crop)
        if not crop_name:
            error = f"Crop '{selected_crop}' not found in dataset."
            return render_template("irrigation.html", error=error, crop_list=crop_list)

        return render_template("irrigation.html",
                               crop=crop_name,
                               status=status,
                               min_moisture=min_moisture,
                               max_moisture=max_moisture,
                               graph_path=graph_path,
                               crop_list=crop_list)

    # Initial page load
    return render_template("irrigation.html", crop_list=crop_list)

weather_results = {}

@app.route("/weather", methods=["GET", "POST"])
def weather():
    if request.method == "POST":
        file = request.files.get("weather_csv")
        if file:
            unique_id = str(uuid.uuid4())
            filename = f"uploaded_weather_{unique_id}.csv"
            filepath = os.path.join("static", filename)
            file.save(filepath)

            # Run prediction logic
            image_path, mae, mse = predict_weather(filepath, unique_id)

            # Store for result display
            weather_results[unique_id] = {
                "image": image_path,
                "mae": round(mae, 2),
                "mse": round(mse, 2)
            }


            # Redirect to GET-based result page
            return redirect(url_for("weather_result", uid=unique_id))

    return render_template("weather.html")


@app.route("/weather/result")
def weather_result():
    uid = request.args.get("uid")
    if not uid or uid not in weather_results:
        return redirect(url_for("weather"))

    result = weather_results[uid]
    return render_template("weather_result.html",
                           image_path=result["image"],
                           mae=result["mae"],
                           mse=result["mse"])



@app.route("/disease", methods=["GET", "POST"])
def disease():
    if request.method == "POST":
        file = request.files.get("leaf_image")
        if file:
            uid = str(uuid.uuid4())
            image_filename = f"leaf_{uid}.jpg"
            image_path = os.path.join("static", image_filename)
            file.save(image_path)

            # Save filename in session or redirect via UID
            return redirect(url_for("disease_result", uid=uid))
    return render_template("disease.html")

@app.route("/disease/result")
def disease_result():
    uid = request.args.get("uid")
    if not uid:
        return redirect(url_for("disease"))

    image_filename = f"leaf_{uid}.jpg"
    image_path = os.path.join("static", image_filename)

    if not os.path.exists(image_path):
        return redirect(url_for("disease"))

    predicted_class, confidence, remedy = predict_disease(image_path)

    return render_template("disease.html",
                           result=True,
                           image_path=image_path,
                           predicted_class=predicted_class,
                           confidence=confidence,
                           remedy=remedy)


# ==== MAIN ====
if __name__ == "__main__":
    app.run(debug=True)
