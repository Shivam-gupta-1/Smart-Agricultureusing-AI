# 🌾 Smart Agriculture AI System

This is a full-stack AI-based prototype system to support farmers with intelligent **crop irrigation**, **weather forecasting**, and **plant disease detection**, all integrated into a beautiful web dashboard using Flask.

---

## 🚀 Features

- **✅ Smart Irrigation System**
  - Uses crop-specific soil moisture thresholds.
  - Simulates soil moisture data and recommends whether to turn irrigation ON or OFF.
  - Dynamic graph generation per crop.

- **✅ Weather Prediction**
  - Predicts temperature using machine learning (Polynomial + Random Forest).
  - Accepts CSV input and visualizes predicted vs actual temperature.
  - Highlights accurate/inaccurate predictions using a color-coded graph.

- **✅ Crop Disease Detection**
  - Deep learning (CNN) model trained from scratch on PlantVillage dataset.
  - Upload a leaf image and get the predicted disease name with confidence score.
  - Auto-suggests remedies based on a curated remedy dataset.

- **✅ Web Dashboard (Multi-page Flask App)**
  - Fully responsive interface.
  - Navigation between Home, Irrigation, Weather, and Disease Detection.
  - Live graph/image outputs.
  - CSV/image cleanup after use.


