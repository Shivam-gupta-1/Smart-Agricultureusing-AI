def predict_weather(filepath, uid):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import pandas as pd
    import numpy as np
    import os

    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    df["DayOfYear"] = df["Date"].dt.dayofyear

    # Slight randomness to simulate real-world data
    df["Humidity (%)"] += np.random.uniform(-5, 5, df.shape[0])
    df["Rainfall (mm)"] += np.random.uniform(-2, 2, df.shape[0])
    df["Wind Speed (km/h)"] += np.random.uniform(-1, 1, df.shape[0])
    df["Temperature (°C)"] += np.random.uniform(-2, 2, df.shape[0])

    X = df[["DayOfYear", "Humidity (%)", "Rainfall (mm)", "Wind Speed (km/h)"]]
    y = df["Temperature (°C)"]

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(df_shuffled[X.columns])

    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly)

    X_train, X_test, y_train, y_test = train_test_split(X_poly_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Plot and save
    plt.figure(figsize=(8, 5))
    errors = np.abs(y_test - y_pred)
    labels = ["Accurate" if err <= 2 else "Inaccurate" for err in errors]

    for a, p, label in zip(y_test, y_pred, labels):
        color = "green" if label == "Accurate" else "red"
        plt.scatter(a, p, color=color, alpha=0.6, label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="Perfect Prediction")
    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.title("Actual vs Predicted Temperature")
    plt.legend()

    # Save plot
    image_filename = f"weather_plot_{uid}.png"
    image_path = os.path.join("static", image_filename)
    plt.savefig(image_path)
    plt.close()

    return image_filename, mae, mse

