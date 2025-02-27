README file

1. Project Title

- Sundial Stick Time Prediction

2. Overview: 

- This project demonstrates how to estimate the time of day based on the length and angle of the shadow cast by a vertical stick (like a simple sundial). We use a machine learning approach to build a model that inputs (shadow_length, shadow_angle) and outputs the predicted time_of_day (in decimal hours).

3. Files and Structure

.
├── README.md                             # This documentation
├── sundial_time_prediction.py            # Main Python script 
├── sundial_astral_data.csv               # dataset 

1. sundial_time_prediction.py

- Contains the end-to-end workflow:
  1. Data generation using the Astral library (Location, Date Range, Time Intervals).
  2. Computing sun elevation/azimuth → deriving shadow length & angle.
  3. Saving the dataset to CSV.
  4. Training a Random Forest model.
  5. Performing cross-validation and final evaluation (MAE, R²).

2. sundial_astral_data.csv

- The CSV file created by the script, containing columns:

  - date, time_str, time_of_day, shadow_length, shadow_angle

4. Detailed Approach

4.1 Problem Statement

- A vertical stick is placed perpendicular to the ground, forming a sundial. The shadow length and angle shift throughout the day as the sun moves. Our goal is to build an AI/ML model that, given these two measurements, predicts the time of day.

4.2 Data Generation or Collection

-  A: Real Data

  1. Place a stick (known height) in a sunny location.
  2. Using a camera, capture images at regular intervals (e.g., every 15–30 minutes).
  3. Measure shadow length (distance from stick base to tip of shadow) and angle (relative to North or another fixed reference).
  4. Record the actual time (HH:MM) for each measurement.
  5. Create a CSV: (time_of_day, shadow_length, shadow_angle).
   B: Synthetic Data with Astral

  1. Choose a location (latitude, longitude, timezone) and a date range.
  2. Use Astral to compute sun elevation (sun_elev_deg) and azimuth (sun_azim_deg) at each time interval.
  3. Calculate shadow_length = stick_height / tan(elevation_radians).
  4. Use sun_azim_deg as shadow_angle.
  5. Store time_of_day as decimal hours (e.g., 10:15 → 10.25).
  6. Optionally add small random noise to mimic real measurements.

4.3 Model and Training

1. Features:
   - shadow_length (float)
   - shadow_angle (float)
2. Label:
   - time_of_day (float, e.g., 6.00 for 6:00 AM, 13.50 for 1:30 PM)
3. Machine Learning Algorithm:
   - RandomForestRegressor (from Scikit-Learn): Chosen for its ability to handle nonlinear relationships and provide good performance out-of-the-box.
4. Evaluation Metrics:
   - Mean Absolute Error (MAE) in hours—how many hours off, on average, the predictions are.
   - R² Score (coefficient of determination)—how well predictions fit compared to a baseline.

4.4 Training Procedure

1. Train/Test Split:
   - We split ~80% of the data for training, 20% for testing.
2. Cross-Validation:
   - (Optional) 5-fold cross-validation is performed to check how stable the model is across different splits.
3. Model Fitting:
   - The model is trained on (shadow_length, shadow_angle) → time_of_day using a random forest with n_estimators=100.
4. Testing:
   - We predict time_of_day on unseen test data and compute MAE, R².

4.5 Results

- Synthetic Example (Astral-based):
   - MAE: Typically between 0.02–0.10 hours (~1–6 minutes) if data is clean.
   - R²: Often close to 1.0 with sufficiently large, noise-free synthetic data.

4.6 Future Enhancements

1. Collect Real Camera Data
   - Replace or supplement synthetic data with real measurements to see how the model fares in real conditions (noise, partial cloud cover, etc.).
2. Use More Locations / Seasons
   - Generate or capture data from different latitudes or months (e.g., winter vs. summer) for broader generalization.
3. Automate Shadow Measurement
   - Use computer vision (OpenCV) to detect the stick base and shadow tip in images. Then feed measurements to the model in real time.

5. Usage Instructions
  1. Install Requirements

    - pip install astral scikit-learn pandas
  2. Run the Script

    - python sundial_time_prediction.py
    
    - This will generate sundial_astral_data.csv, train the model, and output performance metrics.
  3. Check Output
    - Look for final MAE and R² in the terminal.

6. Cross Validation with Actual Time

- By comparing the model’s predicted time vs. actual time in the dataset, we confirm accuracy.
- If you want to do k-fold cross-validation, it’s demonstrated in the script (cross_val_score) to ensure robust evaluation.

7. File Contents in ZIP
Include the following in submission:

1. README.md (this file)
2. sundial_time_prediction.py (the code)
3. sundial_astral_data.csv (generated dataset)
4. real-world photos 

End of README