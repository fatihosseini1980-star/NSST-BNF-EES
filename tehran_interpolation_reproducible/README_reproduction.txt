TEHRAN PM2.5 AQI — INTERPOLATED 2024 TSAVP FIGURE

Purpose: An alternative continuous cartographic rendering of EXISTING saved model
predictions for 22 representative interior points within Tehran regions per
month (264 point predictions), not an additional model fit.

SOURCE (user-supplied Supplementary_Code_and_Results):
  data/tehran_regions_wgs84.geojson
  data/monthly_supplied.csv (station location metadata only for the map)
  results/tehran_regional_map/regional_point_predictions.csv
  results/tehran_vecchia15/tehran_predictions.csv

COPY tehran_interpolated_map.py into the supplementary root and run:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tehran_interpolated_map.py

Dependencies: numpy, pandas, scipy, geopandas, shapely, pyproj, matplotlib.

Method: reproject to metric UTM EPSG:32639, fit an EXACT thin-plate-spline
RBF (smoothing=0) to 22 model-predicted means separately for each month,
evaluate on regular grid, mask outside union of Tehran's municipal polygons.
Single global color scale across all months. Month-specific monitored stations
are shown in black. 22 region boundaries appear for geographic orientation,
but polygon fill is INTERPOLATED across boundaries, not a constant regional value.

SCIENCE/INTERPRETATION:
  - This is spatial interpolation of the means predicted by the existing model.
  - It is NOT a direct dense-grid neural-field posterior predictive calculation,
    an areal average, or a map of actual station measurements.
  - No 2024 observed AQI values or model re-training were used for interpolation.
  - RBF can overshoot near the boundary; this is not an uncertainty map.
  - Internal validation: compare RBF at the 339 held-out station locations
    with the separately saved model predictions (not observed AQI outcomes):
    MAE=1.04559 AQI, RMSE=1.29201 AQI; IDW p=2 MAE=1.82547 AQI.
  - Maximum control-point reconstruction difference <= 8.6e-14 AQI.

Files:
 Tehran_2024_interpolated_TSAVP_12_panels.png — 320 dpi map
 Tehran_2024_interpolated_TSAVP_12_panels.pdf — publication export
 regional_anchor_predictions_with_utm.csv — 264 saved control predictions
 interpolated_grid_predictions_2024.csv.gz — masked-grid values, 12 months
 interpolation_vs_station_predictions.csv — validation comparison
 interpolation_diagnostics.json — audit metrics and caveats
 tehran_interpolated_map.py — fully reproducible source (included in ZIP)
