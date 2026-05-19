- # TCIF-LPUE Data Preprocessing & Formatting Guidelines

  This document explains the preprocessing logic and the final data schema. 

  ---

  ## 1. Raw Data Sources

  Our dataset is constructed from three authoritative meteorological sources:
  1. **CMA Best Track Data (BST):** Provides the TC center coordinates (latitude/longitude) and the true Maximum Sustained Wind (MSW) at 6-hour intervals. Link: https://tcdata.typhoon.org.cn/zjljsjj_en.html
  2. **GridSat-B1 Satellite Imagery:** Provides Geostationary Infrared (IR) and Water Vapor (WV) observations. Link: https://www.ncei.noaa.gov/products/gridded-geostationary-brightness-temperature
  3. **ERA5 Reanalysis:** Provides crucial atmospheric physical factors, including Sea Surface Temperature (SST), Relative Humidity at 600hPa (RH_600), and Vertical Wind Shear derived from U/V components at 200hPa and 850hPa. Link: https://cds.climate.copernicus.eu/

  ---

  ## 2. Preprocessing & Alignment Strategy

  For each TC event, we extract sliding windows comprising **5 historical time steps** (at $t-24h, t-18h, t-12h, t-6h, t$) to predict the intensity at $t+24h$.

  ### Temporal Alignment
  All satellite imagery (originally 3h resolution) and ERA5 data (originally 1h resolution) are interpolated and subsampled to match the **6-hour intervals** of the CMA BST records.

  ### Spatial Alignment Strategy
  * **Satellite Imagery:** Cropped to a $400 \times 400$ spatial grid centered on the TC center. Combining the 2 selected spectral channels (IR and WV) across 5 historical time steps yields a **10-channel** spatial tensor.
  * **Physical Factors:** Originally cropped to an $80 \times 80$ grid. We employ a Bilinear Spatial Alignment strategy, upsampling the ERA5 grids to $400 \times 400$ during the preprocessing phase. Combining 3 physical factors across 5 historical time steps yields a **15-channel** spatial tensor.

  ---

  ## 3. The Output Data Schema (`.npy` Format)

  The processed data for each valid sliding window is saved as a single Python dictionary in a `.npy` file. This format allows for highly efficient I/O operations during PyTorch `DataLoader` iterations. 

  Each `.npy` file contains the following keys and dimensions:

  | Key            | Shape            | Data Type | Description                                                  |
  | :------------- | :--------------- | :-------- | :----------------------------------------------------------- |
  | `tc_img`       | `(10, 400, 400)` | `float32` | The spatially aligned satellite imagery. (2 channels [IR, WV] $\times$ 5 time steps). Values are Min-Max normalized. |
  | `tc_era`       | `(15, 400, 400)` | `float32` | The spatially upsampled ERA5 physical factors (RH_600, SST, Wind Shear). (3 factors $\times$ 5 time steps). |
  | `tc_intensity` | `(5,)`           | `float32` | Historical intensity metrics, containing the true MSW $(m/s)$ at the 5 historical time steps. |
  | `target`       | `(1,)`           | `float32` | The ground-truth MSW $(m/s)$ at $t+24h$, which serves as the prediction target. |

  
