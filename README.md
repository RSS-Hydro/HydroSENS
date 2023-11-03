# HydroSENS
HydroSENS is an algorithm for the automatic hydrological characterization of natural and urban landscapes.
## Data
The required inputs include: 
- Shapefile of the intended study area in a projected coordinate system
- Soil Texture Classes (USDA System) dataset by Hengl (2018). The dataset is available online at: https://zenodo.org/records/2525817.
- Spectral library file (Data sourced from USGS)
- Curve Number lookup table 

The algorithm is currently designed to extract optical imagery and digital elevation models (DEMs) from Google Earth Engine (GEE)

## References

United States Geological Survey. (n.d.). USGS Spectral Library Version 7 [dataset]. https://doi.org/10.3133/ds1035

Hengl, T. (2018). Soil texture classes (USDA system) for 6 soil depths (0, 10, 30, 60, 100 and 200 cm) at 250 m (v0.2) [dataset]. Zenodo. https://doi.org/10.5281/zenodo.2525817

