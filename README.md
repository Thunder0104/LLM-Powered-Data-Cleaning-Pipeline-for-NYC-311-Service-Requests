# LLM-Powered-Data-Cleaning-Pipeline-for-NYC-311-Service-Requests

## Overview

This project processes and analyzes data, computes key metrics, and visualizes results using a scatterplot over New York City boundaries.

It consists of the following main scripts:

- **`main.py`** – Core data processing and analysis pipeline  
- **`extra_metrics.py`** – Computes additional performance and error metrics  
- **`nyc_bound.py`** – Generates a scatterplot within NYC geographic boundaries

---

## Prerequisites

- **Python 3.7+**

### Required Python Packages

Install dependencies using:

```bash
pip install pandas numpy matplotlib geopandas shapely scikit-learn
```

---

## Project Structure

```
/your-project-folder
│
├── main.py
├── extra_metrics.py
├── nyc_bound.py
├── projectdata.csv
└── README.md

```

---

## Usage

### Run the Main Pipeline

```bash
python main.py
```

This script ingests raw data, performs data cleaning, modeling, and saves the primary outputs.

---

### Compute Extra Metrics

```bash
python extra_metrics.py
```

Calculates additional performance metrics and error analysis based on the outputs from `main.py`.

---

### Generate NYC Scatterplot

```bash
python nyc_bound.py
```

- Loads NYC boundary shapefile using GeoPandas  
- Plots scatter data points within the boundary  
- Saves output as **`nyc_scatterplot.png`**

---

## Notes

- Ensure that all **input data files** are placed in the correct paths expected by each script.  
- Modify **configuration paths** inside scripts if needed.
