# Universities Data Pipeline Project


This project implements an ETL pipeline to extract, transform, and load university data for Germany and Egypt from the Hipolabs University API. The project is organized into Python scripts and notebooks for processing and exploring the data.

## Table of Contents

- [Project Overview](#project-overview)
- [Clone the Repository](#clone-the-repository)
- [Run the ETL Scripts](#run-the-etl-scripts)
- [Explore Data](#explore-data)


## Project Overview
This project performs the following steps:
- Extract: Fetches university data from the API based on country filters.
- Transform: Processes the data to flatten nested fields and filter based on specific criteria (e.g., universities in Berlin).
- Load: Saves the processed data into SQLite databases for further analysis


## Clone the Repository

To start using this project, first, clone the repository:

```bash
git clone https://github.com/Abdelrahman-Elshahed/Qafza_Tasks.git
```

## Run the ETL Scripts
- File: **`etl_GERMANY.py`**
- Description: Fetches universities in Germany, filters universities in Berlin, and saves the data to **`GER_UNIs.db`**.
- Run the script:
```bash
python etl_GERMANY.py
```
![GER_Screenshot](https://github.com/user-attachments/assets/01252aff-6582-48a3-b510-dd4b3afd59ca)
![GER_Screenshot2](https://github.com/user-attachments/assets/a948ca53-801f-4d43-9928-c0693763cf2c)


## Explore Data
### Use **`Explore_df_GER.ipynb`** to explore the processed data in **`GER_UNIs.db`**. The notebook includes:
- Table listing.
- Data preview (head).
- Summary statistics.
