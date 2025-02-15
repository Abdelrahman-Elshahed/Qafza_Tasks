import io
import pandas as pd
import requests
import logging

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_api(*args, **kwargs):
    """
    Template for loading data from API
    """
    url = 'https://raw.githubusercontent.com/Abdelrahman-Elshahed/Qafza_Tasks/refs/heads/main/Task%202/listings.csv'
    response = requests.get(url)
    
    # Debug response
    logging.info(f"API Response status: {response.status_code}")
    
    # Load data
    df = pd.read_csv(io.StringIO(response.text), sep=',')
    
    # Debug dataframe
    logging.info(f"Loaded DataFrame columns: {df.columns.tolist()}")
    logging.info(f"DataFrame shape: {df.shape}")
    logging.info(f"DataFrame sample:\n{df.head()}")
    
    return df

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
