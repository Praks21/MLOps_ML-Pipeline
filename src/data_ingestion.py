# it will take data from the define source like aws s3

import pandas as pd  # as we have to work with data frame
import os            # used to make directory where we will push our data
from sklearn.model_selection import train_test_split # prevent data leakage
import logging  
import yaml

# ensure the "logs" directory exists
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# making console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# creating file handler
log_file_path = os.path.join(log_dir,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# creating formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# above to lines or handler , now we will put that output in logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

# creating function to load the data
def load_data(data_url: str) -> pd.DataFrame:
    """Loading Data from a CSV file also we're performing exception handling below"""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)   # this will print in the terimal or in the file
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the dataA %s", e)
        raise

# creating exception handling func to thandle the preprocess function or data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3','Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

# saving the data in a folder which we have made
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str)-> None:
    """Save the train and test datasets."""
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occured while saving the data: %s', e)
        raise

# defineing the main function
def main():
    try:
        # test_size=0.2
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        
        data_path='https://raw.githubusercontent.com/vikashishere/YT-MLOPS-Complete-ML-Pipeline/refs/heads/main/experiments/spam.csv'
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error(' Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()
