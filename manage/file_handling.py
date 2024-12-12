######################################## PACKAGES ########################################
import os # for path etc
import chardet # for separator identification - relative to csv files processing
import pandas as pd # for Dataframes manipulation
from detect_delimiter import detect
import logging  # Added for logging

######################################## LOAD DATAFRAME FROM FILES ########################################

def create_folders(config):
    """
    Create folders based on the configuration.

    Parameters:
        config (dict): Configuration dictionary containing folder names.

    Returns:
        None
    """
    folder_names = config.get("folder_names", {})
    
    for folder_name in folder_names.values():
        folder_path = os.path.join(os.getcwd(), folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            

def list_files_in_directory(directory, file_extensions=None):
    """
    Lists files with specified extensions in a directory.

    Parameters:
    - directory (str): The path to the directory.
    - file_extensions (list of str, optional): List of file extensions to filter. Default is None.

    Returns:
    - list of str: List of file paths that match the specified extensions.

    If no file_extensions are provided, all files in the directory are listed.
    """
    if file_extensions is None:
        file_extensions = []

    files = []
    for filename in os.listdir(directory):
        if filename.endswith(tuple(file_extensions)):
            files.append(os.path.join(directory, filename))

    return files


def detect_encoding(file_path, num_lines=100):
    """
    Detects the encoding of a file using chardet.

    Parameters:
    - file_path (str): The path to the file.
    - num_lines (int): The number of lines to read from the file for encoding detection. Default is 100.

    Returns:
    - str: The detected encoding of the file.

    Note:
    The function attempts to detect the encoding of the file by analyzing a portion of its content. 
    It reads the specified number of lines from the file and uses the chardet library to determine the encoding.
    """
    detector = chardet.UniversalDetector(should_rename_legacy=True)
    detector.reset()
    with open(file_path, 'rb') as rawdata:
        for _ in range(num_lines):
            line = rawdata.readline()
            if not line:
                break  # Stop if end of file is reached
            detector.feed(line)
        detector.close()  # Close the detector after processing
        result = detector.result

        if result['encoding'] is None:
            print("None......")
            result = chardet.detect(rawdata.read())
        else:
            result2 = chardet.detect(rawdata.read())
            if result2['confidence'] > result['confidence']:
                print("\n\n",file_path,"----encoding ---> ", result2, result, " ----\n")
                return result2['encoding']

    print("\n\n",file_path,"----encoding ---> ", result, " ----\n")
    return result['encoding']


def load_csv_with_separator(file_path, encoding, num_lines=30):
    """
    Load a CSV file with automatic delimiter detection.

    Parameters:
    - file_path (str): The path to the CSV file.
    - encoding (str): The encoding of the file.
    - num_lines (int): The number of lines to read for delimiter detection. Default is 10.

    Returns:
    - pd.DataFrame: The loaded DataFrame.
    """
    try:
        separators_count = dict()

        with open(file_path, 'rb') as file:
            for _ in range(num_lines):
                line = file.readline()
                if not line:
                    break  # Stop if end of file is reached
                line = line.decode(encoding=encoding)
                separators = detect(line, default=',')
                if separators not in separators_count:
                    separators_count[separators] = 1
                else:
                    separators_count[separators] += 1

        print("Séparateurs éligibles: ",separators_count)
        most_represented_separator = max(separators_count, key=separators_count.get)
        print("-----> Most represented separator:", most_represented_separator)

        df = pd.read_csv(file_path, sep=most_represented_separator, encoding=encoding)
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file '{file_path}': {str(e)}")
        return None


def load_dataframe(file_path, encoding='utf-8'):
    """
    Loads a DataFrame from a file.

    Parameters:
    - file_path (str): The path to the file to load.
    - encoding (str, optional): The encoding of the file. Default is 'utf-8'.

    Returns:
    - DataFrame or None: The loaded DataFrame if successful, otherwise None.

    This function attempts to load a DataFrame from the specified file. It first detects the file extension
    to determine the file type. For CSV files, it uses a custom function to handle loading with the specified
    encoding. For Excel files, it uses pandas read_excel method. If the file format is unsupported or an error
    occurs during loading, it logs an error message and returns None.
    """
    # Identify the file extension
    file_extension = os.path.splitext(file_path)[-1].lower()
    #detect the encoding
    print("\n -------------------------------- IN ",file_path)
    encoding = detect_encoding(file_path)

    if file_extension == '.csv':
        try:
            df = load_csv_with_separator(file_path, encoding, num_lines=30)
        except Exception as e:
            logging.error(f"Error reading CSV file '{file_path}': {str(e)}")
            return None
        if df is not None:
            print("DataFrame loaded successfully.")
            return df
        else:
            logging.error(f"Error loading CSV file '{file_path}': DataFrame is None")
            return None
    elif file_extension in ['.xls', '.xlsx']:
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
        except Exception as e:
            logging.error(f"Error reading Excel file '{file_path}': {str(e)}")
            return None
        if df is not None:
            print("DataFrame loaded successfully.")
            return df
        else:
            logging.error(f"Error loading Excel file '{file_path}': DataFrame is None")
            return None
    else:
        logging.error(f"Unsupported file format: {file_extension}")
        return None

######################################## FUNCTIONS FOR SAVING FILES ########################################

def auto_name(input_file_path,config, output_extension='csv',descriptive=False):
    """
    Generate an output file path based on the input file path.

    Parameters:
        input_file_path (str): The path to the input file.
        output_extension (str): The extension of the output file. Default is 'csv'.
        descriptive (bool): Whether to use a descriptive folder. Default is False.

    Returns:
        str: The output file path.

    Raises:
        None
    """
    DATA_FOLDER = config.get("folder_names", {}).get("DATA_FOLDER")
    ENRICHMENT_FOLDER = config.get("folder_names", {}).get("ENRICHMENT_FOLDER")
    OUTLIERS_FOLDER = config.get("folder_names", {}).get("OUTLIERS_FOLDER")
    DESCRIPTIVE_FOLDER = config.get("folder_names", {}).get("DESCRIPTIVE_FOLDER")

    # Check if the 'data' folder exists, and create it if not
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    # Check if the 'enrichment' folder exists, and create it if not
    if not os.path.exists(ENRICHMENT_FOLDER):
        os.makedirs(ENRICHMENT_FOLDER)
    # Check if the 'outliers' folder exists in 'data' folder, and create it if not
    if not os.path.exists(OUTLIERS_FOLDER):
        os.makedirs(OUTLIERS_FOLDER)
    # Get the file name and extension from the input path
    file_name, file_extension = os.path.splitext(os.path.basename(input_file_path))
    if descriptive:
        if not os.path.exists(DESCRIPTIVE_FOLDER):
            os.makedirs(DESCRIPTIVE_FOLDER)
        output_file_path = os.path.join(DESCRIPTIVE_FOLDER,f"{file_name}_descriptive.{output_extension}")
    else:
        # Construct the output file path with the same name and a different extension or suffix
        output_file_path = os.path.join(OUTLIERS_FOLDER,f"{file_name}_outliers.{output_extension}")
    # Return the output file path
    return output_file_path
