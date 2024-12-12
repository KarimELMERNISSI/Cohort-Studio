import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import sys # some system functions
import pandas as pd # for Dataframes manipulation
import json # deal with json data structures
import logging  # Added for logging
import manage.file_handling as mf
import explore.statistics as es
import enrich.custom_metrics_and_filters as ecm
import enrich.conditional_transformations as ect
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
######################################## PARAMETERS ########################################
# Specify the path to the sample Excel file in the sample_data folder
# csv_name = 'cardiateam_data_extraction_inserm_2023-12-20_14-11-19.csv'
# Specify the directory path

LOG_FILE = 'execution_log.txt'
CONFIG_FILE = 'config.json'

######################################## STATS REPORT FUNCTIONS ########################################

def remove_string_values(df):
    """
    Remove string values from the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to process.
    
    Returns:
    - pd.DataFrame: The DataFrame with string values removed.
    """
    # Identify columns with non-numeric data types
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    
    # Drop columns with non-numeric data types
    df_numeric = df.drop(columns=non_numeric_columns)
    
    return df_numeric



def generate_insights(hypothesis_test_results, method_name, group_labels):
    """
    Generate insights and interpretations based on the analysis.

    Parameters:
    - hypothesis_test_results (dict): Dictionary containing the results of hypothesis tests.
    - method_name (str): Name of the hypothesis testing method used (e.g., "Wilcoxon", "t-test", "Fisher").
    - group_labels (list): List of labels for the groups being compared.

    Returns:
    - str: Insights and interpretations.
    """
    insights = f"Here are some insights based on the analysis using {method_name} test:\n"
    
    # Check for statistically significant difference based on p-value
    if hypothesis_test_results['p_value'] < 0.05:
        insights += f"- There is a statistically significant difference between {group_labels[0]} and {group_labels[1]}.\n"
    else:
        insights += f"- There is no statistically significant difference between {group_labels[0]} and {group_labels[1]}.\n"
    
    # Check for significant results in other tests if applicable
    
    return insights


######################################## STATS REPORT ########################################
#tbd
def process_test_and_generate_reports(df, df_statistics, test, test_info, config, title, multi_index, test_folder, f_name_no_ext, footnotes, output_format):
    logging.info(f"\n\nProcessing test: {test}")
    print(test)
    result = es.process_test(df=df, test_name=test, test_info=test_info, config=config, multi_index=multi_index)
    if result is not None:
        result_df = result
        print("\n RESULT DF #####################\n", result_df)
        result_dftemp = pd.merge(df_statistics, result_df, left_index=True, right_index=True, how='right')
        print("\n RESULT MERGE #####################\n", result_dftemp)
        test_filename = os.path.join(test_folder, f"{f_name_no_ext}_table_{test}")
        
        if output_format in ('tex','pdf','png'):
            es.produce_latex_table(df=result_dftemp, columns=None, filename=f"{test_filename}.tex", unicode_latex_mapping=config.get("unicode_latex_mapping"), title=title, footnotes=footnotes, has_multi_index=multi_index)
            if output_format in ('pdf','png'):
                pdf_file_path = es.compile_latex_to_pdf(f"{f_name_no_ext}_table_{test}.tex", output_folder=test_folder)
                if output_format == 'png':
                    if pdf_file_path:
                        print(pdf_file_path)
                        es.pdf_to_png(pdf_file_path)
        elif output_format == 'xlsx':
            es.produce_xlsx_table(df=result_dftemp, columns=None, filename=f"{test_filename}.xlsx", title=title, float_nb_digits=2, has_multi_index=multi_index)
        elif output_format == 'html':
            es.produce_html_table(df=result_dftemp, columns=None, filename=f"{test_filename}.html", title=title, float_nb_digits=2, has_multi_index=multi_index)
        else:
            logging.warning(f"Incorrect output format. Skipping.")
    else:
        logging.warning(f"Test '{test}' returned None. Skipping.")


def generate_statistical_report(df, output_pdf):
    """
    Generate a statistical report in PDF format based on the provided DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - output_pdf (str): The filename for the output PDF file.

    Returns:
    - None
    """
    # Descriptive statistics
    descriptive_stats = df.describe()
    descriptive_stats_table = Table(descriptive_stats.values.tolist(), 
                                    colWidths=[100]*len(descriptive_stats.columns), 
                                    rowHeights=30)

    # Check for missing values
    missing_values = df.isnull().sum().to_frame()
    missing_values_table = Table(missing_values.values.tolist(), 
                                 colWidths=[100]*len(missing_values.columns), 
                                 rowHeights=30)
    

    # Compile the report
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    elements = []
    elements.append(descriptive_stats_table)
    elements.append(missing_values_table)

    # Write the PDF file
    doc.build(elements)

######################################## FUNCTIONS FOR LOGGING CONFIGS ########################################

def setup_logging():
    """
    Setup logging to write to both console and a log file
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

######################################## MAIN FUNCTION ######################################## 

def main():
    setup_logging()

    try:
        # Load configuration file for masks
        with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
            config = json.load(file)
        ## CREATE DEFAULT FOLDERS IF DO NOT EXIST
        
        mf.create_folders(config)

        test_folder = config.get("folder_names", {}).get("HYPOTHESIS_TESTING_FOLDER")
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        #folder_path = os.path.abspath(test_folder)
        ## LOAD DATAFRAMES FROM FILES
        # list all CSV files in a directory
        #csv_files = list_files_in_directory(DATA_FOLDER, file_extensions=('.csv',))
        # list all Excel files (xls or xlsx) in a directory
        #excel_files = list_files_in_directory(DATA_FOLDER, file_extensions=('.xls', '.xlsx'))
        # list all data files (csv, xls or xlsx) in a directory
        data_files = mf.list_files_in_directory(config.get("folder_names", {}).get("DATA_FOLDER"), file_extensions=('.csv', '.xlsx')) # '.xls', '.xlsx','.csv'

        for data_file in data_files:
            df = mf.load_dataframe(data_file)
            print(f"FILE DF {data_file} is OK")
            # Extract the file name from the path
            f_name = os.path.basename(data_file)
            # Remove the file extension
            f_name_no_ext = os.path.splitext(f_name)[0]
            
            if df is None:
                logging.warning(f"Skipping processing for data file '{data_file}' due to loading errors.")
                continue
            
            # Logging information about the loaded data file
            logging.info(f"Processing data file: {data_file}")
            print("\n-------------")
            units = ["mmol/L", "g/L"]
            eligible_columns= ect.retrieve_transform_column_names(config) #["Triglycerides","Triglycerides.1","Triglycerides.2","LDLc","LDLc.1","LDLc.2","HDLc","HDLc.1","HDLc.2","Cholesterol total","Cholesterol total.1","Cholesterol total.2"]
            if eligible_columns is not None:
                unit_columns = ect.find_unit_columns(df, units,eligible_columns)
                gl_columns = []
                mmoll_columns = []
                if unit_columns:
                    print(f"The unit columns containing values in {units} are: {unit_columns}")
                else:
                    print("No columns with unique values matching the specified units found.")

                for unit_col in unit_columns:
                    measure_columns = ect.find_measures_with_unit_column(eligible_columns, unit_col)
                    if measure_columns:
                        print(f"The columns with unit column name '{unit_col}' are: {measure_columns}")
                    else:
                        print(f"No columns with unit column name '{unit_col}' found.")   

                    for unit_value in units:
                        counts = ect.count_non_null_values(df,unit_col, unit_value)
                        print(f"Non-null value counts when unit column value is {unit_value}:")
                        print(counts) 
                        if unit_value == "mmol/L":
                            mmoll_columns.append(counts.idxmax())
                        else:
                            gl_columns.append(counts.idxmax())
                print("GL: ",gl_columns)
                print("mmol/L: ", mmoll_columns)

                columns_map = ect.global_column_mapping(unit_columns=unit_columns,unit_column_suffix='2',gl_columns=gl_columns,gl_column_suffix='1',mmoll_columns=mmoll_columns,mmoll_column_suffix=None)
                print("\n\n.................\n",columns_map)
                # Rename columns
                df.rename(columns=columns_map, inplace=True)
                
                #Compute missing "mmol/L", "g/L" measures based on the config file...   
                ect.transform_column(df, config)

            ecm.generate_computed_column(df, config)

            exclude_cols = []
            for family_mask_name, family_mask_config in config.get("mask_families", {}).items():   
                #masks_zip = ec.zip_masks(df, config.get("mask_families", {}), family_mask_name)
                masks_zip = ecm.zip_masks(df, family_mask_config)
                print("\n********** family_mask:")
                print(family_mask_name)
                print(family_mask_config)
                ## OUTLIERS MATRIX CONSTRUCTION
                # Step 1: Create a new DataFrame with the same index as df
                masks_df = pd.DataFrame(index=df.index)

                # Step 2: Add columns to the new DataFrame for each mask
                for mask_name, mask_values in masks_zip:
                    print("\n****************************** mask_name",mask_name, ", mask_content:", mask_values)
                    masks_df[mask_name] = mask_values

                ## COMPUTE EXP-CRITERIA-BASED-GROUPS COLUMNS 
                # Custom mask list & tag
                df[f"{family_mask_name}_list"] = masks_df.apply(ecm.tag_masks, axis=1)
                exclude_cols.append(f"{family_mask_name}_list")
                df[f"{family_mask_name}_tag"] = masks_df.any(axis=1)
                exclude_cols.append(f"{family_mask_name}_tag")
                df[f"{family_mask_name}_tag_conj"] = masks_df.all(axis=1)
                exclude_cols.append(f"{family_mask_name}_tag_conj")
            print("\n********** exclude_cols:",exclude_cols)

            multi_index = True
            glob = 'GLOBAL'
            #latex_expression = es.produce_statistics_table_v2(df, output_format='latex')
            df_statistics = es.get_statistics_dataframe(df, nb_top_categories=3, exclude_columns=None, multi_index=multi_index, super_column=glob)
            df_cum_statistics = df_statistics.copy()

            print(df_statistics.columns.tolist())
            filename = os.path.join(test_folder, f"{f_name_no_ext}_table")
            if multi_index:
                selected_cols = [(glob, 'count'), (glob, 'fill_percentage'),  (glob, 'variable_type'), (glob, 'mean'), (glob, 'min'), (glob, '25%'), (glob, '50%'), (glob, '75%'), (glob, 'max'), (glob, 'std')]
            else:
                selected_cols = ['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'count', 'fill_percentage', 'variable_type']# ,'w1_p_value','t2_p_value','chi2_p_value','kw_p_value'
            #df_statistics = es.create_simple_multiindex_dataframe(df_statistics)
            footnotes = {
                "Statistics Report": "mon super titre!",
                "Smoking": "This column indicates the smoking status of the subjects",
                "subject\_id": "The unique identifier for each subject in the study.",
                "25\%": "25th percentile value.",
                "fill\_percentage": "Percentage of non-null values filled in the column",
                "0.314": "This value represents something specific in the data"
            }
            if config.get("statistical_tests", {}):
                for test, test_info in config.get("statistical_tests", {}).items():
                    logging.info(f"\n\nProcessing test: {test}")
                    print(test)
                    result = es.process_test(df=df, test_name=test, test_info=test_info, config=config, multi_index=multi_index)
                    if result is not None:
                        result_df = result #pd.DataFrame.from_dict(result).transpose()
                        print("\n RESULT DF #####################\n", result_df)
                        # Merge result_df with df_statistics based on their indices   
                        print("\n DFSTAT #####################\n", df_statistics)
                        result_dftemp = pd.merge(df_statistics, result_df, left_index=True, right_index=True, how='right') # one by test
                        print("\n RESULT MERGE #####################\n", result_dftemp)
                        test_filename = os.path.join(test_folder, f"{f_name_no_ext}_table_{test}")
                        title = f"Statistical Report for Test {test}"
                        es.produce_latex_table(df=result_dftemp,columns=None,filename=f"{test_filename}.tex",unicode_latex_mapping=config.get("unicode_latex_mapping"),title=title,footnotes=footnotes,has_multi_index=multi_index)
                        pdf_file_path = es.compile_latex_to_pdf(f"{f_name_no_ext}_table_{test}.tex", output_folder=test_folder)
                        if pdf_file_path:
                            print(pdf_file_path)
                            es.pdf_to_png(pdf_file_path)
                        print("\n --------------PAF41\n") 
                        es.produce_xlsx_table(df=result_dftemp,columns=None,filename=f"{test_filename}.xlsx",title=title,float_nb_digits=2,has_multi_index=multi_index)
                        es.produce_xlsx_table(df=result_dftemp,columns=None,filename=f"{filename}.xlsx",title=title,float_nb_digits=2,has_multi_index=multi_index)
                        print("\n --------------PAF42\n") 
                        es.produce_html_table(df=result_dftemp,columns=None,filename=f"{test_filename}.html",title=title,float_nb_digits=2,has_multi_index=multi_index)
                        print("\n --------------PAF43\n")  
                        df_cum_statistics = pd.merge(df_cum_statistics, result_df, left_index=True, right_index=True, how='left') # concatenate all tests
                        print("\n DF MERGE #####################\n", df_statistics)
                    else:
                        logging.warning(f"Test '{test}' returned None. Skipping.")
            print("PDF File Path: ",f"{filename}.tex")
            print("COLUMNS: ",df_statistics.columns.tolist())
            selected_cols_set = set(selected_cols)
            gen_cols = ['mean', 'std', 'min', '25%', 'median', '75%', 'max', 'mode', 'nb_modalities', 'p_value','statistic']
            if multi_index:
                for gen_col in gen_cols:
                    selected_cols_set.update([(col, subcol) for (col, subcol) in df_cum_statistics.columns if subcol.endswith(gen_col)])
                selected_cols = list(selected_cols_set)
                print("selected_cols: ",selected_cols)
            else:
                for gen_col in gen_cols:
                    selected_cols_set.update([col for col in df_cum_statistics.columns if col.endswith(gen_col)])
                selected_cols = list(selected_cols_set)
                print("selected_cols: ",selected_cols)

            es.produce_latex_table(df=df_cum_statistics,columns=selected_cols,filename=f"{filename}.tex",unicode_latex_mapping=config.get("unicode_latex_mapping"),title="Global Statistical Report",footnotes=footnotes,has_multi_index=multi_index)
            es.produce_xlsx_table(df=df_cum_statistics,columns=selected_cols,filename=f"{filename}.xlsx",title="Global Statistical Report",float_nb_digits=2,has_multi_index=multi_index)
            es.produce_html_table(df=df_cum_statistics,columns=selected_cols,filename=f"{filename}.html",title="Global Statistical Report",float_nb_digits=2,has_multi_index=multi_index)
            pdf_file_path = es.compile_latex_to_pdf(f"{f_name_no_ext}_table.tex", output_folder=test_folder)

            if pdf_file_path:
                print(pdf_file_path)
                es.pdf_to_png(pdf_file_path)

    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    sys.exit()


