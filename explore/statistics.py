import datetime
import logging
import os
import subprocess

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer

from utils.statistics_utils import sanitize_dataframe_for_arrow
#from statsmodels.multivariate.manova import MANOVA
# PENSER A REALISER UN LOG DES SELECTIONS ET TRANSFORMATIONS APPLIQUEES AUX DONNEES -- IDEE + D'INTEGRITE DANS LES DONNEES, RENDRE REPRODUCTIBLE TOUS LES BIAIS DUS A DES CHOIX ARBITRAIRE SUR LES DONNEES INITIALES. GENERER UN RAPPORT D'INTEGRITE : DATA TRANS ET SELECT SUMMARY
######################################## GENERATE A DESCRIPTIVE STATISTICS FILE ########################################

@st.cache_data
def get_statistics_dataframe(df, _analyzer=None, nb_top_categories=4, exclude_columns=None, multi_index=False, super_column='', qual_var_threshold=100, dataset_name=None, _db_manager=None):
    """
    Creates a new DataFrame with statistics for each column.

    Parameters:
    - df (pd.DataFrame): The DataFrame for which to calculate statistics.
    - nb_top_categories (int): The number of top categories to consider for categorical columns.
    - exclude_columns (list): A list of column names to exclude from the statistics.

    Returns:
    - pd.DataFrame: A new DataFrame containing statistics for each column.
    """
    # Exclude specified columns
    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors='ignore')

    stats_df = df.describe(include='all').transpose()  # Use describe to get statistics
    stats_df['std'] = df.std(numeric_only=True)

    # Drop the 'top' and 'freq' columns
    stats_df = stats_df.drop(['top', 'freq'], axis=1, errors='ignore')

    # Vectorized computation of distinct value counts
    col_uniques = df.nunique()

    # Identify categorical columns based on the number of distinct values
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool', 'string', 'str']).columns
    if exclude_columns:
        categorical_columns = [col for col in categorical_columns if col not in exclude_columns]

    for col in categorical_columns:
        if col_uniques.get(col, 0) < qual_var_threshold:  # Adjust the threshold as needed
            # Calculate top categories and their percentages for each categorical column
            value_counts = df[col].value_counts()
            percentage_counts = (value_counts / len(df)) * 100
            top_categories = percentage_counts.head(nb_top_categories)

            # Create new columns for top categories and their percentages
            for i, (value, representation) in enumerate(top_categories.items(), start=1):
                stats_df.at[col, f'top_{i}_value'] = str(value)
                stats_df.at[col, f'top_{i}_representation'] = representation
    
    stats_df['fill_percentage'] = (1 - df.isnull().mean()) * 100  # Calculate fill percentage
    stats_df['nb_modalities'] = col_uniques
    stats_df['variable_type'] = df.dtypes.astype(str)

    # Helper to safely convert Timestamp/Timedelta to string for Arrow compatibility
    def safe_str_conversion(val):
        if pd.api.types.is_scalar(val):
             if isinstance(val, (pd.Timestamp, pd.Timedelta, datetime.datetime, datetime.date, np.datetime64)):
                 return str(val)
        return val

    # Apply conversion to the entire stats DataFrame (map replaces deprecated applymap in pandas 3.0)
    stats_df = stats_df.map(safe_str_conversion)

    # Force conversion of potential mixed-type columns to string to prevent ArrowInvalid
    mixed_type_cols = ['mean', 'std', 'min', 'max', '25%', '50%', '75%']
    for col in mixed_type_cols:
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].astype(str)

    stats_df = sanitize_dataframe_for_arrow(stats_df)

    if multi_index:
        stats_df = create_multiindex_dataframe(result=stats_df.transpose(), super_column=super_column)
    
    return stats_df


def calculate_group_stats(group, group_name=None):
    """
    Calculate descriptive statistics for a group.

    Parameters:
    - group (pandas.Series): The group for which to calculate statistics.
    - group_name (str, Optional): The name of the group.

    Returns:
    - dict: Dictionary containing descriptive statistics with group name as prefix.
    """
    stats = {
        'mean': group.mean(),
        'std': group.std(),
        'min': group.min(),
        '25%': group.quantile(0.25),
        'median': group.median(),
        '75%': group.quantile(0.75),
        'max': group.max(),
        'mode': group.mode()[0] if not group.mode().empty else None
    }

    # Helper to safely convert Timestamp/Timedelta to string for Arrow compatibility
    def safe_str(val):
        if pd.api.types.is_scalar(val):
             if isinstance(val, (pd.Timestamp, pd.Timedelta, datetime.datetime, datetime.date, np.datetime64)):
                 return str(val)
        return val

    # Apply conversion
    safe_stats = {k: safe_str(v) for k, v in stats.items()}

    if group_name is not None:
        return {f'{group_name}_{k}': v for k, v in safe_stats.items()}
    else:
        print("##### IN GROUP_STAT", group_name)
        return safe_stats


def calculate_group_stats_for_multiple_groups(groups, group_names=None):
    """
    Calculate descriptive statistics for multiple groups.

    Parameters:
    - groups (list of pandas.Series): A list of groups for which to calculate statistics.
    - group_names (list, Optional): A list of group names. If None, generic names will be generated.

    Returns:
    - dict: Dictionary containing descriptive statistics for all groups.
    """
    print("\n####\n IN calculate_group_stats_for_multiple_groups \n####\n")
    stats_dict = {}
    if group_names is None:
        group_names = [f"Group_{i+1}" for i in range(len(groups))]  # Generate generic group names
    
    for group_name, group in zip(group_names, groups):
        print("################# IN GROUPS STATS--- ",group_name)
        group_stats = calculate_group_stats(group, group_name)
        stats_dict.update(group_stats)

    return stats_dict

######################################## CONVERT DATAFRAME TO LATEX-TABLE SCRIPT & ENRICH IT ########################################

# Define a function to apply different colors based on conditions

def color_cells(val):
    if val < 0.02:
        color = "#b7d1be"
    elif val >= 0.02 and val < 0.05:
        color = "#69eb88"
    else: 
        color = "#000000"
    return f"color: {color}; font-weight: bold;"


def format_ltx_p_value(val):
    """Custom formatter for p-values."""
    threshold = 1e-3
    if val < threshold:
        exponent = int(abs(np.floor(np.log10(val))))
        return f"$<10^{{{{-{exponent}}}}}$"
    else:
        return f"{val:.3f}"


def format_html_p_value(val):
    """Custom formatter for p-values."""
    threshold = 1e-3
    if val < threshold:
        exponent = int(abs(np.floor(np.log10(val))))
        return f"<10<sup>-{exponent}</sup>"
    else:
        return f"{val:.3f}"


def format_xlsx_p_value(val):
    """Custom formatter for p-values."""
    threshold = 1e-3
    if val < threshold:
        exponent = int(abs(np.floor(np.log10(val))))
        return f"<1e-{exponent}"
    else:
        return f"{val:.3f}"
    


################ PROD

def produce_latex_table(df, columns=None, filename="output.tex", unicode_latex_mapping=None, title=None, footnotes=None, float_nb_digits=2, has_multi_index=False):
    """
    Convert DataFrame to LaTeX table and write it to a .tex file.

    Parameters:
        df (DataFrame): The DataFrame containing data.
        columns (list, optional): The list of columns to include in the table. Defaults to None.
        filename (str, optional): The name of the .tex file to write. Defaults to "output.tex".
        unicode_latex_mapping (dict, optional): A dictionary mapping Unicode characters to LaTeX representations. Defaults to None.
        title (str, optional): The title of the table. Defaults to None.
        footnotes (dict, optional): A dictionary mapping column or index names to their associated footnotes. Defaults to None.
        float_nb_digits (int, optional): Number of digits to display for floating-point numbers. Defaults to 2.
    """
    # Error handling
    if df.empty:
        raise ValueError("DataFrame is empty")
    if not filename.endswith(".tex"):
        raise ValueError("Filename must have a .tex extension")
      
    # Unicode to LaTeX mapping
    if unicode_latex_mapping:
        unicode_mapping = unicode_latex_mapping
    
    if columns:
        df = df[columns].copy()
    else:
        df = df.copy()
    
    # Define multi-level columns
    if has_multi_index:
        df = df.drop(list(df.filter(regex='top|unique')), axis=1) # LAAAAAAAAA
        print("####--> in : ",df.columns)
        #df.drop(columns=[(col, subcol) for (col,subcol) in df.columns if not (subcol.endswith('unique') or subcol.str.contains('top'))])
        column_tuples = [('INFO', col, subcol) if subcol in ['nb_modalities','count', 'fill_percentage','variable_type'] else ('DESCRIPTIVE STATISTICS', col, subcol) if not (subcol.endswith('_p_value') or subcol.endswith('_statistic')) else ('AB TESTING', col, subcol) for (col, subcol) in df.columns]
    else:
        column_tuples = [('INFO', col) if col in ['nb_modalities','count', 'fill_percentage','variable_type'] else ('DESCRIPTIVE STATISTICS', col) if not (col.endswith('_p_value') or col.endswith('_statistic')) else ('AB TESTING', col) for col in df.columns]

    # Validate columns
    if columns:
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Produce Table :: The following columns do not exist in the DataFrame: {', '.join(missing_columns)}")
        # Ensure the number of elements in column_tuples matches the number of DataFrame columns
        if len(column_tuples) != len(columns):
            print("\n------------- BEFORE TEST MULTI INDEX COL-------------\n",column_tuples)
            raise ValueError(f"Length mismatch: Expected axis has {df.shape[1]} elements, new values have {len(column_tuples)} elements") 
    print("\n --------------PAF4IN\n") 
    print("\n------------- BEFORE TEST MULTI INDEX COL-------------\n",column_tuples)
    multi_index = pd.MultiIndex.from_tuples(column_tuples)
    print("\n------------- TEST MULTI INDEX COL-------------\n",multi_index)
    
    df.columns = multi_index
    print("\n------------- TEST MULTI INDEX DF -------------\n",df.head())

    # Get unique super columns
    super_columns = df.columns.get_level_values(0).unique()
    print("\n------------- super_columns -------------\n",super_columns)
    print("\n--------------------------\n",df.columns)
    # Sort the columns based on super column and then sub-column
    sorted_columns = []
    for super_col in super_columns:
        sub_cols = df.loc[:, super_col].columns
        print("\nsub_cols: ",sub_cols)
        sorted_sub_cols = sorted(sub_cols)
        print("\nsorted_sub_cols: ",sorted_sub_cols)
        if has_multi_index:
            sorted_columns.extend([(super_col, col ,sub_col) for (col, sub_col) in sorted_sub_cols])
        else:
            sorted_columns.extend([(super_col, sub_col) for sub_col in sorted_sub_cols])

    print("\n------------- sorted_columns -------------\n",super_columns)

    # Reorder the DataFrame columns
    df = df.reindex(columns=sorted_columns)

    #adapt the levels of variables
    ind_lvl = 2 if has_multi_index else 1

    # Subsets for styling
    print("\n --------------STUFF TO TEST ------------ \n")
    fill_percentage_subset = [col for col in df.columns if col[ind_lvl] in ['count','fill_percentage']]
    print("\n -------------- fp\n",fill_percentage_subset)
    p_value_columns = [col for col in df.columns if col[ind_lvl].endswith('p_value')] #[col for col in df.columns.get_level_values(1) if col.endswith('_p_value')]
    print("\n --------------pvc\n",p_value_columns)
    non_p_value_columns = [col for col in df.columns if not col[ind_lvl].endswith('_p_value')]
    print("\n --------------npvc\n",non_p_value_columns)
    
    styler = (df
                  .style.set_properties(**{"font-weight": "bold /* --dwrap */", "font-size": "12pt"})
                  .map_index(lambda v: "rotatebox:{45}--rwrap--latex;", level=ind_lvl, axis=1)  
                  .format_index(escape="latex", axis=1)
                  .format_index(escape="latex", axis=0)
                  .background_gradient(cmap="autumn", subset=fill_percentage_subset, text_color_threshold=0.5)
                  .background_gradient(cmap="summer", subset=p_value_columns, vmin=0, vmax=0.05)
                  .format(precision=float_nb_digits, subset=non_p_value_columns)
                  .format(lambda x: format_ltx_p_value(x), subset=p_value_columns)
                )
    print("\n --------------PAF\n")  
    # Convert the styled DataFrame to a LaTeX table
    latex_table = styler.to_latex(multicol_align="|c|", hrules=True, convert_css=True)
    print("\n --------------PAF2\n")  
    # Write LaTeX to a .tex file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage[symbol]{footmisc}\n")
        f.write("\\usepackage{adjustbox}\n")
        f.write("\\usepackage[active,tightpage]{preview}\n")
        f.write("\\usepackage{varwidth}\n")
        f.write("\\usepackage[table]{xcolor}\n")  # For conditional color

        f.write("\\AtBeginDocument{\\begin{preview}\\begin{varwidth}{\\linewidth}}\n")
        f.write("\\AtEndDocument{\\end{varwidth}\\end{preview}}\n")
        f.write("\\begin{document}\n\n")
        
        # Write \DeclareUnicodeCharacter statements for Greek letters and special symbols
        if unicode_latex_mapping:
            for unicode_char, latex_repr in unicode_mapping.items():
                latex_table = latex_table.replace(unicode_char, latex_repr)
        
        # Apply footnotes if any
        if footnotes:
            for name, footnote_text in footnotes.items():
                latex_table = latex_table.replace(name, f"{name}\\footnote{{{footnote_text}}}")
        
        # Apply title if any
        if title:
            f.write(f"\\title{{{title}}}\\date{{}}\\author{{}}\n")
            f.write("\\maketitle\n\\vspace{-2cm}\n")

        f.write("\\begin{adjustbox}{width=\\textwidth, max height=\\textheight}\n")
        f.write(latex_table)
        
        # End document
        f.write("\\end{adjustbox}\n")
        f.write("\\end{document}\n")
        print("\n --------------PAF3\n")  


def produce_xlsx_table(df, columns=None, filename="output.xlsx", title='Data Report', float_nb_digits=2, has_multi_index=False):
    """
    Convert DataFrame to an excel table and write it to a .xlsx file.

    Parameters:
        df (DataFrame): The DataFrame containing data.
        columns (list, optional): The list of columns to include in the table. Defaults to None.
        filename (str, optional): The name of the .tex file to write. Defaults to "output.xlsx".
        title (str, optional): The title of the table. Defaults to None.
        float_nb_digits (int, optional): Number of digits to display for floating-point numbers. Defaults to 2.
    """
    # Error handling
    if df.empty:
        raise ValueError("DataFrame is empty")
    if not filename.endswith(".xlsx"):
        raise ValueError("Filename must have a .xlsx extension")

    if columns:
        df = df[columns].copy()
    else:
        df = df.copy()
    
    # Define multi-level columns
    if has_multi_index:
        df = df.drop(list(df.filter(regex='top|unique')), axis=1) # LAAAAAAAAA
        print("####--> in : ",df.columns)
        #df.drop(columns=[(col, subcol) for (col,subcol) in df.columns if not (subcol.endswith('unique') or subcol.str.contains('top'))])
        column_tuples = [('INFO', col, subcol) if subcol in ['nb_modalities','count', 'fill_percentage','variable_type'] else ('DESCRIPTIVE STATISTICS', col, subcol) if not (subcol.endswith('_p_value') or subcol.endswith('_statistic')) else ('AB TESTING', col, subcol) for (col, subcol) in df.columns]
    else:
        column_tuples = [('INFO', col) if col in ['nb_modalities','count', 'fill_percentage','variable_type'] else ('DESCRIPTIVE STATISTICS', col) if not (col.endswith('_p_value') or col.endswith('_statistic')) else ('AB TESTING', col) for col in df.columns]

    # Validate columns
    if columns:
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Produce Table :: The following columns do not exist in the DataFrame: {', '.join(missing_columns)}")
        # Ensure the number of elements in column_tuples matches the number of DataFrame columns
        if len(column_tuples) != len(columns):
            print("\n------------- BEFORE TEST MULTI INDEX COL-------------\n",column_tuples)
            raise ValueError(f"Length mismatch: Expected axis has {df.shape[1]} elements, new values have {len(column_tuples)} elements") 
    print("\n --------------PAF4IN\n") 
    print("\n------------- BEFORE TEST MULTI INDEX COL-------------\n",column_tuples)
    multi_index = pd.MultiIndex.from_tuples(column_tuples)
    print("\n------------- TEST MULTI INDEX COL-------------\n",multi_index)
    
    df.columns = multi_index
    print("\n------------- TEST MULTI INDEX DF -------------\n",df.head())

    # Get unique super columns
    super_columns = df.columns.get_level_values(0).unique()
    print("\n------------- super_columns -------------\n",super_columns)
    print("\n--------------------------\n",df.columns)
    # Sort the columns based on super column and then sub-column
    sorted_columns = []
    for super_col in super_columns:
        sub_cols = df.loc[:, super_col].columns
        print("\nsub_cols: ",sub_cols)
        sorted_sub_cols = sorted(sub_cols)
        print("\nsorted_sub_cols: ",sorted_sub_cols)
        if has_multi_index:
            sorted_columns.extend([(super_col, col ,sub_col) for (col, sub_col) in sorted_sub_cols])
        else:
            sorted_columns.extend([(super_col, sub_col) for sub_col in sorted_sub_cols])

    print("\n------------- sorted_columns -------------\n",super_columns)

    # Reorder the DataFrame columns
    df = df.reindex(columns=sorted_columns)

    #adapt the levels of variables
    ind_lvl = 2 if has_multi_index else 1

    # Subsets for styling
    print("\n --------------STUFF TO TEST ------------ \n")
    fill_percentage_subset = [col for col in df.columns if col[ind_lvl] in ['count','fill_percentage']]
    print("\n -------------- fp\n",fill_percentage_subset)
    p_value_columns = [col for col in df.columns if col[ind_lvl].endswith('p_value')] #[col for col in df.columns.get_level_values(1) if col.endswith('_p_value')]
    print("\n --------------pvc\n",p_value_columns)
    non_p_value_columns = [col for col in df.columns if not col[ind_lvl].endswith('_p_value')]
    print("\n --------------npvc\n",non_p_value_columns)
    
    styler = ( df
                .style
                .set_properties(**{"font-weight": "bold", "font-size": "12pt"})
                .background_gradient(cmap="autumn", subset=fill_percentage_subset, text_color_threshold=0.5)
                .background_gradient(cmap="summer", subset=p_value_columns, vmin=0, vmax=0.05)
                .format(precision=float_nb_digits, subset=non_p_value_columns)
                #.format(lambda x: format_xlsx_p_value(x), subset=p_value_columns)
            )
    print("OKKKK1")
    if os.path.isfile(filename):
        print("OKKKK2")
        # If the file exists, append the new sheet to the existing file
        with pd.ExcelWriter(filename, mode='a', if_sheet_exists="replace") as writer:
            print("OKKKK3")
            styler.to_excel(writer, sheet_name=title, encoding='utf-8', na_rep='NA')
    else:
        with pd.ExcelWriter(filename) as writer:  
            styler.to_excel(writer, sheet_name=title, encoding='utf-8', na_rep='NA')


def produce_html_table(df, columns=None, filename="output.html", title='Data Report', float_nb_digits=2, has_multi_index=False):
    """
    Convert DataFrame to an html table and write it to a .html file.

    Parameters:
        df (DataFrame): The DataFrame containing data.
        columns (list, optional): The list of columns to include in the table. Defaults to None.
        filename (str, optional): The name of the .tex file to write. Defaults to "output.xlsx".
        title (str, optional): The title of the table. Defaults to None.
        float_nb_digits (int, optional): Number of digits to display for floating-point numbers. Defaults to 2.
    """
    # Error handling
    if df.empty:
        raise ValueError("DataFrame is empty")
    if not filename.endswith(".html"):
        raise ValueError("Filename must have a .xlsx extension")
    
    if columns:
        df = df[columns].copy()
    else:
        df = df.copy()
    
    # Define multi-level columns
    if has_multi_index:
        df = df.drop(list(df.filter(regex='top|unique')), axis=1) # LAAAAAAAAA
        print("####--> in : ",df.columns)
        #df.drop(columns=[(col, subcol) for (col,subcol) in df.columns if not (subcol.endswith('unique') or subcol.str.contains('top'))])
        column_tuples = [('INFO', col, subcol) if subcol in ['nb_modalities', 'count', 'fill_percentage','variable_type'] else ('DESCRIPTIVE STATISTICS', col, subcol) if not (subcol.endswith('_p_value') or subcol.endswith('_statistic')) else ('AB TESTING', col, subcol) for (col, subcol) in df.columns]
    else:
        column_tuples = [('INFO', col) if col in ['nb_modalities', 'count', 'fill_percentage','variable_type'] else ('DESCRIPTIVE STATISTICS', col) if not (col.endswith('_p_value') or col.endswith('_statistic')) else ('AB TESTING', col) for col in df.columns]

    # Validate columns
    if columns:
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Produce Table :: The following columns do not exist in the DataFrame: {', '.join(missing_columns)}")
        # Ensure the number of elements in column_tuples matches the number of DataFrame columns
        if len(column_tuples) != len(columns):
            print("\n------------- BEFORE TEST MULTI INDEX COL-------------\n",column_tuples)
            raise ValueError(f"Length mismatch: Expected axis has {df.shape[1]} elements, new values have {len(column_tuples)} elements") 
    print("\n --------------PAF4IN\n") 
    print("\n------------- BEFORE TEST MULTI INDEX COL-------------\n",column_tuples)
    multi_index = pd.MultiIndex.from_tuples(column_tuples)
    print("\n------------- TEST MULTI INDEX COL-------------\n",multi_index)
    
    df.columns = multi_index
    print("\n------------- TEST MULTI INDEX DF -------------\n",df.head())

    # Get unique super columns
    super_columns = df.columns.get_level_values(0).unique()
    print("\n------------- super_columns -------------\n",super_columns)
    print("\n--------------------------\n",df.columns)
    # Sort the columns based on super column and then sub-column
    sorted_columns = []
    for super_col in super_columns:
        sub_cols = df.loc[:, super_col].columns
        print("\nsub_cols: ",sub_cols)
        sorted_sub_cols = sorted(sub_cols)
        print("\nsorted_sub_cols: ",sorted_sub_cols)
        if has_multi_index:
            sorted_columns.extend([(super_col, col ,sub_col) for (col, sub_col) in sorted_sub_cols])
        else:
            sorted_columns.extend([(super_col, sub_col) for sub_col in sorted_sub_cols])

    print("\n------------- sorted_columns -------------\n",super_columns)

    # Reorder the DataFrame columns
    df = df.reindex(columns=sorted_columns)

    #adapt the levels of variables
    ind_lvl = 2 if has_multi_index else 1

    # Subsets for styling
    print("\n --------------STUFF TO TEST ------------ \n")
    fill_percentage_subset = [col for col in df.columns if col[ind_lvl] in ['count','fill_percentage']]
    print("\n -------------- fp\n",fill_percentage_subset)
    p_value_columns = [col for col in df.columns if col[ind_lvl].endswith('p_value')] #[col for col in df.columns.get_level_values(1) if col.endswith('_p_value')]
    print("\n --------------pvc\n",p_value_columns)
    non_p_value_columns = [col for col in df.columns if not col[ind_lvl].endswith('_p_value')]
    print("\n --------------npvc\n",non_p_value_columns)
    
    styler = ( df
                .style
                .set_properties(**{"font-weight": "bold", "font-size": "12pt"})
                .background_gradient(cmap="autumn", subset=fill_percentage_subset, text_color_threshold=0.5)
                .background_gradient(cmap="summer", subset=p_value_columns, vmin=0, vmax=0.05)
                .format(precision=float_nb_digits, subset=non_p_value_columns)
                .format(lambda x: format_html_p_value(x), subset=p_value_columns)
            )

    # Add borders to the table
    styler = styler.set_table_styles([
        {'selector': 'th', 'props': [('border', '1px solid black')]},
        {'selector': 'td', 'props': [('border', '1px solid lightgray')]},
        {'selector': 'tr:hover', 'props': [('background-color', '#f5f5f5')]}
    ])
    # Convert the styled DataFrame to an HTML table
    html_table = styler.to_html(caption=title)

    # Write the HTML table to a file
    with open(filename, encoding='utf-8', mode='w') as f:
        f.write(html_table)

######################################## PRODUCE PDF & PNG TABLE FROM LATEX SCRIPT ########################################

def compile_latex_to_pdf(tex_file, output_folder):
    """
    Compile LaTeX to PDF using pdflatex and return the path of the created PDF file.

    Parameters:
        tex_file (str): The path to the.tex file.
        output_folder (str): The path to the folder where the PDF file will be saved.
    Returns:
        str: The path of the created PDF file.
    """
    try:
        # Change the working directory to the output folder
        original_cwd = os.getcwd()
        os.chdir(output_folder)

        # Compile the LaTeX file to PDF
        subprocess.run(['pdflatex', tex_file], check=True)
        print("PDF generated successfully:", tex_file)

        # Get the path of the created PDF file
        pdf_file = os.path.join(output_folder, os.path.basename(tex_file).replace('.tex', '.pdf'))

        # Change the working directory back to the original directory
        os.chdir(original_cwd)

        return pdf_file
    except subprocess.CalledProcessError as e:
        print(f"Error generating PDF: {e}")
        return None

    
def pdf_to_png(pdf_file, output_dir=None):
    """
    Convert PDF to PNG using pdftoppm with high resolution and return the path of the created PNG file.

    Parameters:
        pdf_file (str): The path to the PDF file.
        output_dir (str, optional): The directory to save the output PNG file.
    Returns:
        str: The path of the created PNG file.
    """
    if output_dir is None:
        output_dir = os.path.dirname(pdf_file)  # Use the same directory as the PDF file
    
    pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]  # Extract base name of the PDF file
    png_file = os.path.join(output_dir, pdf_name)  # Construct output PNG file name
    
    # Convert PDF to PNG using pdftoppm with high resolution
    try:
        subprocess.run(['pdftoppm', '-png', '-r', '300', pdf_file, png_file], check=True)
        print(f"High-quality PNG generated successfully: {png_file}")
        return png_file
    except FileNotFoundError:
        print("pdftoppm not found. Please install Poppler-utils and add it to your system's PATH.")
        return None
    
######################################## HANDLE MISSING VALUES ########################################

def handle_missing_values(df, variables=None, target_variable=None, strategy=None):
    """
    Handle missing values in the DataFrame based on the specified strategy.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - variables (list): A list of variable names for which to handle missing values.
    - target_variable (str): The name of the target variable.
    - strategy (str): Strategy to handle missing values. Options: 'remove', 'median', 'mean', 'missforest', 'knn', 'mode'.

    Returns:
    - pandas.DataFrame: DataFrame with missing values handled based on the specified strategy.
    """
    print("\n handle_missing_values : start \n ")
    if target_variable and variables:
        all_variables = variables + [target_variable]
        df_filtered = df[all_variables].copy()
    elif variables:
        all_variables = variables
        df_filtered = df[all_variables].copy()
    elif target_variable:
        all_variables=target_variable
        df_filtered = df[all_variables].copy()
    else:
        df_filtered = df.copy()

    df_filtered.info()

    print(f"\n handle_missing_values : group, all_variables {df_filtered.shape,len(all_variables)} \n ")
    if strategy == 'remove':
        # Remove rows with missing values for the specified variables
        df_imputed = df_filtered.dropna(subset=all_variables)
    elif strategy == 'median':
        # Impute missing values with median for the specified variables
        df_imputed = df_filtered.fillna(df_filtered.median())
    elif strategy == 'mean':
        # Impute missing values with mean for the specified variables
        df_imputed = df_filtered.fillna(df_filtered.mean())
    elif strategy == 'missforest': # in progress
        # Use MissForest imputer for missing value imputation
        clf = RandomForestClassifier(n_jobs=-1)
        rgr = RandomForestRegressor(n_jobs=-1)
        #imputer = MissForest(clf, rgr)
        #imputation = imputer.fit_transform(df_filtered)
        #df_imputed = pd.DataFrame(imputation, columns=all_variables,index=df_filtered.index)
    elif strategy == 'knn':
        # Use KNN imputer for missing value imputation
        imputer = KNNImputer(n_neighbors=10, weights='distance',keep_empty_features=True,missing_values=np.nan)
        print(f"\n handle_missing_values, before knn : group, all_variables {df_filtered.shape,len(all_variables)} \n ")
        imputation = imputer.fit_transform(df_filtered)
        print("\n------> ",imputer.feature_names_in_, len(imputer.feature_names_in_), imputer.n_features_in_)
        print(f"\n handle_missing_values, after knn : group, all_variables {imputation.shape,len(all_variables)} \n ")
        # Create DataFrame with imputed values
        
        df_imputed = pd.DataFrame(imputation, columns=all_variables, index=df_filtered.index)

        # Check shapes and indices
        print(f"Number of eligible variables: {len(all_variables)}")
        print(f"Original DataFrame shape: {df.shape}, Original DataFrame index length: {len(df.index)}")
        print(f"Filtered DataFrame shape: {df_filtered.shape}, Filtered DataFrame index length: {len(df_filtered.index)}")
        print(f"Imputed DataFrame shape: {df_imputed.shape}, Filtered DataFrame index length: {len(df_imputed.index)}")


    elif strategy == 'mode':
        # Use Mode imputer for missing value imputation
        imputer = SimpleImputer(strategy='most_frequent')
        imputation = imputer.fit_transform(df_filtered)
        df_imputed = pd.DataFrame(imputation, columns=all_variables,index=df_filtered.index)
    else:
        logging.warning(f"Invalid missing_strategy: '{strategy}'. Using 'remove' strategy instead.")
        df_imputed = df.dropna(subset=all_variables)
    print(f"\n handle_missing_values out: group, all_variables {df_filtered.shape,len(all_variables)} \n ")
    return df_imputed


######################################## DEFINE STATISTICAL TESTS ########################################

################ categorical variables 

def chi2_test(df, variables, target_variable, missing_strategy=None, test_name=''):
    """
    Perform a chi-square test of independence between categorical variables and a target variable.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - variables (list): A list of column names representing categorical variables.
    - target_variable (str): The name of the target variable.
    - missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'.

    Returns:
    - dict: A dictionary containing the chi-square test results for each categorical variable.
    """
    chi2_results = {}

    # Handle missing values for all variables once before the loop
    try:
        if missing_strategy:
            df = handle_missing_values(df, variables, target_variable, missing_strategy).copy()
    except Exception as e:
        logging.error(f"Error occurred during data preprocessing: {e!s}")
        return chi2_results

    for cat_var in variables:
        try:
            contingency_table = pd.crosstab(df[cat_var], df[target_variable])
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            chi2_results[cat_var] = {f'{test_name}_statistic': chi2, f'{test_name}_p_value': p_value}
        except Exception as e:
            logging.error(f"Error occurred while computing chi-square test for variable '{cat_var}': {e!s}")

    return chi2_results


def fisher_exact_test(df, variables, target_variable, missing_strategy='remove', test_name=''):
    """
    Perform Fisher's exact test for binary variables in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - variables (list): A list of names of binary variables for which to perform the test.
    - target_variable (str): The name of the target variable for the contingency table.
    - missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'.

    Returns:
    - dict: Dictionary containing the results of the Fisher's exact test for each variable.
    """
    fisher_results = {}

    # Handle missing values for all variables once before the loop
    df_filtered = handle_missing_values(df, variables, target_variable, missing_strategy)

    for variable in variables:
        try:
            # Check if the variable has only two unique values (binary variable)
            unique_values = df_filtered[variable].unique()
            if len(unique_values) != 2:
                logging.error(f"Fisher's exact test on {target_variable}: Variable '{variable}' is not binary. Fisher's exact test requires a binary variable with only two unique values.")
                continue

            # Perform Fisher's exact test
            contingency_table = pd.crosstab(df_filtered[variable], df_filtered[target_variable])
            if contingency_table.shape != (2, 2):
                logging.error(f"Contingency table for variable '{variable}' does not have shape (2, 2). Fisher's exact test requires a contingency table of shape (2, 2).")
                continue
            
            fisher_statistic, fisher_p_value = stats.fisher_exact(contingency_table)

            # Store the results in the dictionary
            fisher_results[variable] = {f'{test_name}_statistic': fisher_statistic, f'{test_name}_p_value': fisher_p_value}
        except Exception as e:
            logging.error(f"Error occurred while computing Fisher's exact test for variable {variable} on {target_variable}: {e!s}")

    return fisher_results


########## numerical variables with descriptive

def wilcoxon_rank_sum_test_with_descriptive(df, variables, target_variable=None, control_group=None, target_group=None, missing_strategy='remove', descriptive=False, group_names=None, test_name=''):
    """
    Perform Wilcoxon rank sum test for continuous variables in a DataFrame.
    
    'scipy.stats.mannwhitneyu' and 'scipy.stats.ranksums' will be applied depending on the context.
    Mann-Whitney U test ('scipy.stats.mannwhitneyu'): The test statistic is based on the ranks of the observations from both samples.
    It calculates the Mann-Whitney U statistic, which represents the probability of one randomly selected observation from one sample being greater than a randomly selected observation from the other sample.
    Wilcoxon rank-sum test: It also calculates a U statistic, but it's slightly different from the Mann-Whitney U statistic. The Wilcoxon rank-sum test is a special case of the Mann-Whitney U test when the sample sizes are equal.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - variables (list): A list of names of continuous variables for which to perform the test.
    - target_variable (str): The name of the target variable that is used to infer target (True) and control (False) groups. It should be a binary variable.
    - control_group (pandas.DataFrame, optional) and target_group (pandas.DataFrame, optional): Instead of using the target variable, it is possible to provide explicitly the control and target groups.
    - missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'.
    - descriptive (bool): Whether to include descriptive statistics for control and target groups.

    Returns:
    - dict: Dictionary containing the results of the Wilcoxon rank sum test for each variable.
    """
    wilcoxon_results = {}
    # Handle missing values and filter the dataset
    if target_variable is not None:
        df_filtered = handle_missing_values(df=df, variables=variables, target_variable=target_variable, strategy=missing_strategy)
    if control_group is not None:
        control_group_full = handle_missing_values(df=control_group, variables=variables, strategy=missing_strategy)
    if target_group is not None:
        target_group_full = handle_missing_values(df=target_group, variables=variables, strategy=missing_strategy)


    for variable in variables:
        try:
            if target_variable:
                # Separate data into control and target groups based on the target variable
                control_group = df_filtered.loc[df_filtered[target_variable] == 0, variable]
                target_group = df_filtered.loc[df_filtered[target_variable] == 1, variable]
            elif control_group_full is not None and target_group_full is not None:
                control_group = control_group_full[variable]
                target_group = target_group_full[variable]
                # Check if control and target groups are of the same length
                if len(control_group) != len(target_group):
                    logging.warning("Control and target groups does not have the same length. A more general version of Mann-Whitney U test will be used")
            else:
                logging.error("Either target variable or explicit control and target groups must be provided.")
            
            if len(control_group) == len(target_group):
                # Perform Wilcoxon rank sum test
                wilcoxon_statistic, wilcoxon_p_value = stats.ranksums(control_group, target_group)
            else:
                wilcoxon_statistic, wilcoxon_p_value = stats.mannwhitneyu(control_group, target_group)

            if descriptive:
                # Calculate descriptive statistics for control and target groups
                if group_names is None:
                    target_stats = calculate_group_stats(target_group, group_name=f'{test_name}_target_group')
                    control_stats = calculate_group_stats(control_group, group_name=f'{test_name}_control_group')
                else:
                    target_stats = calculate_group_stats(target_group, group_name=group_names[0])
                    control_stats = calculate_group_stats(control_group, group_name=group_names[1])

                wilcoxon_results[variable] = {
                    f'{test_name}_statistic': wilcoxon_statistic,
                    f'{test_name}_p_value': wilcoxon_p_value,
                    **control_stats,
                    **target_stats
                }
            else:
                wilcoxon_results[variable] = {
                    f'{test_name}_statistic': wilcoxon_statistic,
                    f'{test_name}_p_value': wilcoxon_p_value
                }
        except Exception as e:
            logging.error(f"Error occurred while computing Wilcoxon rank sum test for variable '{variable}': {e!s}")

    return wilcoxon_results


def t_test_with_descriptive(df, variables, target_variable=None, control_group=None, target_group=None, missing_strategy='remove', descriptive=False, group_names=None, test_name=''):
    """
    Compute a t-test between two groups based on given variables.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - variables (list): A list of names of continuous variables for which to perform the test.
    - target_variable (str): The name of the target variable that is used to infer target (True) and control (False) groups. It should be a binary variable. If provided, the function will split the data into two groups based on this variable. Either this parameter or both control_group and target_group must be provided.
    - control_group (pandas.DataFrame, optional): A DataFrame representing the control group. If provided, this group will be used as the control group in the t-test. This parameter must be used in conjunction with target_group.
    - target_group (pandas.DataFrame, optional): A DataFrame representing the target group. If provided, this group will be used as the target group in the t-test. This parameter must be used in conjunction with control_group.
    - missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'. Default is 'remove'.
    - descriptive (bool): Whether to include descriptive statistics for control and target groups. If True, the function will calculate and include descriptive statistics such as mean, standard deviation, quartiles, and min/max for each group. Default is False.

    Returns:
    - dict: Dictionary containing the results of the t-test for each variable. The dictionary keys are the variable names, and the values are dictionaries containing the t-statistic, p-value, and optionally, descriptive statistics for the control and target groups.
    """
    t_test_results = {}
    # Handle missing values and filter the dataset
    if target_variable is not None:
        df_filtered = handle_missing_values(df=df, variables=variables, target_variable=target_variable, strategy=missing_strategy)
    if control_group is not None:
        control_group_full = handle_missing_values(df=control_group, variables=variables, strategy=missing_strategy)
    if target_group is not None:
        target_group_full = handle_missing_values(df=target_group, variables=variables, strategy=missing_strategy)

    
    for variable in variables:
        try:
            if target_variable:
                # Separate data into control and target groups based on the target variable
                control_group = df_filtered.loc[df_filtered[target_variable] == 0, variable]
                target_group = df_filtered.loc[df_filtered[target_variable] == 1, variable]
            elif control_group is not None and target_group is not None:
                control_group = control_group_full[variable]
                target_group = target_group_full[variable]
                # log specific case
                logging.warning(f"Custom control and target groups will be used for t-test on variable {variable}")
            else:
                logging.error("Either target variable or explicit control and target groups must be provided.")

            # Perform t-test
            t_statistic, t_p_value = stats.ttest_ind(control_group, target_group, nan_policy='omit')

            if descriptive:
                # Calculate descriptive statistics for control and target groups
                if group_names is None:
                    target_stats = calculate_group_stats(target_group, group_name=f'{test_name}_target_group')
                    control_stats = calculate_group_stats(control_group, group_name=f'{test_name}_control_group')
                else:
                    target_stats = calculate_group_stats(target_group, group_name=group_names[0])
                    control_stats = calculate_group_stats(control_group, group_name=group_names[1])
                
                # Store the results in a dictionary
                t_test_results[variable] = {
                    f'{test_name}_statistic': t_statistic,
                    f'{test_name}_p_value': t_p_value,
                    **control_stats,
                    **target_stats
                }
            else:
                # Store the results in a dictionary
                t_test_results[variable]  = {
                    f'{test_name}_statistic': t_statistic,
                    f'{test_name}_p_value': t_p_value
                }
            
        except Exception as e:
            logging.error(f"Error occurred while computing t-test for variable '{variable}': {e!s}")

    return t_test_results


def kruskal_wallis_test_with_descriptive(groups=None, variables=None, missing_strategy='remove', descriptive=False, group_names=None, test_name=''):
    """
    Perform Kruskal-Wallis test for continuous variables in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - variables (list): A list of names of continuous variables for which to perform the test.
    - groups (list of pandas.DataFrame): List of DataFrames representing different groups.
    - missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'.
    - descriptive (bool): Whether to include descriptive statistics for groups.

    Returns:
    - dict: Dictionary containing the results of the Kruskal-Wallis test for each variable.
    """
    kruskal_results = {}
    
    # Check if groups are provided and contain valid dataframes
    if not groups or not all(isinstance(group, pd.DataFrame) for group in groups):
        logging.error("Groups must be a non-empty list of pandas DataFrames.")
        return kruskal_results
    
    # Check if each group dataframe contains all variables
    missing_variables = [variable for variable in variables if not all(variable in group.columns for group in groups)]
    if missing_variables:
        missing_variables_str = ", ".join(missing_variables)
        logging.error(f"All groups must contain all variables. Missing variables: {missing_variables_str}.")
        return kruskal_results
    
    # Restrict each group to variables and handle missing values in each group DataFrame
    processed_groups = []
    for group in groups:
        print(f"\n GROUP {group.shape, len(variables)}!!!\n ")
        processed_group = group[variables].copy()  # Create a copy to avoid modifying the original group
        print(f"\n PROCESSED GROUP {processed_group.shape}!!!\n ")
        processed_group = handle_missing_values(df=processed_group, variables=variables, strategy=missing_strategy)
        print("\n ADD...\n ")
        processed_groups.append(processed_group)
        print("\n ADDED!!!\n ")
    
    # Iterate over each variable
    for variable in variables:
        try:
            # Extract data for each group
            group_data = [processed_group[variable] for processed_group in processed_groups]
            
            # Perform Kruskal-Wallis test
            print("\n ON Y EST presqueeeeee!!!\n ")
            k_statistic, p_value = stats.kruskal(*group_data)

            # Store the results in the dictionary
            if descriptive:
                # Calculate descriptive statistics for groups
                group_stats = calculate_group_stats_for_multiple_groups(group_data,group_names)
                print(group_stats)

                kruskal_results[variable] = {
                    f'{test_name}_statistic': k_statistic,
                    f'{test_name}_p_value': p_value,
                    **group_stats
                }
            else:
                kruskal_results[variable] = {
                    f'{test_name}_statistic': k_statistic,
                    f'{test_name}_p_value': p_value
                }
        except Exception as e:
            logging.error(f"Error occurred while computing Kruskal-Wallis test for variable '{variable}': {e!s}")

    return kruskal_results


def anova_test_with_descriptive(groups=None, variables=None,  missing_strategy='remove', descriptive=False, group_names=None, test_name=''):
    """
    Perform ANOVA test for continuous variables in a list of DataFrames representing different groups.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - variables (list): A list of names of continuous variables for which to perform the test.
    - groups (list of pandas.DataFrame): List of DataFrames representing different groups.
    - missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'.
    - descriptive (bool): Whether to include descriptive statistics for groups.

    Returns:
    - dict: Dictionary containing the results of the ANOVA test for each variable.
    """
    anova_results = {}
    
    # Check if groups are provided and contain valid dataframes
    if not groups or not all(isinstance(group, pd.DataFrame) for group in groups):
        logging.error("Groups must be a non-empty list of pandas DataFrames.")
        return anova_results
    
    # Check if each group dataframe contains all variables
    missing_variables = [variable for variable in variables if not all(variable in group.columns for group in groups)]
    if missing_variables:
        missing_variables_str = ", ".join(missing_variables)
        logging.error(f"All groups must contain all variables. Missing variables: {missing_variables_str}.")
        return anova_results
    
    # Restrict each group to variables and handle missing values in each group DataFrame
    processed_groups = []
    for group in groups:
        processed_group = group[variables].copy()  # Create a copy to avoid modifying the original group
        processed_group = handle_missing_values(df=processed_group, variables=variables, strategy=missing_strategy)
        processed_groups.append(processed_group)
    
    # Iterate over each variable
    for variable in variables:
        try:
            # Extract data for each group
            group_data = [processed_group[variable] for processed_group in processed_groups]
            
            # Perform ANOVA test
            f_statistic, p_value = stats.f_oneway(*group_data)

            # Store the results in the dictionary
            if descriptive:
                # Calculate descriptive statistics for groups
                group_stats = calculate_group_stats_for_multiple_groups(group_data,group_names)

                anova_results[variable] = {
                    f'{test_name}_statistic': f_statistic,
                    f'{test_name}_p_value': p_value,
                    **group_stats
                }
            else:
                anova_results[variable] = {
                    f'{test_name}_statistic': f_statistic,
                    f'{test_name}_p_value': p_value
                }
        except Exception as e:
            logging.error(f"Error occurred while computing ANOVA test for variable '{variable}': {e!s}")

    return anova_results


######################################## PROCESS STATISTICAL TESTS ########################################

def create_multiindex_dataframe(result, group_keys=[], test_name=None, super_column=''):
    if len(group_keys) > 1:
        g_names = group_keys
    elif test_name is not None:
        g_names = [f"{test_name}_control_group", f"{test_name}_target_group"]
    else:
        g_names=None

    if isinstance(result, dict):
        result_df = pd.DataFrame.from_dict(result)
    elif isinstance(result, pd.DataFrame):
        result_df = result
    else:
        raise ValueError(f"result is of type {type(result)}, it must be a dictionary or a DataFrame")
    
    new_index_levels = []
    for level in result_df.index:
        if g_names is not None:
            for prefix in g_names:
                if level.startswith(prefix):
                    new_index_levels.append((prefix, level[len(prefix)+1:]))
                    break
            else:
                new_index_levels.append((super_column, level))
        else:
            new_index_levels.append((super_column, level))

    new_index = pd.MultiIndex.from_tuples(new_index_levels, names=['Groups', 'Variables'])

    result_df.index = new_index
    result_df = result_df.transpose()

    return result_df


def process_group(info_group, df):
    """
    Process a group defined by information in 'info_group' dictionary.

    Parameters:
    - info_group (dict): Dictionary containing information about the group.
    - df (pandas.DataFrame): DataFrame containing the data.

    Returns:
    - pandas.DataFrame: DataFrame representing the processed group.
    """
    # Extract information from info_group dictionary
    group_mask = info_group.get("family_mask")
    print("group_mask:", group_mask)
    group_operator = info_group.get("operator")
    print("group_operator:", group_operator)

    # Check the operator type and filter the DataFrame accordingly
    if group_operator == 'UNION':
        group = df[df[f"{group_mask}_tag"] == True]
    elif group_operator == 'INTER':
        group = df[df[f"{group_mask}_tag_conj"] == True]
    else:
        raise ValueError(f"Operator {group_operator} is not supported")

    return group.copy()


def make_test(test_type, df, variables, target_variable=None, control_group=None, target_group=None, groups=None, missing_strategy='remove', descriptive=False, group_names=None,test_name=''):
    """
    Select the test to compute between two groups based on given variables.

    Parameters:
    - test_type(str): the name of the test to process
    - df (pandas.DataFrame): The DataFrame containing the data.
    - variables (list): A list of names of continuous variables for which to perform the test.
    - target_variable (str, optional): The name of the target variable that is used to infer target (True) and control (False) groups. It should be a binary variable. If provided, the function will split the data into two groups based on this variable. Either this parameter or both control_group and target_group must be provided.
    - control_group (pandas.DataFrame, optional): A DataFrame representing the control group. If provided, this group will be used as the control group in the t-test. This parameter must be used in conjunction with target_group.
    - target_group (pandas.DataFrame, optional): A DataFrame representing the target group. If provided, this group will be used as the target group in the t-test. This parameter must be used in conjunction with control_group.
    - missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'. Default is 'remove'.
    - descriptive (bool): Whether to include descriptive statistics for control and target groups. If True, the function will calculate and include descriptive statistics such as mean, standard deviation, quartiles, and min/max for each group. Default is False.

    Returns:
    - dict: Dictionary containing the results of the corresponding test for each variable.
    """

    # SPLIT NUMERICAL AND CATEGORICAL VARIABLES
    cat_vars = df[variables].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    num_vars = df[variables].select_dtypes(include=['number']).columns.tolist()

    # CONVERT IN PROPER INT AND FLOAT TYPES
    #int_vars = df[num_vars].select_dtypes(include=['int']).columns
    #float_vars = df[num_vars].select_dtypes(include=['float']).columns
    #df[int_vars] = df[int_vars].astype(int)
    #df[float_vars] = df[float_vars].astype(float)

    if test_type == 'chi2':
        selected_vars = cat_vars
        result = chi2_test(df=df, variables=selected_vars, target_variable=target_variable, test_name=test_name)  
    elif test_type == 'fisher_exact':
        selected_vars = cat_vars
        result = fisher_exact_test(df=df, variables=selected_vars, target_variable=target_variable, test_name=test_name)
    elif test_type == 'wilcoxon':
        selected_vars = num_vars
        result = wilcoxon_rank_sum_test_with_descriptive(df, selected_vars, target_variable, control_group, target_group, missing_strategy, descriptive, group_names, test_name)
    elif test_type == 't-test':
        selected_vars = num_vars
        print("#####ffff####\n ----> ",test_name)
        result = t_test_with_descriptive(df, selected_vars, target_variable, control_group, target_group, missing_strategy, descriptive, group_names, test_name)
    elif test_type == 'anova':
        selected_vars = num_vars
        if groups is None:
            groups=[target_group,control_group]
        result = anova_test_with_descriptive(groups, selected_vars, missing_strategy, descriptive, group_names, test_name)
    elif test_type == 'kruskal-wallis':
        selected_vars = num_vars
        if groups is None:
            groups=[target_group,control_group]
        result = kruskal_wallis_test_with_descriptive(groups, selected_vars, missing_strategy, descriptive, group_names, test_name)

    if target_variable:
        print(f"\n###\n Processing {test_type}: \n On eligible variables [{selected_vars}] \n Based on target variable'{target_variable}': \n Results: {result} ###\n")
    else:
        print(f"\n###\n Processing {test_type}: \n On eligible variables [{selected_vars}]")

    return result


def process_test(df, test_name, test_info, config=None,multi_index=False): # config,
    """
    Process the statistical test based on the config file data.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the data.
    - test_name (str): Name of the hypothesis test being processed.
    - test_info (dict): Information about the hypothesis test, including test type, variables, groups, etc.
    - config (dict, optional): Additional configuration parameters. Default is None.

    Returns:
    - tuple: A dictionary containing the results (statistics and p-value for each required and eligible variable) for the processed statistical test.

    Raises:
    - None
    """
    try:
        print("#######################\n")
        groups = []  # Store DataFrames for each group
        group_keys = []
        test_name = test_name
        logging.info(f"Processing hypothesis testing: {test_name}")
        
        # Test (anova, chi2, wilcoxon, kruskal-wallis, etc.)
        if test_info.get("TEST_TYPE"):
            test_type = test_info.get("TEST_TYPE")

        # VARIABLES TO ANALYZE
        if test_info.get("VARIABLES"):
            variables = test_info.get("VARIABLES")
            print("variables:", variables)
        else:
            variables = df.columns.tolist()
        
        # COMPUTE GROUP DESCRIPTIVE STATISTICS
        if test_info.get("DESCRIBE_GROUPS"):
            describe_groups = test_info.get("DESCRIBE_GROUPS")
        else:
            describe_groups = False

        # MISSING STRATEGY FOR IMPUTATION
        missing_strategy = test_info.get("MISSING_STRATEGY")
           
        # Iterate over keys in test_info
        for key, value in test_info.items():
            if key.startswith("GROUP"):
                group = process_group(info_group=value, df=df)
                key = test_name + '_' + key
                group_keys.append(key)
                groups.append(group)
                print(group.head())
        
        if len(group_keys) > 1:
            group_names_str = ", ".join([key for key in group_keys])

        print("\nVERIFICATIONS\n")
        if test_info.get("TARGET_VARIABLE"):
            print("\nTARGET_VARIABLE\n")
            target_var = test_info.get("TARGET_VARIABLE")
            result = make_test(test_type=test_type,df=df,variables=variables,target_variable=target_var,missing_strategy=missing_strategy,descriptive=describe_groups,test_name=test_name)
        
        elif len(groups) == 2:  # Perform test if there are exactly two groups
            print("\ngroup_1 and group_2\n")
            result = make_test(test_type=test_type, df=df, variables=variables, target_group=groups[0], control_group=groups[1], missing_strategy=missing_strategy,descriptive=describe_groups,group_names=group_keys,test_name=test_name)
            print(f" Based on groups: {group_names_str}\n Results: {result} ###\n")

        elif len(groups) > 2:  # Perform test if there are more than two groups
            print("\nMultiple groups involved in the test\n")  
            result = make_test(test_type=test_type, df=df, variables=variables, groups=groups, missing_strategy=missing_strategy,descriptive=describe_groups,group_names=group_keys,test_name=test_name)
            print(f" Based on groups: {group_names_str}\n Results: {result} ###\n")
        else:
            return None
        
        if multi_index:
            result_df = create_multiindex_dataframe(result, group_keys, test_name)
        else:
            result_df = pd.DataFrame.from_dict(result).transpose()
        return result_df
    except Exception as e:
        logging.error(f"An unexpected error occurred for test: {test_name}. Error: {e!s}")
        return None
    
