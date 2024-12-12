# import pandas as pd
# from scipy.stats import chisquare
# from scipy.stats import fisher_exact

# ## To use a chi-squared test on your dataset, you can follow these general steps:

# # 1.Identify Categorical Variables: Determine which variables in your dataset are categorical. These are typically variables that represent categories or groups rather than numerical values.
# # 2.Prepare Data: Extract the columns containing categorical variables from your dataset and prepare them for analysis. This may involve encoding categorical variables if they are not already encoded.
# # 3.Compute Observed Frequencies: Compute the observed frequencies for each category in your categorical variables. You can use the value_counts() function in pandas to compute these frequencies.
# # 4.Perform Chi-Squared Test: Use the scipy.stats.chisquare() function to perform the chi-squared test. This function takes the observed frequencies as input and returns the chi-squared statistic and p-value.

# # Example dataset with categorical variables
# data = {
#     'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
#     'Smoker': ['Yes', 'No', 'Yes', 'No', 'Yes'],
#     'Outcome': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']
# }

# # Convert data to pandas DataFrame
# df = pd.DataFrame(data)

# # Compute observed frequencies for each categorical variable
# observed_frequencies = {}
# for column in df.columns:
#     observed_frequencies[column] = df[column].value_counts()

# # Perform chi-squared test for each categorical variable
# chi2_results = {}
# for column, frequencies in observed_frequencies.items():
#     chi2_statistic, p_value = chisquare(frequencies)
#     chi2_results[column] = {'statistic': chi2_statistic, 'p_value': p_value}

# # Print chi-squared test results
# for column, result in chi2_results.items():
#     print(f"Chi-squared test results for {column}:")
#     print(f"Statistic: {result['statistic']}")
#     print(f"P-value: {result['p_value']}")

# ##To use Fisher's exact test on your DataFrame, you can follow these steps:

# # 1.Identify Contingency Tables: Determine which pairs of variables you want to compare using Fisher's exact test. Fisher's exact test is typically used to compare two categorical variables.
# # 2.Prepare Data: Extract the columns containing the variables you want to compare from your DataFrame and prepare them for analysis. Ensure that the data is in the form of a contingency table, with rows representing one categorical variable and columns representing the other.
# # 3.Compute Contingency Table: Create a contingency table from your data. The contingency table should contain the counts or frequencies of each combination of categories from the two variables you are comparing.
# # 4.Perform Fisher's Exact Test: Use the scipy.stats.fisher_exact() function to perform Fisher's exact test. This function takes the contingency table as input and returns the odds ratio and p-value.

# # Example DataFrame with two categorical variables
# data = {
#     'Treatment': ['A', 'B', 'A', 'B', 'A'],
#     'Outcome': ['Success', 'Success', 'Failure', 'Success', 'Failure']
# }

# # Convert data to pandas DataFrame
# df = pd.DataFrame(data)

# # Create a contingency table
# contingency_table = pd.crosstab(df['Treatment'], df['Outcome'])

# # Perform Fisher's exact test
# fisher_statistic, p_value = fisher_exact(contingency_table)

# # Print Fisher's exact test results
# print("Fisher's Exact Test Results:")
# print(f"Statistic: {fisher_statistic}")
# print(f"P-value: {p_value}")


# ######################################## DRAFT & OLD FUNCTIONS ########################################

# def produce_simple_table(df, columns=None, filename="output.tex", unicode_latex_mapping=None, title=None, footnotes=None, float_nb_digits=2):
#     """
#     Convert DataFrame to LaTeX table and write it to a .tex file.

#     Parameters:
#         df (DataFrame): The DataFrame containing data.
#         columns (list, optional): The list of columns to include in the table. Defaults to None.
#         filename (str, optional): The name of the .tex file to write. Defaults to "output.tex".
#         unicode_latex_mapping (dict, optional): A dictionary mapping Unicode characters to LaTeX representations. Defaults to None.
#         title (str, optional): The title of the table. Defaults to None.
#         footnotes (dict, optional): A dictionary mapping column or index names to their associated footnotes. Defaults to None.
#         float_nb_digits (int, optional): Number of digits to display for floating-point numbers. Defaults to 3.
#     """
#     # Error handling
#     if df.empty:
#         raise ValueError("DataFrame is empty")
#     if not filename.endswith(".tex"):
#         raise ValueError("Filename must have a .tex extension")
    
#     # Validate columns
#     if columns:
#         #if not all(col in df.columns for col in columns):
#         #    raise ValueError("Specified columns do not exist in the DataFrame")
#         missing_columns = [col for col in columns if col not in df.columns]
#         if missing_columns:
#             raise ValueError(f"Produce Table :: The following columns do not exist in the DataFrame: {', '.join(missing_columns)}")
    
#     # Unicode to LaTeX mapping
#     if unicode_latex_mapping:
#         unicode_mapping = unicode_latex_mapping

#     # Convert DataFrame to LaTeX
#     if columns:
#         # Display the DataFrame with highlighted values
#         #df.style.map(color_cells, subset=columns)
#         # Apply styling to the DataFrame
#         styler = (df[columns]
#                   .style.set_properties(**{"font-weight": "bold /* --dwrap */", "font-size": "12pt"})
#                   #.format(escape=True)
#                   .format_index(escape="latex", axis=1)
#                   .format_index(escape="latex", axis=0)
#                   )
        
#         # Apply background gradient and color mapping
#         styler = (
#             styler
#             .background_gradient(cmap="autumn", subset="fill_percentage", vmin=df["fill_percentage"].min(), vmax=df["fill_percentage"].max(), text_color_threshold=0.5)
#             .background_gradient(cmap="summer", subset=[col for col in columns if col.endswith('_p_value')], vmin=0, vmax=0.1)
#             .format(precision=float_nb_digits, subset=[col for col in columns if not col.endswith('_p_value')])
#             .format(format_ltx_p_value, subset=[col for col in columns if col.endswith('_p_value')])
#             #.map(color_cells, subset=[col for col in columns if col.endswith('_p_value')])
#             )

#         # Convert the styled DataFrame to a LaTeX table
#         latex_table = (
#             styler
#             .to_latex(multicol_align="|c|",hrules=True,convert_css=True)
#         )


#         #latex_table = df[columns].to_latex(escape=True, index=True, float_format=f"%.{float_nb_digits}f")
#     else:
#         styler = df.style
#         styler = (
#             styler
#             .background_gradient(cmap="autumn", subset="fill_percentage", vmin=df["fill_percentage"].min(), vmax=df["fill_percentage"].max(), threshold=99)
#             .background_gradient(cmap="summer", subset=[col for col in columns if col.endswith('_p_value')], vmin=0, vmax=0.1, threshold=0.05)
#             .format(precision=float_nb_digits, subset=[col for col in columns if not col.endswith('_p_value')])
#             .format(format_ltx_p_value, subset=[col for col in columns if col.endswith('_p_value')])
#             )
#         latex_table = styler.to_latex(multicol_align="|c|",hrules=True,convert_css=True)
#         #latex_table = df.to_latex(escape=True, index=True) #, float_format=f"%.{float_nb_digits}f"

#     # Write LaTeX to a .tex file
#     with open(filename, 'w', encoding='utf-8') as f:
#         f.write("\\documentclass{article}\n")
#         f.write("\\usepackage{booktabs}\n")
#         f.write("\\usepackage[symbol]{footmisc}\n")
#         f.write("\\usepackage{adjustbox}\n")
#         f.write("\\usepackage[active,tightpage]{preview}\n")
#         f.write("\\usepackage{varwidth}\n")

#         #for conditional color
#         f.write("\\usepackage[table]{xcolor}\n") ##

#         f.write("\\AtBeginDocument{\\begin{preview}\\begin{varwidth}{\\linewidth}}\n")
#         f.write("\\AtEndDocument{\\end{varwidth}\\end{preview}}\n")
#         f.write("\\begin{document}\n\n")
        
#         # Write \DeclareUnicodeCharacter statements for Greek letters and special symbols
#         if unicode_latex_mapping:
#             for unicode_char, latex_repr in unicode_mapping.items():
#                 latex_table = latex_table.replace(unicode_char, latex_repr)
        
#         # Apply footnotes if it is
#         if footnotes:
#             for name, footnote_text in footnotes.items():
#                 latex_table = latex_table.replace(name, f"{name}\\footnote{{{footnote_text}}}")
        
#         #Apply title if it is
#         if title:
#             f.write(f"\\title{{{title}}}\\date{{}}\\author{{}}\n")
#             f.write("\\maketitle\n\\vspace{-2cm}\n")

#         f.write("\\begin{adjustbox}{width=\\textwidth, max height=\\textheight}\n")
#         f.write(latex_table)
        
#         # End document
#         f.write("\\end{adjustbox}\n")
#         f.write("\\end{document}\n")


# def manova_test(df=None, variables=None, grouping_variable='Group', missing_strategy='remove'): # à finir
#     """
#     Perform Multivariate Analysis of Variance (MANOVA) for multiple continuous variables in a DataFrame.

#     Parameters:
#     - df (pandas.DataFrame): DataFrame with the input data.
#     - variables (list): A list of names of continuous variables for which to perform the test.
#     - grouping_variable (str): The name of the grouping variable in the DataFrame.
#     - missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'.

#     Returns:
#     - dict: Dictionary containing the results of the MANOVA test for each variable.
#     """
#     manova_results = {}

#     # Handle missing values in the input DataFrame
#     df_filtered = handle_missing_values(df=df, variables=variables, strategy=missing_strategy)

#     try:
#         # Construct MANOVA formula string
#         formula = ' + '.join(variables) + f' ~ C({grouping_variable})'

#         # Perform MANOVA
#         manova = MANOVA.from_formula(formula, data=df_filtered)

#         # Extract MANOVA results
#         manova_stats = manova.mv_test()
#         manova_results['statistic'] = manova_stats.stats['Wilks\' lambda']
#         manova_results['p_value'] = manova_stats.p_value
#     except Exception as e:
#         logging.error(f"Error occurred while computing MANOVA: {str(e)}")

#     return manova_results


# def produce_statistics_table_v8(df_statistics, columns=None, filename="output.tex", unicode_latex_mapping=None, title=None, label=None, footnote=None, float_nb_digits=3, table_width=0.8, table_height=0.5):
#     """
#     Convert DataFrame to LaTeX table and write it to a .tex file.

#     Parameters:
#         df_statistics (DataFrame): The DataFrame containing statistics data.
#         columns (list, optional): The list of columns to include in the table. Defaults to None.
#         filename (str, optional): The name of the .tex file to write. Defaults to "output.tex".
#         unicode_latex_mapping (dict, optional): A dictionary mapping Unicode characters to LaTeX representations. Defaults to None.
#         title (str, optional): The title of the table. Defaults to None.
#         label (str, optional): The label of the table. Defaults to None.
#         footnote (str, optional): The footnote for the table. Defaults to None.
#         float_nb_digits (int, optional): Number of digits to display for floating-point numbers. Defaults to 3.
#         table_width (float, optional): Width of the table as a fraction of textwidth. Defaults to 0.9.
#         table_height (float, optional): Maximum height of the table as a fraction of textheight. Defaults to 0.9.
#     """
#     # Error handling
#     if df_statistics.empty:
#         raise ValueError("DataFrame is empty")
#     if not filename.endswith(".tex"):
#         raise ValueError("Filename must have a .tex extension")
    
#     # Validate columns
#     if columns:
#         #if not all(col in df_statistics.columns for col in columns):
#         #    raise ValueError("Specified columns do not exist in the DataFrame")
#         missing_columns = [col for col in columns if col not in df_statistics.columns]
#         if missing_columns:
#             raise ValueError(f"The following columns do not exist in the DataFrame: {', '.join(missing_columns)}")
            
    
#     # Unicode to LaTeX mapping
#     if unicode_latex_mapping:
#         unicode_mapping = unicode_latex_mapping

#     # Convert DataFrame to LaTeX
#     if columns:
#         latex_table = df_statistics[columns].to_latex(escape=True, index=True, float_format=f"%.{float_nb_digits}f")
#     else:
#         latex_table = df_statistics.to_latex(escape=True, index=True, float_format=f"%.{float_nb_digits}f")

#     # Calculate required document size
#     document_width = max(table_width, 0.8)  # Set a minimum width of 0.8 textwidth
#     document_height = max(table_height*0.6, 0.5)  # Set a minimum height of 0.5 textheight

#     # Write LaTeX to a .tex file
#     with open(filename, 'w', encoding='utf-8') as f:
#         f.write("\\documentclass{article}\n")
#         f.write("\\usepackage{booktabs}\n")
#         f.write("\\usepackage{adjustbox}\n")
#         f.write("\\usepackage{geometry}\n")
#         f.write("\\geometry{margin=1in}\n")
#         f.write("\\usepackage{threeparttable}\n")
#         f.write("\\begin{document}\n\n")
        
#         # Write \DeclareUnicodeCharacter statements for Greek letters and special symbols
#         if unicode_latex_mapping:
#             for unicode_char, latex_repr in unicode_mapping.items():
#                 latex_table = latex_table.replace(unicode_char, latex_repr)
        
#         # Write table environment with adjusted width and height
        
#         f.write("\\begin{table}\n")
#         f.write("\\begin{adjustbox}{width=" + str(document_width) + "\\textwidth, max height=" + str(document_height) + "\\textheight}\n")
#         f.write("\\begin{threeparttable}[htbp]\n")
#         f.write("\\centering\n")
#         if title:
#             f.write("\\caption{" + title + "\\tnote{1}}\n")
#         if label:
#             f.write("\\label{tab:"+ label +"}\n")
        
#         f.write(latex_table)
        
#         if footnote:
#             f.write("\\begin{tablenotes}\n")
#             f.write("\\item [1] This is an early draft.\n")
#             f.write("\\end{tablenotes}\n")
#         f.write("\\end{threeparttable}\n\n")
#         f.write("\\end{adjustbox}\n")
#         f.write("\\end{table}\n")
        
#         # End document
#         f.write("\\end{document}\n")


# def conduct_hypothesis_tests_old2(group1, group2, nan_strategy='remove'):
#     """
#     Conduct hypothesis tests between two groups.

#     Parameters:
#     - group1 (pandas.Series): The data for group 1.
#     - group2 (pandas.Series): The data for group 2.
#     - nan_strategy (str): Strategy to handle NaN values. Options: 'remove', 'impute_median', 'impute_mean'.

#     Returns:
#     - dict: Dictionary containing the results of hypothesis tests.
#     """
#     # Handle NaN values
#     if nan_strategy == 'remove':
#         group1 = group1.dropna()
#         group2 = group2.dropna()
#     elif nan_strategy == 'impute_median':
#         group1.fillna(group1.median(), inplace=True)
#         group2.fillna(group2.median(), inplace=True)
#     elif nan_strategy == 'impute_mean':
#         group1.fillna(group1.mean(), inplace=True)
#         group2.fillna(group2.mean(), inplace=True)
#     else:
#         raise ValueError("Invalid nan_strategy. Choose from 'remove', 'impute_median', or 'impute_mean'.")

#     if len(group1) == 0 or len(group2) == 0:
#         raise ValueError("At least one of the groups has no valid data after NaN handling.")

#     # Example of conducting a t-test between two groups
#     t_statistic, t_p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
    
#     # Wilcoxon rank sum test for non-normally distributed data
#     try:
#         wilcoxon_statistic, wilcoxon_p_value = stats.ranksums(group1, group2)
#     except ValueError:
#         wilcoxon_statistic, wilcoxon_p_value = None, None
    
#     # Pearson's Chi-squared test for categorical data
#     # Will need some pretreatment
#     try:
#         # Combine group1 and group2 into a single array
#         combined_data = np.concatenate([group1, group2])

#         # Count frequencies of unique values to obtain observed frequencies
#         observed_frequencies = Counter(combined_data)

#         # Compute expected frequencies under the assumption of independence
#         total_count = len(combined_data)
#         unique_values = len(observed_frequencies)
#         expected_frequency = total_count / unique_values

#         # Compute chi-square statistic
#         chi2_statistic = sum(((observed_frequencies[val] - expected_frequency) ** 2) / expected_frequency
#                             for val in observed_frequencies)

#         # Compute degrees of freedom
#         degrees_of_freedom = unique_values - 1

#         # Compute p-value using chi-square distribution
#         chi2_p_value = 1 - stats.chi2.cdf(chi2_statistic, degrees_of_freedom)

#         # Return the results in a dictionary
#         hypothesis_test_results = {
#             'chi2_test': {'statistic': chi2_statistic, 'p_value': chi2_p_value}
#         }
#         print("#######\nhypothesis_test_results:",hypothesis_test_results)
#         chi2_statistic, chi2_p_value = stats.chisquare(f_obs=observed_frequencies, f_exp=expected_frequency,ddof=degrees_of_freedom)
#         print("stats.chisquare --> chi2_statistic, chi2_p_value:",chi2_statistic, chi2_p_value)
#     except ValueError:
#         print("error chi")
#         chi2_statistic, chi2_p_value = None, None
    
#     # Fisher's exact test for contingency tables (2x2 tables)
#     try:
#         contingency_table = np.array([[group1.sum(), len(group1) - group1.sum()], 
#                                       [group2.sum(), len(group2) - group2.sum()]])
#         fisher_statistic, fisher_p_value = stats.fisher_exact(contingency_table)
#     except ValueError:
#         print("error fisher")
#         fisher_statistic, fisher_p_value = None, None
    
#     # Return the results in a dictionary
#     hypothesis_test_results = {
#         't_test': {'statistic': t_statistic, 'p_value': t_p_value},
#         'wilcoxon_test': {'statistic': wilcoxon_statistic, 'p_value': wilcoxon_p_value},
#         'chi2_test': {'statistic': chi2_statistic, 'p_value': chi2_p_value},
#         'fisher_test': {'statistic': fisher_statistic, 'p_value': fisher_p_value}
#     }
#     return hypothesis_test_results


# def conduct_hypothesis_tests_old(group1, group2):
#     """
#     Conduct hypothesis tests between two groups.

#     Parameters:
#     - group1 (pandas.Series): The data for group 1.
#     - group2 (pandas.Series): The data for group 2.

#     Returns:
#     - dict: Dictionary containing the results of hypothesis tests.
#     """
#     # Example of conducting a t-test between two groups
#     t_statistic, t_p_value = stats.ttest_ind(group1, group2)
    
#     # Wilcoxon rank sum test for non-normally distributed data
#     try:
#         wilcoxon_statistic, wilcoxon_p_value = stats.ranksums(group1, group2)
#     except ValueError:
#         wilcoxon_statistic, wilcoxon_p_value = None, None
    
#     # Pearson's Chi-squared test for categorical data
#     try:
#         chi2_statistic, chi2_p_value = stats.chisquare(group1, group2)
#     except ValueError:
#         chi2_statistic, chi2_p_value = None, None
    
#     # Fisher's exact test for contingency tables (2x2 tables)
#     try:
#         fisher_statistic, fisher_p_value = stats.fisher_exact([[group1.sum(), len(group1) - group1.sum()], [group2.sum(), len(group2) - group2.sum()]])
#     except ValueError:
#         fisher_statistic, fisher_p_value = None, None
    
#     # Return the results in a dictionary
#     hypothesis_test_results = {
#         't_test': {'statistic': t_statistic, 'p_value': t_p_value},
#         'wilcoxon_test': {'statistic': wilcoxon_statistic, 'p_value': wilcoxon_p_value},
#         'chi2_test': {'statistic': chi2_statistic, 'p_value': chi2_p_value},
#         'fisher_test': {'statistic': fisher_statistic, 'p_value': fisher_p_value}
#     }
#     return hypothesis_test_results

# def conduct_hypothesis_tests_old(group1, group2, target , cat_var=False,nan_strategy='remove'):
#     """
#     Conduct hypothesis tests between two groups.

#     Parameters:
#     - group1 (pandas.Series): The data for group 1.
#     - group2 (pandas.Series): The data for group 2.
#     - nan_strategy (str): Strategy to handle NaN values. Options: 'remove', 'impute_median', 'impute_mean'.

#     Returns:
#     - dict: Dictionary containing the results of hypothesis tests.
#     """
#     # Handle NaN values
#     if nan_strategy == 'remove':
#         group1 = group1.dropna()
#         group2 = group2.dropna()
#     elif nan_strategy == 'impute_median':
#         group1.fillna(group1.median(), inplace=True)
#         group2.fillna(group2.median(), inplace=True)
#     elif nan_strategy == 'impute_mean':
#         group1.fillna(group1.mean(), inplace=True)
#         group2.fillna(group2.mean(), inplace=True)
#     else:
#         raise ValueError("Invalid nan_strategy. Choose from 'remove', 'impute_median', or 'impute_mean'.")

#     if len(group1) == 0 or len(group2) == 0:
#         raise ValueError("At least one of the groups has no valid data after NaN handling.")

#     # Check data types of input variables
#     if cat_var:
#         # Categorical variables
#         try:
#             # Compute observed frequencies
#             group1_counts = Counter(group1)
#             group2_counts = Counter(group2)

#             # Combine counts
#             all_values = set(group1_counts.keys()).union(set(group2_counts.keys()))

#             observed_frequencies = {
#                 val: [group1_counts.get(val, 0), group2_counts.get(val, 0)] for val in all_values
#             }
#             print("\n******************\n osbs freq: ",observed_frequencies)

#             # Compute chi-squared statistic
#             chi2_statistic, chi2_p_value = stats.chisquare(list(observed_frequencies.values()))

#             # Store results with category labels
#             chi2_results = {
#                 category: {'statistic': stat, 'p_value': p_value}
#                 for category, (stat, p_value) in zip(all_values, zip(chi2_statistic, chi2_p_value))
#             }
#         except ValueError:
#             chi2_results = {}

#         # Return the results in a dictionary
#         return {'chi2_tests': chi2_results}
#     else:
#         # Numerical variables
#         # Example of conducting a t-test between two groups
#         t_statistic, t_p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
        
#         # Wilcoxon rank sum test for non-normally distributed data
#         try:
#             wilcoxon_statistic, wilcoxon_p_value = stats.ranksums(group1, group2)
#         except ValueError:
#             wilcoxon_statistic, wilcoxon_p_value = None, None
        
#         # Fisher's exact test for contingency tables (2x2 tables)
#         try:
#             contingency_table = np.array([[group1.sum(), len(group1) - group1.sum()], 
#                                           [group2.sum(), len(group2) - group2.sum()]])
#             fisher_statistic, fisher_p_value = stats.fisher_exact(contingency_table)
#         except ValueError:
#             fisher_statistic, fisher_p_value = None, None

#         # Return the results in a dictionary
#         return {
#             't_test': {'statistic': t_statistic, 'p_value': t_p_value},
#             'wilcoxon_test': {'statistic': wilcoxon_statistic, 'p_value': wilcoxon_p_value},
#             'fisher_test': {'statistic': fisher_statistic, 'p_value': fisher_p_value}
#         }


# def conduct_hypothesis_tests(group1, group2, target='sex' , cat_var=False,nan_strategy='remove'):
#     """
#     Conduct hypothesis tests between two groups.

#     Parameters:
#     - group1 (pandas.Series): The data for group 1.
#     - group2 (pandas.Series): The data for group 2.
#     - nan_strategy (str): Strategy to handle NaN values. Options: 'remove', 'impute_median', 'impute_mean'.

#     Returns:
#     - dict: Dictionary containing the results of hypothesis tests.
#     """
#     # Handle NaN values
#     if nan_strategy == 'remove':
#         group1 = group1.dropna()
#         group2 = group2.dropna()
#     elif nan_strategy == 'impute_median':
#         group1.fillna(group1.median(), inplace=True)
#         group2.fillna(group2.median(), inplace=True)
#     elif nan_strategy == 'impute_mean':
#         group1.fillna(group1.mean(), inplace=True)
#         group2.fillna(group2.mean(), inplace=True)
#     else:
#         raise ValueError("Invalid nan_strategy. Choose from 'remove', 'impute_median', or 'impute_mean'.")

#     if len(group1) == 0 or len(group2) == 0:
#         raise ValueError("At least one of the groups has no valid data after NaN handling.")

#     # Check data types of input variables
#     if cat_var:
#         # Categorical variables
#         try:
#             # Compute observed frequencies
#             group1_counts = Counter(group1)
#             group2_counts = Counter(group2)

#             # Combine counts
#             all_values = set(group1_counts.keys()).union(set(group2_counts.keys()))

#             observed_frequencies = {
#                 val: [group1_counts.get(val, 0), group2_counts.get(val, 0)] for val in all_values
#             }
#             print("\n******************\n osbs freq: ",observed_frequencies)

#             # Compute chi-squared statistic
#             chi2_statistic, chi2_p_value = stats.chisquare(list(observed_frequencies.values()))

#             # Store results with category labels
#             chi2_results = {
#                 category: {'statistic': stat, 'p_value': p_value}
#                 for category, (stat, p_value) in zip(all_values, zip(chi2_statistic, chi2_p_value))
#             }
#         except ValueError:
#             chi2_results = {}

#         # Return the results in a dictionary
#         return {'chi2_tests': chi2_results}
#     else:
#         # Numerical variables
#         # Example of conducting a t-test between two groups
#         t_statistic, t_p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
        
#         # Wilcoxon rank sum test for non-normally distributed data
#         try:
#             wilcoxon_statistic, wilcoxon_p_value = stats.ranksums(group1, group2)
#         except ValueError:
#             wilcoxon_statistic, wilcoxon_p_value = None, None
        
#         # Fisher's exact test for contingency tables (2x2 tables)
#         try:
#             contingency_table = np.array([[group1.sum(), len(group1) - group1.sum()], 
#                                           [group2.sum(), len(group2) - group2.sum()]])
#             fisher_statistic, fisher_p_value = stats.fisher_exact(contingency_table)
#         except ValueError:
#             fisher_statistic, fisher_p_value = None, None

#         # Return the results in a dictionary
#         return {
#             't_test': {'statistic': t_statistic, 'p_value': t_p_value},
#             'wilcoxon_test': {'statistic': wilcoxon_statistic, 'p_value': wilcoxon_p_value},
#             'fisher_test': {'statistic': fisher_statistic, 'p_value': fisher_p_value}
#         }
    

# def process_test_old(df, test_name, test_info, config=None): # config,
#     """
#     Process the statistical test based on the config file data.

#     Parameters:
#     - df (pandas.DataFrame): DataFrame containing the data.
#     - test_name (str): Name of the hypothesis test being processed.
#     - test_info (dict): Information about the hypothesis test, including test type, variables, groups, etc.
#     - config (dict, optional): Additional configuration parameters. Default is None.

#     Returns:
#     - tuple: A tuple containing the results of the comparison, including DataFrame 1, DataFrame 2, common row identifier name, filtered comparison result, 
#     rows only in DataFrame 1, rows only in DataFrame 2, modified common row identifiers, and the output file path. Returns None if an error occurs.

#     Raises:
#     - None
#     """
#     try:
#         print("#######################\n")
#         test_name = test_name
#         logging.info(f"Processing hypothesis testing: {test_name}")
#         if test_info.get("TEST_TYPE"):
#             test_type = test_info.get("TEST_TYPE")

#         # VARIABLES TO ANALYZE
#         if test_info.get("VARIABLES"):
#             variables = test_info.get("VARIABLES")
#             print("variables:", variables)
#         else:
#             variables = df.columns.tolist()
        
#         # MISSING STRATEGY FOR IMPUTATION
#         missing_strategy = test_info.get("MISSING_STRATEGY")
        
#         # TARGET GROUP
#         if test_info.get("GROUP_1"):
#             group_1 = process_group(info_group=test_info.get("GROUP_1"),df=df)
#             print(group_1.head())

#         # CONTROL GROUP
#         if test_info.get("GROUP_2"):
#             group_2 = process_group(info_group=test_info.get("GROUP_2"),df=df)
#             print(group_2.head())
 
#         print("\nVERIFICATIONS\n")
#         if test_info.get("TARGET_VARIABLE"):
#             print("\nTARGET_VARIABLE\n")
#             target_var = test_info.get("TARGET_VARIABLE")

#             result = make_test(test_type=test_type,df=df,variables=variables,target_variable=target_var,missing_strategy=missing_strategy)
        
#         elif test_info.get("GROUP_1") and test_info.get("GROUP_2"):
#             print("\ngroup_1 and group_2\n")
#             result = make_test(test_type=test_type,df=df,variables=variables,target_group=group_1,control_group=group_2,missing_strategy=missing_strategy)
#             print(f"{test_info.get('GROUP_1')} and {test_info.get('GROUP_2')}\n Results: {result} ###\n")
            
#         else:
#             return None

#         return result 
#     except Exception as e:
#         logging.error(f"An unexpected error occurred for test: {test_name}. Error: {str(e)}")
#         return None
    
# def chi2_test_old(df, variables, target_variable,missing_strategy=None):
#     """
#     Perform a chi-square test of independence between categorical variables and a target variable.

#     Parameters:
#     - df (pandas.DataFrame): The DataFrame containing the data.
#     - variables (list): A list of column names representing categorical variables.
#     - target_variable (str): The name of the target variable.

#     Returns:
#     - dict: A dictionary containing the chi-square test results for each categorical variable.
#     """
#     chi2_results = {}

#     # Handle missing values for all variables once before the loop
#     if missing_strategy: 
#         df = handle_missing_values(df, variables, target_variable, missing_strategy).copy()

#     for cat_var in variables:
#         contingency_table = pd.crosstab(df[cat_var], df[target_variable])
#         chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
#         chi2_results[cat_var] = {'chi2_statistic': chi2, 'p_value': p_value}

#     return chi2_results

# ########## numerical variables

# def wilcoxon_rank_sum_test(df, variables, target_variable, missing_strategy='remove'):
#     """
#     Perform Wilcoxon rank sum test for continuous variables in a DataFrame.

#     Parameters:
#     - df (pandas.DataFrame): The DataFrame containing the data.
#     - variables (list): A list of names of continuous variables for which to perform the test.
#     - target_variable (str): The name of the target variable.
#     - missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'.

#     Returns:
#     - dict: Dictionary containing the results of the Wilcoxon rank sum test for each variable.
#     """
#     # Dictionary to store the results of the Wilcoxon rank sum test
#     wilcoxon_results = {}
#     # Handle missing values
#     df_filtered = handle_missing_values(df, variables, target_variable, missing_strategy)
#     print(df_filtered.head())
#     # Iterate over each variable in the list
#     for variable in variables:
#         try:
#             # Check if the variable exists in the DataFrame
#             if variable not in df.columns:
#                 logging.warning(f"Variable '{variable}' not found in the DataFrame.")
#                 continue
#             # Select data for the two groups based on the target variable
#             control_group = df_filtered.loc[df_filtered[target_variable] == 0, variable]
#             target_group = df_filtered.loc[df_filtered[target_variable] == 1, variable]

#             # Check if there is sufficient data for the test
#             if len(control_group) < 2 or len(target_group) < 2:
#                 logging.warning(f"Insufficient data to perform Wilcoxon rank sum test for variable '{variable}'.")
#                 continue

#             # Perform Wilcoxon rank sum test
#             wilcoxon_statistic, wilcoxon_p_value = stats.ranksums(control_group, target_group)

#             # Store the results in the dictionary
#             wilcoxon_results[variable] = {'statistic': wilcoxon_statistic, 'p_value': wilcoxon_p_value}
#         except Exception as e:
#             # Log any errors that occur during computation
#             logging.error(f"Error occurred while computing Wilcoxon rank sum test for variable '{variable}': {str(e)}")

#     return wilcoxon_results


# def t_test(df, variables, target_variable, missing_strategy='remove'):
#     """
#     Compute a t-test between two groups based on given variables.

#     Parameters:
#     - df (pandas.DataFrame): The raw dataset.
#     - variables (list): A list of variable names to test.
#     - target_variable (str): The name of the target variable.
#     - missing_strategy (str): Strategy to handle NaN values. Options: 'remove', 'impute_median', 'impute_mean', knn.

#     Returns:
#     - dict: Dictionary containing the results of the t-test for each variable.
#     """
#     t_test_results = {}
#     # Handle missing values and filter the dataset
#     df_filtered = handle_missing_values(df, variables, target_variable, missing_strategy)

#     for variable in variables:
#         try:
#             # Separate data into control and target groups based on the target variable
#             control_group = df_filtered.loc[df_filtered[target_variable] == 0, variable]
#             target_group = df_filtered.loc[df_filtered[target_variable] == 1, variable]

#             # Perform t-test
#             t_statistic, t_p_value = stats.ttest_ind(control_group, target_group, nan_policy='omit')
      
#             # Store the results in a dictionary
#             t_test_results[variable] = {
#                 't_statistic': t_statistic,
#                 'p_value': t_p_value
#             }
#         except Exception as e:
#             logging.error(f"Error occurred while computing t-test for variable '{variable}': {str(e)}")

#     return t_test_results

# def produce_table_v1(df, columns=None, filename="output.tex", unicode_latex_mapping=None, title=None, footnotes=None, float_nb_digits=3):
#     """
#     Convert DataFrame to LaTeX table and write it to a .tex file.

#     Parameters:
#         df (DataFrame): The DataFrame containing data.
#         columns (list, optional): The list of columns to include in the table. Defaults to None.
#         filename (str, optional): The name of the .tex file to write. Defaults to "output.tex".
#         unicode_latex_mapping (dict, optional): A dictionary mapping Unicode characters to LaTeX representations. Defaults to None.
#         title (str, optional): The title of the table. Defaults to None.
#         footnotes (dict, optional): A dictionary mapping column or index names to their associated footnotes. Defaults to None.
#         float_nb_digits (int, optional): Number of digits to display for floating-point numbers. Defaults to 3.
#     """
#     # Error handling
#     if df.empty:
#         raise ValueError("DataFrame is empty")
#     if not filename.endswith(".tex"):
#         raise ValueError("Filename must have a .tex extension")
    
#     # Validate columns
#     if columns:
#         #if not all(col in df.columns for col in columns):
#         #    raise ValueError("Specified columns do not exist in the DataFrame")
#         missing_columns = [col for col in columns if col not in df.columns]
#         if missing_columns:
#             raise ValueError(f"Produce Table :: The following columns do not exist in the DataFrame: {', '.join(missing_columns)}")
    
#     # Unicode to LaTeX mapping
#     if unicode_latex_mapping:
#         unicode_mapping = unicode_latex_mapping

#     # Convert DataFrame to LaTeX
#     if columns:
#         # Display the DataFrame with highlighted values
#         #df.style.map(color_cells, subset=columns)
        
#         latex_table = df[columns].to_latex(escape=True, index=True, float_format=f"%.{float_nb_digits}f")
#     else:
#         latex_table = df.to_latex(escape=True, index=True, float_format=f"%.{float_nb_digits}f")

#     # Write LaTeX to a .tex file
#     with open(filename, 'w', encoding='utf-8') as f:
#         f.write("\\documentclass{article}\n")
#         f.write("\\usepackage{booktabs}\n")
#         f.write("\\usepackage[symbol]{footmisc}\n")
#         f.write("\\usepackage{adjustbox}\n")
#         f.write("\\usepackage[active,tightpage]{preview}\n")
#         f.write("\\usepackage{varwidth}\n")

#         #for conditional color
#         #f.write("\\usepackage[table]{xcolor}\n") ##

#         f.write("\\AtBeginDocument{\\begin{preview}\\begin{varwidth}{\\linewidth}}\n")
#         f.write("\\AtEndDocument{\\end{varwidth}\\end{preview}}\n")
#         f.write("\\begin{document}\n\n")
        
#         # Write \DeclareUnicodeCharacter statements for Greek letters and special symbols
#         if unicode_latex_mapping:
#             for unicode_char, latex_repr in unicode_mapping.items():
#                 latex_table = latex_table.replace(unicode_char, latex_repr)
        
#         # Apply footnotes if it is
#         if footnotes:
#             for name, footnote_text in footnotes.items():
#                 latex_table = latex_table.replace(name, f"{name}\\footnote{{{footnote_text}}}")
        
#         #Apply title if it is
#         if title:
#             f.write(f"\\title{{{title}}}\\date{{}}\\author{{}}\n")
#             f.write("\\maketitle\n\\vspace{-2cm}\n")

#         f.write("\\begin{adjustbox}{width=\\textwidth, max height=\\textheight}\n")
#         f.write(latex_table)
        
#         # End document
#         f.write("\\end{adjustbox}\n")
#         f.write("\\end{document}\n")
