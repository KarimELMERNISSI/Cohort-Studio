"""
Data Analyzer Module.

Provides the DataAnalyzer class for analyzing and categorizing DataFrame columns.
This is a consolidated module combining functionality from home.py and visualization_utils.py.
"""
import pandas as pd
import numpy as np


class DataAnalyzer:
    """
    Helper class to analyze and validate data columns.
    
    Categorizes columns into:
    - numeric_cols: Numeric columns (excluding low cardinality)
    - categorical_cols: Categorical and object columns
    - date_cols: Date/datetime columns
    - binary_cols: Columns with exactly 2 unique values
    - low_cardinality_numeric_cols: Numeric columns with <= 10 unique values
    - high_cardinality_cat_cols: Categorical columns with many unique values
    - timedelta_cols: Time interval columns
    """
    
    def __init__(self, df):
        """
        Initialize the analyzer with a DataFrame.
        
        Args:
            df: pandas DataFrame to analyze
        """
        self.df = df
        self.date_formats = {}
        self.analyze_columns()
    
    def refresh(self, df):
        """
        Refresh the column metadata based on the latest DataFrame state.
        
        Args:
            df: Updated pandas DataFrame
        """
        self.df = df
        self.date_formats = {}
        self.analyze_columns()
    
    def analyze_columns(self):
        """Analyze and categorize columns by data type."""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_cols = self._detect_date_columns()
        self.binary_cols, self.num_binary_cols, self.non_num_binary_cols = self._detect_binary_columns()
        self.low_cardinality_numeric_cols = self._detect_low_cardinality_numeric_columns()
        self.non_binary_low_cardinality_numeric_cols = [
            col for col in self.low_cardinality_numeric_cols 
            if col not in self.binary_cols
        ]
        self.timedelta_cols = self.df.select_dtypes(include=[np.timedelta64]).columns.tolist()
        
        # Remove low-cardinality numeric columns from numeric_cols and add to categorical_cols
        for col in self.low_cardinality_numeric_cols:
            if col in self.numeric_cols:
                self.numeric_cols.remove(col)
            if col not in self.categorical_cols and col not in self.binary_cols:
                self.categorical_cols.append(col)
        
        # Remove date and time interval columns from categorical_cols and numeric_cols
        for col in self.date_cols + self.timedelta_cols:
            if col in self.categorical_cols:
                self.categorical_cols.remove(col)
            if col in self.numeric_cols:
                self.numeric_cols.remove(col)

        self.high_cardinality_cat_cols = self._detect_high_cardinality_cat_columns()
    
    def _detect_date_columns(self):
        """Detect columns that are likely dates, with additional checks for accuracy."""
        date_cols = []
        formats = [
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%d %b %Y",
            "%b %d, %Y",
            "%Y/%m/%d",
            "%d-%b-%Y",
            "%Y/%m/%d %H:%M:%S",
            "%H:%M:%S",
        ]
        
        for col in self.df.columns:
            # Add proper date format columns
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                date_cols.append(col)
                continue

            # Proceed only if column is of object or string type
            if pd.api.types.is_object_dtype(self.df[col]):
                is_date_column = False
                
                non_null_values = self.df[col].dropna().astype(str)
                if len(non_null_values) == 0:
                    continue
                    
                sample_size = min(10, len(non_null_values))
                sample_values = non_null_values.sample(
                    n=sample_size, 
                    replace=(sample_size > len(non_null_values))
                )
                
                date_regex = r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4}[-/]\d{1,2}[-/]\d{1,2} \d{1,2}:\d{2}(:\d{2})?)'
                
                # Check for keyword strong signals in column name
                name_signal = any(x in col.upper() for x in ['DATE', 'DOB', 'BIRTH', 'TIME', 'YEAR', 'MONTH'])
                
                # Check match rate in sample (allow some garbage)
                matches_regex = sample_values.str.match(date_regex).mean() > 0.6  # >60% match regex
                
                if matches_regex or name_signal:
                    # Try specific formats first
                    best_format = None
                    min_nat_count = len(self.df) + 1
                    
                    # Heuristic: Check for dayfirst (13/05/2021)
                    # If we see day > 12 at position 0, it's Day First.
                    is_dayfirst_strong_signal = False
                    for val in sample_values:
                        import re
                        if isinstance(val, str):
                            m = re.match(r'(\d{1,2})[-/](\d{1,2})[-/]\d{2,4}', val)
                            if m:
                                d, m_val = int(m.group(1)), int(m.group(2))
                                if d > 12:
                                    is_dayfirst_strong_signal = True
                                    break
                    
                    if is_dayfirst_strong_signal:
                        # Prioritize Day-First formats
                        priority_formats = ["%d/%m/%Y", "%d-%m-%Y"] + formats
                    else:
                        priority_formats = formats

                    for fmt in priority_formats:
                        try:
                            converted = pd.to_datetime(self.df[col], format=fmt, errors='coerce')
                            nat_count = converted.isna().sum()
                            
                            # If this format parses everything (or mostly everything), it's a winner
                            if nat_count < min_nat_count:
                                min_nat_count = nat_count
                                best_format = fmt
                                
                            if nat_count == 0: # Perfect match
                                break
                        except Exception:
                            continue
                    
                    if best_format:
                         # Threshold: More than 40% valid dates (relaxed from 50% to catch messy columns)
                         converted = pd.to_datetime(self.df[col], format=best_format, errors='coerce')
                         if converted.notna().sum() > len(self.df[col]) * 0.4:
                            date_cols.append(col)
                            self.date_formats[col] = best_format
                            is_date_column = True
                    # Fallback: exact format failed, but name suggested Date?
                    # Try generic parser
                    elif name_signal:
                         converted = pd.to_datetime(self.df[col], errors='coerce')
                         if converted.notna().sum() > len(self.df[col]) * 0.4:
                            date_cols.append(col)
                            self.date_formats[col] = "mixed"
                            is_date_column = True

                    if not is_date_column:
                        try:
                            # Fallback to dayfirst=True if signaled
                            converted = pd.to_datetime(self.df[col], dayfirst=is_dayfirst_strong_signal, errors='coerce')
                            non_na_dates = converted.notna().sum()
                            if non_na_dates > len(self.df[col]) * 0.5:
                                date_cols.append(col)
                        except Exception:
                            continue
        
        return date_cols

    def _detect_binary_columns(self):
        """
        Detect columns with only two unique values, including numeric columns.
        
        Returns:
            tuple: (all_binary, numeric_binary, non_numeric_binary)
        """
        all_binary = [col for col in self.df.columns if self.df[col].nunique() == 2]
        num_binary = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                      if self.df[col].nunique() == 2]
        non_num_binary = [col for col in self.df.select_dtypes(exclude=[np.number]).columns 
                          if self.df[col].nunique() == 2]
        return all_binary, num_binary, non_num_binary
    
    def _detect_low_cardinality_numeric_columns(self):
        """Detect numeric columns with <= 10 unique values to treat as categorical."""
        return [col for col in self.numeric_cols if self.df[col].nunique() <= 10]
    
    def _detect_high_cardinality_cat_columns(self):
        """Detect categorical columns with many unique values."""
        return [col for col in self.categorical_cols 
                if self.df[col].nunique() > 0.5 * len(self.df)]

    def get_suitable_columns(self, plot_type):
        """
        Get suitable columns for different plot types.
        
        Args:
            plot_type: Type of plot (e.g., 'Histogram', 'Box Plot', 'Scatter Plot')
            
        Returns:
            dict: Mapping of axis/parameter to suitable columns
        """
        num_vars = sorted(set(self.numeric_cols))
        date_cols = sorted(set(self.date_cols + self.timedelta_cols))
        cat_vars = sorted(set(
            col for col in self.categorical_cols + self.binary_cols + self.low_cardinality_numeric_cols 
            if col not in self.high_cardinality_cat_cols
        ))

        plot_configs = {
            "Histogram": {"x": num_vars, "color": cat_vars},
            "Box Plot": {"y": num_vars, "x": cat_vars},
            "Violin Plot": {"y": num_vars, "x": cat_vars},
            "Scatter Plot": {"x": num_vars, "y": num_vars, "color": cat_vars, "size": num_vars},
            "Line Plot": {"x": date_cols, "y": num_vars, "color": cat_vars},
            "Bar Plot": {"x": cat_vars, "y": num_vars, "color": cat_vars},
            "Pie Chart": {
                "names": [col for col in self.categorical_cols if self.df[col].nunique() <= 10],
                "values": num_vars
            },
            "Correlation Matrix": {
                "numeric": num_vars,
                "categorical": [col for col in cat_vars 
                               if all(pd.api.types.is_numeric_dtype(val) for val in self.df[col].unique())]
            },
            "Bland-Altman Plot": {"method1": num_vars, "method2": num_vars},
            "ROC Curve": {
                "true_class": self.binary_cols + [c for c in cat_vars if self.df[c].nunique() == 2],
                "score": num_vars
            },
            "Clustermap": {
                "categorical": cat_vars + self.high_cardinality_cat_cols,
                "numeric": num_vars,
                "identifiers": sorted(self.df.columns.tolist())
            },
            "Sunburst Chart": {
                "categorical": cat_vars + self.high_cardinality_cat_cols
            },
            "Icicle Chart": {
                "categorical": cat_vars + self.high_cardinality_cat_cols
            }
        }
        
        return plot_configs.get(plot_type, {})
