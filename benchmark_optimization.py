
import time
import pandas as pd
import numpy as np
import duckdb
# import streamlit as st

def generate_data(n_rows=100000, n_cols=20):
    print(f"Generating data with {n_rows} rows and {n_cols} columns...")
    data = {f'col_{i}': np.random.rand(n_rows) for i in range(n_cols)}
    data['group_col'] = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows)
    return pd.DataFrame(data)

def pandas_describe(df):
    start = time.time()
    res = df.describe()
    end = time.time()
    return end - start, res

def duckdb_describe(df):
    start = time.time()
    con = duckdb.connect()
    con.register('df', df)
    
    # DuckDB SUMMARIZE is the equivalent of describe
    res = con.execute("SUMMARIZE df").df()
    end = time.time()
    return end - start, res

def pandas_groupby_describe(df, group_col):
    start = time.time()
    # Simulating what home.py does: iterating and describing
    results = {}
    for name, group in df.groupby(group_col):
        results[name] = group.describe()
    end = time.time()
    return end - start, results

def duckdb_groupby_describe(df, group_col):
    start = time.time()
    con = duckdb.connect()
    con.register('df', df)
    
    # DuckDB doesn't have a direct "GROUP BY SUMMARIZE" that returns a nice table for all columns.
    # We have to construct it.
    # However, we can use UNPIVOT or similar to calculate stats for all columns grouped by group_col.
    
    numeric_cols = [c for c in df.columns if c != group_col]
    
    # This is a complex query to replicate "describe" for all columns per group
    # Strategy: Unpivot numeric columns, then group by group_col and variable_name
    
    unpivot_query = f"""
    WITH unpivoted AS (
        UNPIVOT df
        ON {', '.join(numeric_cols)}
        INTO
            NAME variable
            VALUE value
    )
    SELECT 
        {group_col},
        variable,
        COUNT(value) as count,
        AVG(value) as mean,
        STDDEV(value) as std,
        MIN(value) as min,
        QUANTILE_CONT(value, 0.25) as q25,
        QUANTILE_CONT(value, 0.50) as q50,
        QUANTILE_CONT(value, 0.75) as q75,
        MAX(value) as max
    FROM unpivoted
    GROUP BY {group_col}, variable
    ORDER BY {group_col}, variable
    """
    
    res = con.execute(unpivot_query).df()
    end = time.time()
    return end - start, res

def pandas_corr(df):
    start = time.time()
    # Select only numeric
    numeric_df = df.select_dtypes(include=[np.number])
    res = numeric_df.corr()
    end = time.time()
    return end - start, res

def duckdb_corr(df):
    start = time.time()
    con = duckdb.connect()
    con.register('df', df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # DuckDB correlation matrix
    # We can use the correlation matrix function if available or construct it
    # DuckDB doesn't have a built-in "CORR MATRIX" function that returns a pivot table directly easily without extension?
    # Actually, let's check if we can do it efficiently.
    # SELECT CORR(col1, col2) ...
    
    # For benchmarking, let's try to generate the query for all pairs
    selects = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i:]:
            selects.append(f"CORR({c1}, {c2}) as \"{c1}:{c2}\"")
            
    query = f"SELECT {', '.join(selects)} FROM df"
    res = con.execute(query).df()
    end = time.time()
    return end - start, res

if __name__ == "__main__":
    df = generate_data(n_rows=1000000, n_cols=20)
    
    print("\n--- Benchmarking Describe (Global) ---")
    t_pd, _ = pandas_describe(df)
    print(f"Pandas: {t_pd:.4f} s")
    
    t_dd, _ = duckdb_describe(df)
    print(f"DuckDB (SUMMARIZE): {t_dd:.4f} s")
    
    print("\n--- Benchmarking GroupBy Describe ---")
    t_pd_g, _ = pandas_groupby_describe(df, 'group_col')
    print(f"Pandas (Iterative): {t_pd_g:.4f} s")
    
    t_dd_g, _ = duckdb_groupby_describe(df, 'group_col')
    print(f"DuckDB (Unpivot + Agg): {t_dd_g:.4f} s")
    
    print("\n--- Benchmarking Correlation Matrix ---")
    t_pd_c, _ = pandas_corr(df)
    print(f"Pandas: {t_pd_c:.4f} s")
    
    t_dd_c, _ = duckdb_corr(df)
    print(f"DuckDB (Pairwise SQL): {t_dd_c:.4f} s")
