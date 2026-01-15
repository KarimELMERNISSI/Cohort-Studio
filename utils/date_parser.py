
import pandas as pd
import numpy as np

def smart_parse_dates(series):
    """
    Parses dates by enforcing a single consistent format across the column.
    Detects components (Day, Month, Year) by analyzing global numeric ranges.
    
    Strategy:
    1. Identify Separator & Split Strings
    2. Analyze each column position (0, 1, 2)
    3. Deduction Rules:
       - Value > 31 => YEAR
       - Value > 12 (and not Year) => DAY
       - Remaining => MONTH
    4. Reconstruct Date
    """
    if series.empty:
        return series, ["Empty Column"]

    # -------------------------------------------------------------
    # 1. EXCEL SERIALS / NUMERIC HANDLING
    # -------------------------------------------------------------
    try:
        numeric_series = pd.to_numeric(series, errors='coerce')
        # If > 50% are valid numbers, chance of Serial Date
        if numeric_series.notna().sum() > 0.5 * len(series):
            # Check validity range (e.g. year > 1900)
            res_serial = pd.to_datetime(numeric_series, unit='D', origin='1899-12-30', errors='coerce')
            if (res_serial.dt.year > 1900).sum() > 0.1 * len(series):
                return res_serial, ["Excel Serial Detected"]
    except:
        pass

    # -------------------------------------------------------------
    # 2. STRICT STRING STRUCTURE ANALYSIS
    # -------------------------------------------------------------
    try:
        # Work on non-null copy
        s = series.dropna().astype(str).copy()
        if s.empty: 
            return pd.to_datetime(series, errors='coerce'), ["Empty/All-Null Strings"]

        # Normalize Separators
        s = s.str.replace(r'[-.]', '/', regex=True)
        
        # Split
        splits = s.str.split('/', expand=True)
        
        # Must have at least 3 parts for standard dates
        if splits.shape[1] < 3:
             # Fallback for weird formats? Or try generic parser?
             # User requested "rebuild date properly", implying standard dates.
             # Let's fallback to generic for safety if split fails.
             return pd.to_datetime(series, errors='coerce'), ["Fallback: Non-standard separators"]

        # Convert cols to numeric
        cols = []
        for i in range(3):
            cols.append(pd.to_numeric(splits[i], errors='coerce'))

        p0, p1, p2 = cols[0], cols[1], cols[2]
        
        # Global Range Analysis
        p0_max = p0.max(skipna=True)
        p1_max = p1.max(skipna=True)
        p2_max = p2.max(skipna=True)
        
        # --- IDENTIFY YEAR ---
        year_idx = -1
        # Priority: If any column goes way above 31 (e.g. 1999, 2020)
        # Check standard positions first to be safe (0 or 2)
        if p0_max > 31: year_idx = 0
        elif p2_max > 31: year_idx = 2
        elif p1_max > 31: year_idx = 1
        
        # --- IDENTIFY DAY / MONTH ---
        day_idx = -1
        month_idx = -1
        
        if year_idx != -1:
            # We found the year. Now distinguish Day vs Month in remaining cols.
            rem = [i for i in [0,1,2] if i != year_idx]
            a, b = rem[0], rem[1]
            max_a = cols[a].max(skipna=True)
            max_b = cols[b].max(skipna=True)
            
            if max_a > 12: # Must be Day
                day_idx = a
                month_idx = b
            elif max_b > 12: # Must be Day
                day_idx = b
                month_idx = a
            else:
                # Both <= 12. Ambiguous.
                # Default: Day comes before Month in Euro signals? Or Position Logic?
                # If Year is last (2), usually Day is first (0). (DD/MM/YYYY)
                # If Year is first (0), usually Month is first (1). (YYYY-MM-DD - ISO)
                if year_idx == 2:
                    # DD/MM/YYYY vs MM/DD/YYYY. Euro is safer default.
                    if a == 0: day_idx, month_idx = 0, 1
                    else: day_idx, month_idx = 1, 0
                else:
                    # YYYY-MM-DD vs YYYY-DD-MM. ISO (Start with Y) usually implies Month next.
                    # e.g. 2020-01-25. P1=1, P2=25. P2 is Day. 
                    # If both <= 12 (e.g. 2020-01-01), assume ISO standard (Year-Month-Day).
                    # So Month is likely the "middle" one (1).
                    if a == 1: month_idx, day_idx = 1, 2
                    else: month_idx, day_idx = 2, 1
        else:
            # No Year > 31? Maybe 2-digit year? Or data is garbage.
            return pd.to_datetime(series, errors='coerce'), ["Fallback: No identifiable Year"]

        # Reconstruct
        # Filter valid rows? No, just map index.
        df_parts = pd.DataFrame({
            'year': cols[year_idx],
            'month': cols[month_idx],
            'day': cols[day_idx]
        })
        
        parsed = pd.to_datetime(df_parts, errors='coerce')
        
        # Realign
        final = pd.Series(pd.NaT, index=series.index)
        final[parsed.index] = parsed
        
        return final, [f"Strict Structure: Y=P{year_idx}, M=P{month_idx}, D=P{day_idx}"]

    except Exception:
        return pd.to_datetime(series, errors='coerce'), ["Fallback: Exception in Struct Parser"]
