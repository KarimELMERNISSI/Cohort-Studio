# 🏠 Home — Dashboard & Data Loading

The **Main View (Home)** page is the entry point for working with your dataset. It handles data loading, provides an overview of your data, and includes tools for column renaming, performance benchmarking, and dataset management.

---

## 📂 Data Loading

### Supported Formats

| Format | Extension |
|---|---|
| CSV | `.csv` |
| Excel | `.xlsx` |
| Parquet | `.parquet` |

After loading, the dataset is displayed as an interactive preview with shape information (rows × columns) and data types per column.

---

## 📊 Dataset Statistics

An automatic summary of the loaded dataset:

| Metric | Description |
|---|---|
| **Shape** | Number of rows and columns. |
| **Data Types** | Breakdown of numerical, categorical, binary, and date columns. |
| **Missing Values** | Count and percentage per column. |
| **Descriptive Statistics** | Mean, median, std, min, max for numerical columns. |
| **Cardinality** | Unique value counts for categorical columns. |

The statistics are cached for performance (`@st.cache_data`).

---

## ✏️ Column Renaming

Rename columns for clarity and consistency. Two approaches are available:

### Manual Renaming

- Edit column names directly in a mapping table.
- Duplicate names are automatically detected and prevented.

### AI-Assisted Renaming (RAG)

- The RAG system suggests standardised names based on medical terminologies (UMLS, SNOMED, LOINC).
- Suggestions can be reviewed, accepted, or modified before applying.
- Requires the RAG system to be initialised (Gemini API key configured).

---

## ⚡ Performance Benchmark

Compare query performance between **Pandas** and **DuckDB** on your dataset.

- Runs a standard set of operations (filtering, grouping, aggregation).
- Displays execution times side by side.
- Helps decide whether to use DuckDB for large datasets.

---

## 🔐 Authentication

Authentication is handled in the sidebar:

| Feature | Description |
|---|---|
| **Login** | Username and password authentication with bcrypt-hashed storage. |
| **Sign Up** | New accounts are created with a "pending" status. |
| **Account Activation** | New accounts require administrator approval before login is permitted. |
| **Logout** | Clears session state and returns to the login form. |

### Data Isolation

Each user's data is isolated through filename prefixes and user-specific directories. Datasets, traces, and analysis results are not shared across accounts.

---

## 💾 Dataset Management

### Versioning

- Datasets are versioned using **DuckDB** and **Parquet snapshots**.
- Each save creates a new version; previous versions remain accessible.
- Load any saved version from the sidebar.

### Operations

| Action | Description |
|---|---|
| **Save** | Persist the current dataset as a new version. |
| **Load** | Restore a previously saved version. |
| **Delete** | Remove a saved version (with confirmation). |
| **Export** | Download the current dataset as CSV, Excel, or Parquet. |

---

## 💡 Tips

- Use **AI-Assisted Renaming** early in your workflow — standardised column names improve the quality of taxonomy generation and AI suggestions downstream.
- Run the **Performance Benchmark** on large datasets (>100k rows) to determine if DuckDB offers meaningful speedups for your data.
