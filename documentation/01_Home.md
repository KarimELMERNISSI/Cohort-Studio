# Home — Dashboard & Data Loading

Data enters the application through **Data > Import** (a three-step flow: choose a file, check how it is read, name and load it) or through the sidebar dataset card (saved versions, sample cohort). **Explore > Statistics & correlations** then provides the overview of your data, column renaming, the correlation network and the performance benchmark.

---

## Data Loading

### Supported Formats

| Format | Extension |
| --- | --- |
| CSV | `.csv` |
| Excel | `.xlsx` |
| Parquet | `.parquet` |

After loading, the dataset is displayed as an interactive preview with shape information (rows × columns) and data types per column.

---

## Dataset Statistics

An automatic summary of the loaded dataset:

| Metric | Description |
| --- | --- |
| **Shape** | Number of rows and columns. |
| **Data Types** | Breakdown of numerical, categorical, binary, and date columns. |
| **Missing Values** | Count and percentage per column. |
| **Descriptive Statistics** | Mean, median, std, min, max for numerical columns. |
| **Cardinality** | Unique value counts for categorical columns. |

The statistics are cached for performance (`@st.cache_data`).

---

## Column Renaming

Rename columns for clarity and consistency. Two approaches are available:

### Manual Renaming
- Edit column names directly in a mapping table.
- Duplicate names are automatically detected and prevented.

### AI-Assisted Renaming (RAG)
- The RAG system suggests standardised names based on medical terminologies (UMLS, SNOMED, LOINC).
- Suggestions can be reviewed, accepted, or modified before applying.
- Requires the RAG system to be initialised (Gemini API key configured).

---

## Performance Benchmark

Compare query performance between **Pandas** and **DuckDB** on your dataset.
- Runs a standard set of operations (filtering, grouping, aggregation).
- Displays execution times side by side.
- Helps decide whether to use DuckDB for large datasets.

---

## Authentication

Authentication is a centred sign-in screen (the application opens only after sign-in):

| Feature | Description |
| --- | --- |
| **Login** | Username and password authentication with bcrypt-hashed storage. |
| **Sign Up** | New accounts are created with a "pending" status. |
| **Account Activation** | New accounts require administrator approval before login is permitted. |
| **Logout** | Clears session state and returns to the login form. |

### Data Isolation

Each account works in its own workspace folder (`<DATA_ROOT>/users/<username>/`). Datasets, traces, uploads, taxonomies, knowledge graphs, the document library and the vector store are not shared across accounts, and file paths typed into the app must point inside your workspace.

---

## Dataset Management

### Versioning
- Datasets are versioned using **DuckDB** and **Parquet snapshots**.
- Each save creates a new version; previous versions remain accessible.
- Load any saved version from the sidebar dataset card (**Change**) or from the Start page.

### Operations

| Action | Description |
| --- | --- |
| **Save** | Persist the current dataset as a new version. |
| **Load** | Restore a previously saved version. |
| **Delete** | Remove a saved version (with confirmation). |
| **Export** | Download the current dataset as CSV, Excel, or Parquet. |

---

## Practical Guidance
- Use **AI-Assisted Renaming** early in your workflow — standardised column names improve the quality of taxonomy generation and AI suggestions downstream.
- Run the **Performance Benchmark** on large datasets (>100k rows) to determine if DuckDB offers meaningful speedups for your data.
