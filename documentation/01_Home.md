# Home — Dashboard & Data Loading

Data enters the application through **Data > Import** (a three-step flow: choose a file, check how it is read, name and load it) or through the sidebar dataset card (saved versions, sample cohort). **Explore > Statistics & correlations** then provides the overview of your data, column renaming, the correlation network and the performance benchmark.

---

## Data Loading

### Supported Formats

| Format | Extension |
| --- | --- |
| Delimited text (CSV, TSV) | `.csv`, `.tsv`, `.txt` |
| Excel | `.xlsx`, `.xlsm` and legacy `.xls` workbooks |
| Parquet | `.parquet` |

After loading, the dataset is displayed as an interactive preview with shape information (rows × columns) and data types per column.

### Delimiters

The import flow detects the delimiter of a text file. It finds the usual ones first: comma, semicolon, tab and pipe.
When none of them fits, it tries rarer delimiters used by some laboratory, HL7 or older exports: colon, caret, tilde,
hash, exclamation mark, broken bar and the ASCII unit separator. A delimiter is accepted when every line, the header
included, holds it the same number of times, outside quoted text.

To read the file with another delimiter, choose it under **Delimiter**:
- **space**, for files written with spaces between the columns;
- **other…**, then type any single character.

A delimiter cannot be a letter, a digit, a double quote, or the decimal or thousands separator; the page says so
and waits for another one. When only one column is read, the page suggests checking the delimiter. The options
chosen are recorded in the trace, and the replay reads the file the same way.

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
- New names are not checked for duplicates: give each column a unique name.
- **Apply Renaming** renames the columns as one step of the trace; it counts as a change to the data.

### AI-Assisted Renaming (RAG)
- The AI engine suggests clearer, standardised names. The suggestions come from the language model; they are not checked against a terminology (UMLS, SNOMED CT or LOINC are not loaded).
- Suggestions can be reviewed, accepted, or modified before applying.
- Requires the AI engine (Settings > AI engine). The default is a local Ollama model, which needs no API key; the cloud providers need an API key and your consent to send data.

---

## Performance Benchmark

Compare query performance between **Pandas** and **DuckDB** on your dataset.
- Runs four operations: numerical statistics, the top values of categorical columns, a grouped aggregation and the correlation matrix.
- Displays execution times side by side, with a check that both engines agree.
- The result is for information only: the application computes its statistics with pandas.

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
- Each saved version is a complete **Parquet** file (`<name>_v1.parquet`, `<name>_v2.parquet`, ...) in the `datasets` folder of your workspace (shown on Settings > Workspace); column types and row identifiers are kept.
- Each save creates a new version; previous versions remain accessible.
- Load any saved version from the sidebar dataset card (**Change**) or from the Start page.

### Operations

| Action | Description |
| --- | --- |
| **Save** | Persist the current dataset as a new version. |
| **Load** | Restore a previously saved version. |

Saved versions cannot be deleted or downloaded from the app. To download the rows of the analysis population, use **Filtered Cohort CSV** or **Filtered Cohort Excel** under a figure on Explore > Visualize (*Figure Export & Publication Options*).

---

## Practical Guidance
- Use **AI-Assisted Renaming** early in your workflow — standardised column names improve the quality of taxonomy generation and AI suggestions downstream.
