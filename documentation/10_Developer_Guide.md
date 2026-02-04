# Developer Guide: RAG & LLM Utilities

This guide covers the architecture and usage of the RAG system and LLM utilities.

---

## RAG Manager Architecture

The RAG system uses a **Mixin Pattern** for maintainability. The main `RAGManager` class inherits from three focused mixins:

```text
RAGManager (manage/rag_manager.py)
├── TaxonomyMixin      (manage/rag_taxonomy.py)
├── DocumentsMixin     (manage/rag_documents.py)
└── ComputedVarsMixin  (manage/rag_computed_vars.py)
```

### Key Benefits

- **Separation of Concerns**: Each mixin handles one domain
- **Maintainability**: Smaller, focused files (~300-500 lines each)
- **Backward Compatibility**: External code sees a single `RAGManager` class

---

## Authentication & User Isolation Architecture

**Location**: `manage/db_manager.py` (Backend), `main.py` (Frontend)

The application implements a secure, sidebar-based authentication system with strict data isolation.

### 1. Authentication Flow

- **Credential Storage**: Users are stored in a dedicated DuckDB table (`users.duckdb`).
- **Password Hashing**: Passwords are hashed using `bcrypt` (salt + hash) before storage.
- **Session**: `st.session_state` stores the authenticated `username`.

### 2. Data Isolation Model

The system uses a **Prefix & Directory** strategy to isolate user data.

| Component | Storage Strategy | Format/Path | Admin Access |
| :--- | :--- | :--- | :--- |
| **Datasets** | Filename Prefix | `data/{username}_{filename}.parquet` | ✅ Sees all files |
| **Traces** | Filename Prefix | `data/traces/{username}_session_{id}.json` | ✅ Sees all traces |
| **Taxonomies** | User Directory | `data/taxonomy/{username}/v{N}/` | ✅ Sees all user dirs |
| **Kn. Graphs** | Filename Prefix | `data/knowledge_graphs/{username}_{graph}.json` | ✅ Sees all graphs |

### 3. Security Implementation

- **Managers**: `DBManager`, `TransformationManager`, and frontend pages check `st.session_state.username`.
- **Fail-Closed**: If `username` is missing (e.g., session timeout), access methods return empty lists or errors, preventing data leakage.
- **Admin Override**: The user `admin` bypasses these filters to provide system oversight.

---

## Pydantic Response Schemas

**Location**: `manage/rag_schemas.py`

All LLM responses are validated using Pydantic models:

```python
from manage.rag_schemas import (
    TheoreticalConcept,     # Medical formula concept
    SuggestionResponse,     # Computed variable suggestions
    ProxyVariable,          # Proxy variable response
    AlternativeFormula,     # Alternative formula response
)
```

### Usage Example

```python
from manage.rag_schemas import ProxyVariable
from utils.llm_utils import validate_and_parse

response_text = '{"proxy_found": true, "proxy_name": "bmi"}'
result, error = validate_and_parse(response_text, ProxyVariable)

if result:
    print(result.proxy_name)  # Type-safe access
```

---

## LLM Utilities

**Location**: `utils/llm_utils.py`

### Core Functions

#### `parse_json_safe(text, default=None)`

Robust JSON parsing with multiple fallback strategies:

1. Extract from markdown code blocks
2. Find JSON by brace/bracket matching
3. Repair trailing commas
4. Balance brackets
5. Fall back to `ast.literal_eval`

```python
from utils.llm_utils import parse_json_safe

# Handles messy LLM output
text = '''Here's the result:
```json
{"name": "test", "value": 42,}
```

'''
result = parse_json_safe(text)  # {"name": "test", "value": 42}

```

#### `validate_and_parse(text, schema)`
Parse JSON and validate against a Pydantic schema:

```python
from utils.llm_utils import validate_and_parse
from manage.rag_schemas import SuggestionResponse

result, error = validate_and_parse(llm_output, SuggestionResponse)
if error:
    print(f"Validation failed: {error}")
else:
    for s in result.suggestions:
        print(s.name, s.formula)
```

#### `StructuredOutputHelper`

Helper class for schema-validated LLM calls with retry:

```python
from utils.llm_utils import StructuredOutputHelper
from manage.rag_schemas import ProxyVariable

helper = StructuredOutputHelper(llm, max_retries=2)
result, error = helper.invoke_with_schema(prompt, ProxyVariable)
```

---

## Data Analysis Utilities

### DataAnalyzer

**Location**: `utils/data_analyzer.py`

Analyzes DataFrame columns and categorizes them by type:

```python
from utils.data_analyzer import DataAnalyzer

analyzer = DataAnalyzer(df)

# Access categorized columns
analyzer.numeric_cols           # Numeric columns (high cardinality)
analyzer.categorical_cols       # Categorical + low-cardinality numeric
analyzer.date_cols              # Date/datetime columns
analyzer.binary_cols            # Columns with exactly 2 unique values
analyzer.low_cardinality_numeric_cols  # Numeric with ≤10 unique values
analyzer.high_cardinality_cat_cols     # Categorical with many values

# Refresh after DataFrame changes
analyzer.refresh(new_df)

# Get suitable columns for plot types
cols = analyzer.get_suitable_columns("Box Plot")
# {"y": [...], "x": [...]}
```

---

## Statistics Utilities

**Location**: `utils/statistics_utils.py`

### Normality Testing

```python
from utils.statistics_utils import normality_test

p_value = normality_test(column, method='shapiro')
# Methods: 'shapiro', 'dagostino', 'ks', 'anderson'
```

### Test Guidelines UI

```python
from utils.statistics_utils import show_test_guidelines

show_test_guidelines()  # Renders Streamlit help UI
```

---

## Export Utilities

**Location**: `utils/export_utils.py`

```python
from utils.export_utils import to_excel, to_excel_sheets

# Single DataFrame
excel_bytes = to_excel(df)

# Multiple DataFrames to sheets
excel_bytes = to_excel_sheets({
    "Sheet1": df1,
    "Sheet2": df2
})
```

---

## Best Practices

### 1. Always Use Schema Validation for LLM Outputs

```python
# ❌ Fragile
result = json.loads(llm_response)

# ✅ Robust
result, error = validate_and_parse(llm_response, MySchema)
```

### 2. Use parse_json_safe for Unstructured JSON

```python
# Handles all edge cases
data = parse_json_safe(text, default={})
```

### 3. Add Logging for Debugging

```python
import logging
logger = logging.getLogger(__name__)

logger.debug(f"Parsed {len(items)} items")
logger.warning(f"Failed to parse: {error}")
```
