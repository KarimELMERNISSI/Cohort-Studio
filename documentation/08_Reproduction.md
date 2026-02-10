# 🔁 Reproduce Analysis

The **Reproduce Analysis** page enables full reproducibility of data transformation sessions. Every analysis step is automatically recorded in a trace file, which can be replayed to recreate results or generate documentation reports.

---

## 📋 Trace Management

### What Is a Trace?

A **trace** is a JSON file that captures every transformation applied during a session:

| Field | Description |
|---|---|
| **Session ID** | Unique identifier for the session. |
| **Timestamp** | When the session started. |
| **Username** | The user who ran the session. |
| **Dataset Name** | The initial dataset loaded. |
| **Steps** | Ordered list of transformations with parameters, descriptions, and snapshot paths. |

### Viewing Traces

- Browse available traces from past sessions.
- Select a specific trace to inspect its steps, parameters, and outputs.
- Download the raw JSON trace file for external archiving.

---

## ▶️ Replay Engine

Replay a trace to reproduce the exact sequence of transformations.

### Replay Modes

| Mode | Description |
|---|---|
| **Fast Replay** | Re-applies transformations using parameters only. Requires the original dataset to be available. Quick and lightweight. |
| **Full Replay** | Uses embedded dataset snapshots from each step. Works even if the original dataset has been modified or deleted. |

### Path Resolution

Traces include file paths to dataset snapshots. The replay engine handles **cross-platform portability**:

- Relative paths are resolved against the project root.
- Absolute paths are adapted to the current system.
- Missing files are reported with clear error messages.

### Step-by-Step Execution

- Steps are replayed sequentially with progress indication.
- Each step shows: function name, parameters, description, and result.
- On error, the replay pauses and reports which step failed and why.

---

## 📄 Automated Reporting

Generate a `.docx` report from any trace file using the **TraceDocumenter**.

### Report Contents

| Section | Description |
|---|---|
| **Header** | Session metadata (ID, user, dataset, date). |
| **Steps Table** | Each transformation step with its function, parameters, and description. |
| **Summary** | Total number of steps and overall session description. |

### How to Generate

1. Select a trace from the history or upload a trace JSON file.
2. Click **Download Report**.
3. A `.docx` file is generated and downloaded immediately.

---

## 💡 Tips

- **Save traces regularly** — they are your audit trail for data provenance.
- Use **Full Replay** when sharing analyses with collaborators who may not have the original dataset.
- Generated `.docx` reports are useful for **supplementary materials** in publications or **audit documentation** for regulatory submissions.
- The trace system also appears in the **Data Enrichment** sidebar, allowing quick access to replay and reporting without navigating away.
