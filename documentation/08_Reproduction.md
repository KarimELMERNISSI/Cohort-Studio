# Reproduce Analysis

The **Versions & traces** page replays and documents data transformation sessions. Every change of the data is
recorded in a trace file, with what it did to the table; the trace can be replayed to recreate the data, checked
step by step against what was recorded, and documented in a Word report.

---

## Trace Management

### What Is a Trace?

A **trace** is a JSON file that captures every transformation applied during a session:

| Field | Description |
| --- | --- |
| **Session ID** | Unique identifier for the session. |
| **Timestamp** | When the session started. |
| **Username** | The user who ran the session. |
| **Dataset Name** | The initial dataset loaded. |
| **Steps** | Ordered list of transformations with parameters, descriptions, and snapshot paths. |
| **Data** (each step) | Rows and columns before and after, columns added, removed or retyped, and the fingerprints of the table before and after. |
| **Population** (some steps) | The ticked criteria and the counts of the participant flow, for imputations, saved versions and results. |
| **Hash chain** | Each step carries the hash of the previous one: editing, removing or reordering a step afterwards is detected. |

### Integrity

The page checks the hash chain of every trace and shows it next to its name: *integrity verified*, *integrity not
verifiable* (traces recorded before the chain existed), or *modified after recording*. The chain makes a trace
tamper-evident; it is not an electronic signature.

### Saved versions

Saving a version writes, next to the Parquet file, a small lineage file: the fingerprint of the data, the trace
it came from (session and head hash), the ticked criteria and the population. Loading the version links the new
trace to that parent, restores the ticked criteria, and warns when the file was modified outside CohortStudio.

### Viewing Traces
- Browse available traces from past sessions.
- Select a specific trace to inspect its steps, parameters, and outputs.
- Download the raw JSON trace file for external archiving.

---

## Replay Engine

Replay a trace to reproduce the exact sequence of transformations.

### Replay Modes

| Mode | Description |
| --- | --- |
| **Fast (snapshots)** | Loads the snapshot saved with the step. Works even if the original dataset is no longer available. |
| **Full replay** | Recomputes every step from the source file and the recorded parameters, and checks each step against the fingerprint recorded for it. |

A full replay says when the replayed table is exactly the one that was recorded, reports the first step where it
differs, and lists the steps it cannot recompute (the single imputation adopted from the former
multiple-imputation panel, data replaced outside the app); for those it uses the recorded snapshot when there is
one. A multiple imputation (`mi_impute`) is re-run from its recorded settings and checked against the
fingerprint of its imputed cells. A full replay also finds the cells filled by single imputations, so results
on the replayed data carry the same warning; a fast replay from snapshots does not know them. The replayed state becomes the session dataset, linked to
the trace it was replayed from.

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

## Automated Reporting

Generate a `.docx` report from any trace file using the **TraceDocumenter**.

### Report Contents

| Section | Description |
| --- | --- |
| **Header** | Session metadata (ID, user, dataset, date). |
| **Integrity** | Hash chain verified (with the head hash), not verifiable, or failed; the parent version or trace. |
| **Steps** | Each step with its parameters, what it did to the table (rows, columns, fingerprints) and, for population steps, the participant flow. |
| **Footer** | The report documents the recorded trace; it is not an electronically signed record. |

### How to Generate

1. Select a trace from the history or upload a trace JSON file.
2. Click **Download Report**.
3. A `.docx` file is generated and downloaded immediately.

---

## Practical Guidance

- **Save traces regularly** — they are your audit trail for data provenance.
- Use **Full Replay** when sharing analyses with collaborators who may not have the original dataset.
- Generated `.docx` reports are useful as **supplementary material** in publications. They document the recorded
  steps; they are not electronically signed records.
