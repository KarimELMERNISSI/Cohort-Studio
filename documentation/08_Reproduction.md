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

**What a trace holds about the data and the server.** A trace is meant to be shared with the people who check or
repeat the work, so it holds no value of the data:
- **Malformed lines** skipped at import are counted and located by their line numbers; their content is shown on
  the Import page only.
- **The source file** is identified by its SHA-256 checksum (`sha256:...`). Traces recorded by earlier versions
  hold an MD5 checksum, shown as such.
- **File paths** are written relative to your workspace (`traces/artifacts/.../source.csv`), never as the server's
  absolute paths, which name your account.
- **The software** of the session is recorded when the dataset is loaded: the CohortStudio version and commit,
  Python and the key libraries.

### Integrity

The page checks the hash chain of every trace and shows it next to its name: *integrity verified*, *integrity not
verifiable* (traces recorded before the chain existed), *modified after recording* (a step was changed, removed or
reordered), or *gap in the data chain* (a step starts from data that the previous step did not produce). The chain
makes a trace tamper-evident; it is not an electronic signature.

### Saved versions

Saving a version writes, next to the Parquet file, a small lineage file: the fingerprint of the data, the trace
it came from (session and head hash), the ticked criteria and the population. Loading the version links the new
trace to that parent, restores the ticked criteria, and warns when the file was modified outside CohortStudio.

### Viewing Traces
- Browse available traces from past sessions (**Select Analysis Session**; each entry shows its number of steps
  and its integrity).
- Select a specific trace to inspect its steps: description, function, time and saved snapshot. The parameters of
  each step are in the Word report.
- Open a trace file from elsewhere with **Upload Trace File (.json)**. The page has no download of the trace JSON:
  the trace files stay in the traces folder of your workspace, and the Word report documents them.

---

## Replay Engine

Replay a trace to reproduce the exact sequence of transformations.

### Replay Modes

| Mode | Description |
| --- | --- |
| **Fast (Use Snapshots)** | Loads the snapshot saved with the step. Works even if the original dataset is no longer available. |
| **Full Replay (From Source)** | Recomputes every step from the source file and the recorded parameters, and checks each step against the fingerprint recorded for it. |

A full replay says when the replayed table is exactly the one that was recorded, reports the first step where it
differs, and lists the steps it cannot recompute (the single imputation adopted from the former
multiple-imputation panel, data replaced outside the app); for those it uses the recorded snapshot when there is
one. A multiple imputation (`mi_impute`) is re-run from its recorded settings and checked against the
fingerprint of its imputed cells. A full replay also finds the cells filled by single imputations, so results
on the replayed data carry the same warning; a fast replay from snapshots does not know them. The replayed state becomes the session dataset, linked to
the trace it was replayed from.

### Path Resolution

Traces include file paths to the source file and to dataset snapshots. Paths are resolved **inside your
workspace** only:
- A recorded path is used when it points inside your workspace; a relative path (the form traces now use) is
  read from your workspace folder.
- Otherwise the file is looked up by name in your workspace's uploads, trace artifacts, snapshots and datasets
  folders, so a trace recorded on another system (for example Windows paths read in Docker) finds its files once
  they are in your workspace.
- A missing source or merged file is listed under **Missing Resources**: upload it, pick it from an artifact
  folder, or enter a path inside your workspace. Paths outside your workspace are refused.

### Step-by-Step Execution
- Each step of the list shows its description, function name, time and saved snapshot; **Reproduce to Step N**
  replays the trace up to that step.
- Steps are replayed sequentially with progress indication. In Fast mode, when the snapshot of the chosen step
  cannot be loaded, the trace is replayed from the source up to that step instead.
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
2. Click **Download Transformation Report (.docx)**.
3. A `.docx` file is generated and downloaded immediately.

---

## Study dossier

**Reproduce > Study dossier** documents the analyses of the current dataset in one Word document.

1. On the analysis pages, click **Add to the study dossier** under a result. Each set of settings is a separate
   item, kept with what it was computed on (data, revision, population, N analysed).
2. The Study dossier page draws the participant flow with a box for each analysis of the dossier (N analysed, or
   the imputed datasets pooled) and lists the results. A result computed on data that changed since, or on
   another population, is marked stale.
3. **Download the Word dossier**. Its sections:

| Section | Contents |
| --- | --- |
| 1. Provenance | Dataset, source file and hash, parse options, data fingerprint, trace and integrity, software versions |
| 2. Analysis population | Participant flow figure (STROBE item 13), exclusions with the criteria behind them, definition of every criterion |
| 3. Data handling | The recorded steps that changed the data, with their formulas |
| 4. Missing data | Missing values per variable, values filled by single imputation, the multiple imputation (settings, methods, status, methods paragraph) |
| 5. Results | Each result with its settings, what it was computed on, its tables and, for survival, the Kaplan-Meier figure |
| 6. References | The methods used, STROBE and, with multiple imputation, the pooling references |

Stale results are left out and listed; tick the option to keep them, marked as not current. The dossier supports
the reporting of the participant flow (STROBE item 13) and of multiple imputation (Sterne et al. 2009); it does
not certify compliance with a guideline, and it is not an electronically signed record.

---

## Practical Guidance

- **Save traces regularly** — they are your audit trail for data provenance.
- Use **Full Replay** to check that the recorded steps recompute the same data from the source file. When the
  source file is not available, use **Fast** replay: it loads the snapshot saved with each step.
- Generated `.docx` reports are useful as **supplementary material** in publications. They document the recorded
  steps; they are not electronically signed records.
