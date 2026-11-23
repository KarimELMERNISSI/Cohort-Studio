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
- **Download the trace (JSON)** of the selected session, to keep it with the study. It holds what the steps recorded, row numbers and values of your data included: treat it as the data are treated.
  The file is the trace as recorded, so its integrity check still passes there.
- Open a trace file from elsewhere with **Upload Trace File (.json)**. Its integrity is checked when it is opened.

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
one, and says so: a replay that took a step from its recorded table is not called reproduced. A multiple imputation (`mi_impute`) is re-run from its recorded settings and checked against the
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
- A missing source or merged file is listed under **Missing Resources**, once even when two merges read it (the
  entry names both steps): upload it, pick it from an artifact folder, or enter a path inside your workspace. Paths
  outside your workspace are refused. A saved version your workspace holds is found by its name, and the source by
  the path it was loaded from when its recorded copy is gone.
- A file that is not a trace (its steps are not a list of recorded steps) is refused with the reason, before anything
  reads it.

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

**The study definition.** A strip at the top of the page names the study your dataset belongs to (its name: every
version saved under that name belongs to it, and a file imported or saved under another name starts another study)
and says whether its definition is saved as a revision: *no revision saved*, *changed since revision n* (and what
changed: the criteria, which of them are ticked, the follow-up, the derived variables), *labels changed since
revision n* (a description or a note only), or *revision n, saved* on a date. The definition is what the study
decides apart from the data: the criteria with their descriptions, which of them are ticked, the follow-up and the
declared derived variables with their notes. It holds no data and no result.

- **Save revision n** keeps the definition as it is now, as written, in your workspace: check that no description or
  note names a person. A revision is never changed afterwards. A changed definition, a corrected description
  included, is saved as the next revision; an unchanged one saves nothing.
- **Delete revision n** removes a revision from your workspace, with the descriptions and notes it holds, after a
  confirmation; its number is never given again, and the session's definition is unchanged. A description the trace
  recorded for a ticked criterion stays in the trace.
- **Load revision n** is offered when the session holds no criteria and the latest revision does (the next day, a
  saved version loaded again): the revision's criteria, which of them are ticked, the follow-up and the derived
  variables become the session's. Criteria are filters: no row changes.

The trace records each revision by its number and fingerprint, never its contents; the trace report and the Word
dossier name it by its number. The data have their own counter, the *data revision*, shown with each result.

1. On the analysis pages, click **Add to the study dossier** under a result. Each set of settings is a separate
   item, kept with what it was computed on (data, data revision, population, N analysed).
2. The Study dossier page draws the participant flow with a box for each analysis of the dossier (N analysed, or
   the imputed datasets pooled) and lists the results. A result computed on data that changed since, or on
   another population, is marked stale. The calculations from numbers typed (a calculator's result, added with
   the same button) are listed apart, under *From numbers typed by hand*: they read no data, so they are never
   stale and the flow does not count them. Like the results, they are cleared when another dataset is loaded.
3. **Download the Word dossier**. Its sections:

| Section | Contents |
| --- | --- |
| 1. Provenance | Dataset, source file and hash, parse options, data fingerprint, the study definition and its revision, trace and integrity, software versions (with CohortStudio's commit and a fingerprint of its code) |
| 2. Analysis population | Participant flow figure (STROBE item 13), exclusions with the criteria behind them, definition of every criterion |
| 3. Data handling | The recorded steps that changed the data, with their formulas |
| 4. Missing data | Missing values per variable, values filled by single imputation, the multiple imputation (settings, methods, status, methods paragraph) |
| 5. Results | Each result with its settings, what it was computed on, its tables and, for survival, the Kaplan-Meier figure |
| 6. Calculations from numbers typed by hand | Each calculation with the numbers typed, where they come from when it was said, and its result, computed again from its numbers; never a population, a number analysed or a fingerprint |
| 7. References | The methods used, the calculations' included, STROBE and, with multiple imputation, the pooling references |

Stale results are left out and listed; tick the option to keep them, marked as not current. The dossier supports
the reporting of the participant flow (STROBE item 13) and of multiple imputation (Sterne et al. 2009); it does
not certify compliance with a guideline, and it is not an electronically signed record.

---

## Checking a package you received

A study package is a `.cstudio.zip` file that someone sent you: a study's definition and its results and, in a
package *with the data*, the data themselves, encrypted. Check it before you type its password. The check runs on
a computer where CohortStudio is installed, in a terminal, not in the browser.

1. **Install CohortStudio** on your computer, as its README says (section
   [Installation & Quick Start](https://github.com/KarimELMERNISSI/Cohort-Studio#installation--quick-start)):
   install `uv`, take a copy of CohortStudio's code, and run `uv sync --frozen` in its folder. The latest version
   reads the packages of the earlier ones.
2. **Open a terminal in that folder** (Windows 11: right-click inside the folder, then *Open in Terminal*). Each
   command below runs there, with `uv run` in front (the package's own notice writes the command without it: that
   form works once CohortStudio's environment is activated). Give the package's whole path, in quotes: dragging
   the file onto the terminal's window writes it.
3. **Compare the package hash** with the one the sender gave you apart from the file (by another way than the one
   the file came by; read aloud, its first 16 characters are enough):

   ```
   uv run python -m manage.replay inspect "C:\Users\you\Downloads\study_r3_with_data.cstudio.zip" --package-hash 3f2a-9c1e-4b7d-0a6c
   ```

   It prints the kind of package, its hash and its files, and says whether the hash is the one you gave. If it is
   not, it stops there and shows where the two part: check first what you typed against what the sender sent you
   (the package hash, not a checksum of the zip file). If the hash you gave is the sender's, this is not the
   package they made, or it was changed on the way: do not open it, and ask the sender for it again. Nothing is
   decrypted at this step.
4. **Read it with the password**, which the sender gave you apart from the file too. Add `--ask-password` to type
   it unseen (each key shows as `*`; three tries), or `--password-file FILE` to read it from the first line of a
   file that only you can read, kept away from the package (delete that file afterwards). Never type the password
   as part of the command: a terminal keeps its command lines. Each file is then checked against the package's list
   of checksums, and the command says what the package is: the version that made it, the study and its revision,
   to whom and under what it was sent, the rows and columns of each file of data, and the number of steps of its
   trace. If you are not the recipient it names, do not use the data: delete the package and tell its sender.
5. **When your work with it ends**, delete the package, the password's file and every file you took out of the
   package: a package with the data holds personal data.

`inspect` shows nothing of the study's own texts or of the data, replays nothing and writes nothing to disk.
`-h` lists its options, and `--json` gives the same facts to a program. Its exit code is 0 when the package was
read and nothing failed, 1 when it was read but is not the package its sender made (a file does not match its
checksum, or the hash is not the one given), 2 when it was not read (not a package, one made by a later version,
the wrong password), 64 when the command was not understood or cannot run where it is (a password asked for in a
window that cannot hide it), and 70 for an error of the program itself.

---

## Practical Guidance

- **Save traces regularly** — they are your audit trail for data provenance.
- Use **Full Replay** to check that the recorded steps recompute the same data from the source file. When the
  source file is not available, use **Fast** replay: it loads the snapshot saved with each step.
- Generated `.docx` reports are useful as **supplementary material** in publications. They document the recorded
  steps; they are not electronically signed records.
