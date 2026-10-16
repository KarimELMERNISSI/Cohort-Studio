# 12 // Identity & Access Management (IAM) Architecture & Better Auth Evaluation

## 1. Executive Summary

When several researchers share one CohortStudio server, the application must know who is signed in: accounts separate the researchers' workspaces, and administrators approve new accounts. This is the role of Identity and Access Management (IAM).

This document evaluates modern authentication frameworks—specifically **Better Auth**—and describes the modular, provider-agnostic IAM architecture implemented within **CohortStudio**.

---

## 2. Framework Evaluation: Better Auth in a Python Clinical Stack

### 2.1 What is Better Auth?
[Better Auth](https://better-auth.com/) is a TypeScript-first, framework-agnostic authentication engine. It provides:
- Session management with cryptographically signed tokens.
- Pluggable social logins, OAuth 2.0 / OpenID Connect (OIDC).
- Two-Factor Authentication (2FA / TOTP) and Passkeys / WebAuthn.
- Multi-tenant organization hierarchies with granular Role-Based Access Control (RBAC).
- Extensible plugin ecosystem (rate limiting, email verification, audit logs).

### 2.2 Feasibility & Runtime Constraints
Better Auth is authored for JavaScript/TypeScript runtimes (Node.js, Bun, Deno). Because CohortStudio is a Python 3.14+ platform built on Streamlit and DuckDB, direct in-process library execution is not possible.

However, Better Auth is designed to expose a clean RESTful HTTP API (`/api/auth/*`), making it suitable for architectural integration via a **Sidecar Microservice Pattern**.

```
+-------------------------------------------------------------+
|                  Hospital Network / Cloud                   |
|                                                             |
|   +-----------------------+       +---------------------+   |
|   |  CohortStudio app     | ----> | Better Auth Sidecar |   |
|   |  (Python 3.14)        | REST  | (Node.js / Bun)     |   |
|   |  - BetterAuthIAM      |       | - OAuth / OIDC      |   |
|   |  - In-Memory Cache    |       | - 2FA / WebAuthn    |   |
|   +-----------------------+       +---------------------+   |
|               |                              |              |
|               v                              v              |
|        [DuckDB Storage]               [PostgreSQL DB]       |
+-------------------------------------------------------------+
```

---

## 3. Performance Impact & Latency Mitigation

### 3.1 The Streamlit Performance Challenge
Streamlit re-executes the Python script from top to bottom on every user interaction (widget selection, slider drag, tab switch). 

If session verification required a synchronous HTTP round-trip on every script run:
- Each rerun would wait for the network: roughly tens of milliseconds on a local network, more for a cloud-hosted service (estimates, not measured).
- **Impact**: the interface would respond more slowly during cohort exploration.

### 3.2 CohortStudio's Solution: In-Memory TTL Session Caching
To avoid repeated network calls, `BetterAuthIAMProvider` implements an **in-memory Time-to-Live (TTL) cache**:
1. When a user logs in or a session token is validated, the successful result (with its `UserProfile`) is cached in memory with an expiration timestamp (`CMP_IAM_CACHE_TTL`, default 300 seconds).
2. A repeated check of the same credentials or token within the TTL is answered from the cache, without a network call. The login page itself keeps the signed-in user in the Streamlit session state and does not re-check the session on every rerun.
3. HTTP connection reuse is preserved via persistent keep-alive pooling (`requests.Session()`).
4. If the remote auth server is briefly unavailable, results already in the cache are still returned until they expire.

### 3.3 Comparison of the Modes

No latency benchmark has been run; the table compares what each mode depends on.

| Authentication Mode | Network Dependency | Isolated Network (No Internet) |
| :--- | :---: | :---: |
| **Local DuckDB + bcrypt** (Default) | None | Yes |
| **Better Auth (Uncached)** | A call to the sidecar per check | Only if the sidecar runs on the same network |
| **Better Auth (CohortStudio TTL Cached)** | A call to the sidecar per check, then cached for `CMP_IAM_CACHE_TTL` seconds | Only if the sidecar runs on the same network |
| **Disabled IAM** (`disabled`) | None | Yes |

---

## 4. Modular Architecture Design

The IAM subsystem is located in `manage/iam/` and adheres to strict separation of concerns:

```
manage/iam/
|-- __init__.py           # Unified exports (models, providers, manager)
|-- authz.py              # Role checks and credential rules (admin role, username pattern, password length)
|-- base.py               # BaseIAMProvider abstract interface
|-- manager.py            # IAMManager facade & global factory, administrator bootstrap
|-- schemas.py            # Pydantic schemas (UserProfile, UserRole, AuthResult)
|-- setup_token.py        # One-time setup token for the first administrator (only its hash is stored)
`-- providers/
    |-- __init__.py
    |-- local_duckdb.py   # Local embedded DuckDB + bcrypt (Default)
    |-- better_auth.py    # Remote Better Auth REST adapter with TTL cache
    `-- disabled.py       # Single-user bypass for automated pipelines
```

### 4.1 Base Provider Contract (`BaseIAMProvider`)
Every provider implements the following contract:
- `authenticate(credentials: UserCredentials) -> AuthResult`
- `create_user(username, password, role, auto_activate) -> tuple[bool, str]`
- `verify_session(token: str) -> AuthResult`
- `get_user(username: str) -> UserProfile | None`
- `list_users() -> list[UserProfile]`
- `activate_user(username: str) -> tuple[bool, str]`
- `deactivate_user(username: str) -> tuple[bool, str]`
- `delete_user(username: str) -> tuple[bool, str]`
- `update_password(username: str, new_password: str) -> tuple[bool, str]`
- `is_enabled -> bool`

### 4.2 Supported Providers Out-of-the-Box

1. **`LocalDuckDBIAMProvider` (Default)**:
   - Zero external infrastructure required.
   - Stores users in `users.duckdb` with bcrypt salt rounds = 12.
   - Mandatory admin account approval for newly registered researchers.
   - Ideal for air-gapped clinical trial workstations.

2. **`BetterAuthIAMProvider`**:
   - Delegates authentication to an external or sidecar Better Auth instance.
   - Signs users in with a username (or e-mail) and password, checks session tokens, and calls `/admin/users/...` endpoints for account administration, which the sidecar must provide.
   - Single sign-on (OAuth, OIDC, SAML, LDAP or Active Directory) is not implemented in CohortStudio; a Better Auth sidecar could add it in future, with changes to this adapter and to the login page.
   - In-memory TTL caching avoids a network call for checks repeated within the TTL.

3. **`DisabledIAMProvider`**:
   - Bypasses authentication entirely (`is_enabled == False`).
   - Grants pre-authorized `local_researcher` administrator access.
   - Ideal for headless pipelines, automated testing, or single-user laptops.

---

## 5. Configuration & Provider Switching

CohortStudio dynamically resolves the active provider through environment variables:

### 5.1 Mode 1: Local DuckDB (Default)
```bash
# No configuration needed, defaults to local
export CMP_IAM_PROVIDER=local
uv run streamlit run main.py
```

### 5.2 Mode 2: Better Auth Remote Sidecar
```bash
export CMP_IAM_PROVIDER=better_auth
export CMP_BETTER_AUTH_URL=http://localhost:3000/api/auth
export CMP_BETTER_AUTH_API_KEY=your_secret_admin_token
export CMP_IAM_CACHE_TTL=300
uv run streamlit run main.py
```

### 5.3 Mode 3: Authentication Disabled (Single-User Research Mode)
```bash
export CMP_IAM_PROVIDER=disabled
uv run streamlit run main.py
```

---

## 6. Docker Compose Deployment Recipe (Better Auth Sidecar)

This recipe runs CohortStudio with a Better Auth sidecar. The `better-auth` service needs a Node.js application that
configures Better Auth and provides the endpoints the adapter calls (`/sign-in/username`, `/sign-in/email`,
`/sign-up/email`, `/get-session`, `/admin/users/...`). That application is not part of this repository: the
`node:20-alpine` service below has nothing to run until it is mounted into `/app` or built into an image. Replace
the placeholder secrets and passwords.

```yaml
version: '3.8'

services:
  # 1. Better Auth Microservice
  better-auth:
    image: node:20-alpine
    working_dir: /app
    command: sh -c "npm install && npm start"
    environment:
      - PORT=3000
      - BETTER_AUTH_SECRET=super_secret_encryption_key_32chars
      - DATABASE_URL=postgres://auth_user:auth_pass@postgres:5432/better_auth_db
    ports:
      - "3000:3000"
    depends_on:
      - postgres

  # 2. Auth Database
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: auth_user
      POSTGRES_PASSWORD: auth_pass
      POSTGRES_DB: better_auth_db
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # 3. CohortStudio
  cohort-app:
    build: .
    environment:
      - CMP_IAM_PROVIDER=better_auth
      - CMP_BETTER_AUTH_URL=http://better-auth:3000/api/auth
      - CMP_IAM_CACHE_TTL=300
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    depends_on:
      - better-auth

volumes:
  postgres_data:
```

---

## 7. Conclusion

Identity and Access Management sits behind one provider interface, so CohortStudio can:
- **Choose the provider by configuration**: local DuckDB accounts (no network needed), a Better Auth sidecar (whose application is not part of this repository), or no sign-in for single-user workstations and automated pipelines.
- **Limit network calls**: with Better Auth, the TTL cache answers repeated checks within the TTL without a network call; no latency has been measured.
- **Keep the workspaces**: the per-user workspaces and traces are keyed by username, whatever the provider.
