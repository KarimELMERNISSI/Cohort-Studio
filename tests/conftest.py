"""Suite-wide settings."""

import os

# Strict session state: a page that modifies the dataset views in place, or assigns the dataset keys
# directly, raises instead of being silently corrected (ui/state.py, follow-up programme M1).
os.environ.setdefault("CMP_STRICT_STATE", "1")

# A bootstrap administrator password, so that the app's start-up never issues a first-run setup token: whichever
# AppTest started the app first would otherwise see the first-run page instead of the login form. The first-run
# setup has its own tests (tests/test_first_admin.py, tests/ui_layer/test_first_admin_page.py).
os.environ.setdefault("CMP_ADMIN_PASSWORD", "test-bootstrap-password-1")
