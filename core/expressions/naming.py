"""Safe substitution for user-supplied naming patterns such as ``"{initial_variable}_{method_applied}"``.

``str.format`` on untrusted patterns allows attribute and index access (``{x.__class__}``) and format-spec
tricks; this helper only substitutes bare ``{name}`` fields from an explicit mapping.
"""

from __future__ import annotations

import re

_FIELD = re.compile(r"\{(\w+)\}")
_BRACES = re.compile(r"[{}]")


def safe_format(pattern: str, **fields: object) -> str:
    """Substitute ``{name}`` placeholders; any other brace usage or unknown field is an error."""
    if not isinstance(pattern, str):
        raise ValueError("The naming pattern must be a string.")
    leftover = _FIELD.sub("", pattern)
    if _BRACES.search(leftover):
        raise ValueError("Naming patterns may only contain simple {name} placeholders.")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in fields:
            raise ValueError(
                f"Unknown placeholder '{{{name}}}'. Available: {', '.join(sorted(fields)) or 'none'}."
            )
        return str(fields[name])

    return _FIELD.sub(replace, pattern)


__all__ = ["safe_format"]
