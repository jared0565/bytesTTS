# SPDX-License-Identifier: MPL-2.0
"""Licence gate: every installed distribution must carry an allowlisted licence.

Allowed: MIT, BSD-2/3, Apache-2.0, MPL-2.0, ISC, PSF, Unlicense, CC0. LGPL passes as "lgpl"
(allowed only unmodified and dynamically imported, and listed in NOTICE). GPL is never allowed.
The only exception is NVIDIA_EXCEPTIONS: reviewed nvidia-* CUDA runtime wheels under [train].

    python -P tools/dev/check_licences.py      # exit 0 only if nothing is denied or unknown
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
from collections.abc import Iterable
from typing import Any, Literal

Verdict = Literal["allowed", "lgpl", "denied", "unknown"]

# Reviewed and filled in when [train] with CUDA is first locked (plan, Task 2). Empty until then.
NVIDIA_EXCEPTIONS: frozenset[str] = frozenset()

# Distributions whose EXACT licence expression was reviewed and accepted although it names a licence
# outside the allowlist. A changed expression fails again until it is re-reviewed. (Ledger, Task 2.)
REVIEWED: dict[str, str] = {
    # Core dependency. 0BSD has no conditions at all; Zlib's conditions (no misrepresentation, mark
    # altered sources, keep the notice) are no stronger than BSD's. Both cover bundled code.
    "numpy": "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0",
    # [train], via soundfile. MIT-0 is MIT without even the attribution condition.
    "cffi": "MIT-0",
    # [train]. BSL-1.0 (Boost) is permissive, with conditions no stronger than MIT's; Zlib as above.
    "torch": "Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC "
    "AND MIT AND Zlib",
}

_ALLOWED = frozenset(
    {
        "MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "MPL-2.0", "ISC",
        "PSF-2.0", "Python-2.0", "Unlicense", "CC0-1.0",
    }
)  # fmt: skip
_LGPL = re.compile(r"LGPL-(2\.0|2\.1|3\.0)(-only|-or-later|\+)?")
_GPL = re.compile(r"A?GPL-(1\.0|2\.0|3\.0)(-only|-or-later|\+)?")
_RANK: dict[Verdict, int] = {"denied": 0, "unknown": 1, "lgpl": 2, "allowed": 3}

# Trove classifiers -> SPDX. "BSD License" does not say 2- or 3-clause; both are allowed.
_CLASSIFIERS = {
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: MIT No Attribution License (MIT-0)": "MIT-0",
    "License :: OSI Approved :: BSD License": "BSD-3-Clause",
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: OSI Approved :: The Unlicense (Unlicense)": "Unlicense",
    "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication": "CC0-1.0",
    "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)": "LGPL-2.0-or-later",
    "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)": "LGPL-2.0-only",
    "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.0-or-later",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0-only",
    "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)": "LGPL-3.0-or-later",
    "License :: OSI Approved :: GNU General Public License (GPL)": "GPL-2.0-or-later",
    "License :: OSI Approved :: GNU General Public License v2 (GPLv2)": "GPL-2.0-only",
    "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)": "GPL-2.0-or-later",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)": "GPL-3.0-or-later",
    "License :: OSI Approved :: GNU Affero General Public License v3": "AGPL-3.0-only",
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)": "AGPL-3.0-or-later",
}

# Short free-text `License` fields seen in the wild -> SPDX. Each entry is a reviewed mapping.
_SHORT_NAMES = {
    "mit": "MIT",
    "mit license": "MIT",
    "bsd": "BSD-3-Clause",
    "bsd license": "BSD-3-Clause",
    "new bsd": "BSD-3-Clause",
    "new bsd license": "BSD-3-Clause",
    "bsd 3-clause": "BSD-3-Clause",
    "3-clause bsd license": "BSD-3-Clause",
    "apache 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "mpl 2.0": "MPL-2.0",
    "mpl-2.0": "MPL-2.0",
    "isc": "ISC",
    "isc license": "ISC",
    "psf": "PSF-2.0",
    "psf license": "PSF-2.0",
    "python software foundation license": "PSF-2.0",
    "unlicense": "Unlicense",
    "cc0": "CC0-1.0",
}


def _verdict_of_id(licence_id: str) -> Verdict:
    if licence_id in _ALLOWED:
        return "allowed"
    if _LGPL.fullmatch(licence_id):
        return "lgpl"
    if _GPL.fullmatch(licence_id):
        return "denied"
    return "unknown"


class _Malformed(ValueError):
    pass


def _tokens(expr: str) -> list[str]:
    return re.findall(r"\(|\)|[^\s()]+", expr)


def _parse_or(toks: list[str], i: int) -> tuple[Verdict, int]:
    v, i = _parse_and(toks, i)
    while i < len(toks) and toks[i].upper() == "OR":
        w, i = _parse_and(toks, i + 1)
        v = v if _RANK[v] >= _RANK[w] else w  # most permissive branch
    return v, i


def _parse_and(toks: list[str], i: int) -> tuple[Verdict, int]:
    v, i = _parse_with(toks, i)
    while i < len(toks) and toks[i].upper() == "AND":
        w, i = _parse_with(toks, i + 1)
        v = v if _RANK[v] <= _RANK[w] else w  # least permissive term
    return v, i


def _parse_with(toks: list[str], i: int) -> tuple[Verdict, int]:
    v, i = _parse_atom(toks, i)
    if i < len(toks) and toks[i].upper() == "WITH":
        if i + 1 >= len(toks) or toks[i + 1] in "()" or toks[i + 1].upper() in {"AND", "OR", "WITH"}:
            raise _Malformed("WITH needs an exception id")
        i += 2  # an exception only adds permissions: classify as the base licence
    return v, i


def _parse_atom(toks: list[str], i: int) -> tuple[Verdict, int]:
    if i >= len(toks):
        raise _Malformed("unexpected end")
    tok = toks[i]
    if tok == "(":
        v, i = _parse_or(toks, i + 1)
        if i >= len(toks) or toks[i] != ")":
            raise _Malformed("unbalanced parenthesis")
        return v, i + 1
    if tok == ")" or tok.upper() in {"AND", "OR", "WITH"}:
        raise _Malformed(f"unexpected {tok!r}")
    return _verdict_of_id(tok), i + 1


def classify(licence_expr: str) -> Verdict:
    """Classify an SPDX licence expression. Anything unparsable or unlisted is "unknown"."""
    toks = _tokens(licence_expr)
    if not toks:
        return "unknown"
    try:
        v, i = _parse_or(toks, 0)
    except _Malformed:
        return "unknown"
    return v if i == len(toks) else "unknown"


def licence_of(dist: Any) -> str:
    """The licence as an SPDX expression: License-Expression, else mapped classifiers, else a short License field."""
    meta = dist.metadata
    expr = (meta.get("License-Expression") or "").strip()
    if expr:
        return str(expr)
    mapped = [_CLASSIFIERS[c] for c in (meta.get_all("Classifier") or []) if c in _CLASSIFIERS]
    if mapped:
        return " OR ".join(dict.fromkeys(mapped))  # several licence classifiers mean a choice
    field = (meta.get("License") or "").strip()
    if field and len(field) <= 60 and "\n" not in field:
        return _SHORT_NAMES.get(field.casefold(), field)
    return ""


def _name(dist: Any) -> str:
    return re.sub(r"[-_.]+", "-", str(dist.metadata.get("Name") or "?")).lower()


def violations(dists: Iterable[Any]) -> list[str]:
    """One line per distribution whose licence is denied or unknown, NVIDIA_EXCEPTIONS excepted."""
    out = []
    for d in dists:
        name = _name(d)
        if name in NVIDIA_EXCEPTIONS:
            continue
        expr = licence_of(d)
        if REVIEWED.get(name) == expr:
            continue
        verdict = classify(expr)
        if verdict in ("denied", "unknown"):
            out.append(f"{name} {d.metadata.get('Version')}: {verdict} ({expr or 'no licence metadata'})")
    return out


def main() -> int:
    dists = {_name(d): d for d in importlib.metadata.distributions()}
    lgpl = sorted(n for n, d in dists.items() if classify(licence_of(d)) == "lgpl")
    bad = violations(dists.values())
    for n in lgpl:
        print(f"note: {n} is LGPL: allowed only unmodified, dynamically imported, and listed in NOTICE")
    for line in bad:
        print(f"licence violation: {line}")
    print(f"checked {len(dists)} distributions: {len(bad)} violation(s), {len(lgpl)} LGPL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
