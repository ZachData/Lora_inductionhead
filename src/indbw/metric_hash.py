"""Content hash of the metric-defining code in algebra.py and probes.py.

CLAUDE.md, tier 0: "CI hashes the metric-defining functions in
`probes.py` and `algebra.py`. If the hash changes and `METRIC_VERSION`
did not, the build fails. Changing a metric definition silently
invalidates every prior result; this makes that impossible rather than
merely discouraged."

`.github/workflows/ci.yml` carried this as an explicit `echo` stub
("not yet implemented ... design it test-first in a work session rather
than inventing the mechanism here"). This module is that design.

**What is hashed.** Every top-level definition in each metric module --
functions, classes, and module-level constants -- with docstrings
stripped and the AST normalized via `ast.dump`. Concretely:

  - Reformatting, comments, blank lines, and docstring edits do NOT
    change the hash. A metric version bump forced by a typo fix would
    make the mechanism annoying enough to route around, and routing
    around it is the failure this exists to prevent.
  - Any change to executable code DOES change the hash, including
    renames and reordered arguments. Conservative on purpose: a false
    positive costs one deliberate version bump, a false negative costs
    a silently invalidated results record.
  - Imports are excluded (they carry no metric semantics on their own),
    but an unrecognized top-level construct raises rather than being
    skipped -- a metric must not be able to hide inside a statement
    form this module forgot to handle.

**Why a lockfile and not a hash embedded in source.** The check needs
to compare *this* commit's metric code against the code that the stored
`METRIC_VERSION` was declared for. That baseline has to be a committed
artifact; `metric_hash.json` is it. Regenerating it is a deliberate act
(`python scripts/check_metric_hash.py --update`) that refuses to run
unless `METRIC_VERSION` in schema.py was bumped first -- which is what
mechanically enforces CLAUDE.md's rule rather than restating it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from indbw.schema import METRIC_VERSION

#: major.minor only -- the one confirmed source of cross-version drift
#: (REVIEW.md 2026-09-01) is PEP 695's `type_params=[]` field, which
#: `ast.dump` started emitting in 3.12; patch releases don't move this.
CURRENT_PYTHON_VERSION: Final[str] = ".".join(platform.python_version_tuple()[:2])

#: Modules whose definitions constitute the metric surface (CLAUDE.md
#: names exactly these two). gates.py, lora.py, train.py and the scripts
#: are deliberately excluded -- see gates.py's own module docstring for
#: why gate-detection is not a per-example metric.
METRIC_MODULES: Final[tuple[str, ...]] = ("algebra.py", "probes.py")

SRC_DIR: Final[Path] = Path(__file__).resolve().parent
LOCK_PATH: Final[Path] = Path(__file__).resolve().parents[2] / "metric_hash.json"


@dataclass(frozen=True)
class MetricHash:
    """The metric surface's content hash at one `METRIC_VERSION`.

    `aggregate` covers the code only -- it deliberately does not mix in
    `metric_version`, because the whole point is to detect the case
    where the code moved and the version did not.
    """

    metric_version: str
    aggregate: str
    definitions: dict[str, dict[str, str]]  # module filename -> {def name -> per-def hash}
    #: Interpreter that computed `aggregate`/`definitions`. "unknown" for
    #: lockfiles written before this field existed (2026-08-14's original
    #: commit) -- treated as "don't know, don't warn" rather than a match
    #: or a mismatch.
    python_version: str = "unknown"

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_version": self.metric_version,
            "aggregate": self.aggregate,
            "definitions": self.definitions,
            "python_version": self.python_version,
        }


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _strip_docstrings(tree: ast.AST) -> None:
    """Drop the docstring statement from every scope in `tree`, in place."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        if node.body and _is_docstring(node.body[0]):
            # A scope whose entire body is its docstring still needs a
            # body to remain a valid AST; `pass` is the neutral filler.
            node.body = node.body[1:] or [ast.Pass()]


def _definition_name(node: ast.stmt) -> str:
    """Stable name for one top-level definition.

    Raises on any construct this module does not know how to name --
    fail loud rather than let an unrecognized top-level statement drop
    out of the hashed surface (CLAUDE.md, "Fail-fast in production
    paths").
    """
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return node.name
    if isinstance(node, ast.Assign):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not names:
            raise ValueError(f"unsupported top-level assignment target: {ast.dump(node)}")
        return ",".join(names)
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    raise ValueError(
        f"unhandled top-level construct {type(node).__name__} in a metric module -- "
        "metric_hash.py must be taught how to name it before it can be hashed"
    )


def normalized_definitions(source: str) -> dict[str, str]:
    """Map each top-level definition's name to its normalized AST dump.

    Imports (and `from __future__` statements) are excluded; so is the
    module docstring. Everything else must be nameable by
    `_definition_name` or this raises.
    """
    tree = ast.parse(source)
    _strip_docstrings(tree)
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            continue
        name = _definition_name(node)
        if name in out:
            raise ValueError(f"duplicate top-level definition {name!r} in a metric module")
        out[name] = ast.dump(node)
    return out


def _short(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def compute_metric_hash(
    src_dir: Path = SRC_DIR, metric_version: str = METRIC_VERSION
) -> MetricHash:
    """Hash the metric surface as it exists on disk right now."""
    definitions: dict[str, dict[str, str]] = {}
    for module in METRIC_MODULES:
        path = src_dir / module
        if not path.exists():
            raise FileNotFoundError(f"metric module {module} not found under {src_dir}")
        dumps = normalized_definitions(path.read_text())
        definitions[module] = {name: _short(dump) for name, dump in sorted(dumps.items())}
    aggregate = hashlib.sha256(json.dumps(definitions, sort_keys=True).encode()).hexdigest()
    return MetricHash(
        metric_version=metric_version,
        aggregate=aggregate,
        definitions=definitions,
        python_version=CURRENT_PYTHON_VERSION,
    )


def load_lock(path: Path = LOCK_PATH) -> MetricHash:
    data = json.loads(Path(path).read_text())
    return MetricHash(
        metric_version=str(data["metric_version"]),
        aggregate=str(data["aggregate"]),
        definitions={m: dict(d) for m, d in data["definitions"].items()},
        python_version=str(data.get("python_version", "unknown")),
    )


def write_lock(mh: MetricHash, path: Path = LOCK_PATH) -> None:
    Path(path).write_text(json.dumps(mh.to_dict(), indent=2, sort_keys=True) + "\n")


def _changed_definitions(lock: MetricHash, current: MetricHash) -> list[str]:
    problems: list[str] = []
    for module in METRIC_MODULES:
        old = lock.definitions.get(module, {})
        new = current.definitions.get(module, {})
        for name in sorted(set(old) | set(new)):
            if name not in old:
                problems.append(f"  {module}::{name} added")
            elif name not in new:
                problems.append(f"  {module}::{name} removed")
            elif old[name] != new[name]:
                problems.append(f"  {module}::{name} changed ({old[name]} -> {new[name]})")
    return problems


def check(src_dir: Path = SRC_DIR, lock_path: Path = LOCK_PATH) -> list[str]:
    """Return a list of human-readable problems; empty means the gate passes.

    Two independent failures are reported:

      1. The lockfile's `metric_version` disagrees with schema.py's
         `METRIC_VERSION` -- the lockfile is stale regardless of whether
         the code moved.
      2. The metric code's hash disagrees with the lockfile -- a metric
         definition changed. Legitimate only alongside a `METRIC_VERSION`
         bump, and the bump alone is not enough: the lockfile has to be
         regenerated in the same commit so the next commit compares
         against a real baseline.
    """
    current = compute_metric_hash(src_dir=src_dir)
    lock = load_lock(lock_path)
    problems: list[str] = []
    if lock.metric_version != current.metric_version:
        problems.append(
            f"metric_hash.json records METRIC_VERSION {lock.metric_version!r}, but "
            f"schema.py says {current.metric_version!r}. Regenerate the lockfile: "
            "python scripts/check_metric_hash.py --update"
        )
    if lock.aggregate != current.aggregate:
        if lock.python_version != "unknown" and lock.python_version != current.python_version:
            # REVIEW.md 2026-09-01: this exact mismatch is what produced a
            # false-positive metric change under 3.12 (ast.dump's
            # type_params=[] field, PEP 695). The literal-change message
            # below tells whoever hits it to bump METRIC_VERSION and
            # regenerate -- which would bake this interpreter's AST-dump
            # shape in as the new baseline and break the check for every
            # other machine. Refuse to recommend that here.
            problems.append(
                "the metric surface's hash differs from metric_hash.json, but the "
                f"lockfile was generated under Python {lock.python_version} and this "
                f"interpreter is {current.python_version} -- ast.dump's output is not "
                "stable across Python versions (confirmed: 3.12's type_params=[] field, "
                "PEP 695), so this may be version skew, not a real metric change. "
                f"Re-run this check under Python {lock.python_version} before concluding "
                "anything changed. Do not bump METRIC_VERSION or regenerate the lockfile "
                "from this interpreter."
            )
        else:
            problems.append(
                "the metric surface changed since metric_hash.json was written:\n"
                + "\n".join(_changed_definitions(lock, current))
                + f"\n  aggregate {lock.aggregate[:16]} -> {current.aggregate[:16]}\n"
                "Changing a metric definition invalidates every prior results record. "
                "Bump METRIC_VERSION in src/indbw/schema.py, then regenerate: "
                "python scripts/check_metric_hash.py --update"
            )
    return problems


def update(src_dir: Path = SRC_DIR, lock_path: Path = LOCK_PATH) -> MetricHash:
    """Regenerate the lockfile. Refuses if the metric code changed but
    `METRIC_VERSION` did not -- this refusal *is* CLAUDE.md's rule.
    """
    current = compute_metric_hash(src_dir=src_dir)
    if Path(lock_path).exists():
        lock = load_lock(lock_path)
        if lock.aggregate != current.aggregate and lock.metric_version == current.metric_version:
            raise ValueError(
                "refusing to update metric_hash.json: the metric surface changed while "
                f"METRIC_VERSION stayed at {current.metric_version!r}.\n"
                + "\n".join(_changed_definitions(lock, current))
                + "\nBump METRIC_VERSION in src/indbw/schema.py first."
            )
    write_lock(current, lock_path)
    return current
