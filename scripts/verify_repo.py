#!/usr/bin/env python3
"""Phased integrity checks for the repository delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / ".github" / "import-manifest.json"


class Verification:
    def __init__(self, require_complete: bool) -> None:
        self.require_complete = require_complete
        self.failures: list[str] = []
        self.pending: list[str] = []

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def unavailable(self, phase: str, reason: str) -> None:
        message = f"{phase}: {reason}"
        if self.require_complete:
            self.fail(message)
        else:
            self.pending.append(message)

    def manifest(self) -> dict | None:
        if not MANIFEST.is_file():
            return None
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            self.fail(f"manifest: cannot parse {MANIFEST.relative_to(ROOT)}: {error}")
            return None

    def check_manifest_group(self, phase: str, predicate) -> None:
        manifest = self.manifest()
        if manifest is None:
            self.unavailable(phase, "import manifest has not been delivered")
            return
        entries = [entry for entry in manifest.get("files", []) if predicate(entry)]
        if not entries:
            self.fail(f"{phase}: manifest selects no files")
            return
        missing = 0
        checked = 0
        for entry in entries:
            path = ROOT / entry["path"]
            if not path.is_file():
                missing += 1
                continue
            checked += 1
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != entry["sha256"]:
                self.fail(f"{phase}: hash mismatch: {entry['path']}")
        if missing:
            self.unavailable(phase, f"{missing}/{len(entries)} registered files are not delivered")
        print(f"{phase}: verified {checked}/{len(entries)} delivered files")

    def source(self) -> None:
        self.check_manifest_group("source", lambda entry: entry.get("phase") == "source")

    def historical(self) -> None:
        self.check_manifest_group(
            "historical",
            lambda entry: entry["path"].startswith(("results/", "evidence/carried_forward_pre_grid/")),
        )

    def final_evidence(self) -> None:
        self.check_manifest_group(
            "final-evidence",
            lambda entry: entry.get("phase") == "evidence"
            and not entry["path"].startswith("evidence/carried_forward_pre_grid/"),
        )

    def paper(self) -> None:
        required = [
            ROOT / "report" / "report.tex",
            ROOT / "presentation" / "presentation.tex",
            ROOT / "contribution_report.tex",
        ]
        missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
        if missing:
            self.unavailable("paper", "missing " + ", ".join(missing))
        else:
            print("paper: report, presentation, and contribution sources are present")

    def finish(self) -> int:
        for message in self.pending:
            print(f"PENDING: {message}")
        for message in self.failures:
            print(f"ERROR: {message}", file=sys.stderr)
        if self.failures:
            print(f"verification failed with {len(self.failures)} error(s)", file=sys.stderr)
            return 1
        if self.pending:
            print(f"verification completed with {len(self.pending)} pending phase(s)")
        else:
            print("verification passed")
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="store_true")
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--final-evidence", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = args.all or any((args.source, args.historical, args.final_evidence, args.paper))
    if not selected:
        raise SystemExit("select a phase or use --all")
    verification = Verification(args.require_complete)
    if args.all or args.source:
        verification.source()
    if args.all or args.historical:
        verification.historical()
    if args.all or args.final_evidence:
        verification.final_evidence()
    if args.all or args.paper:
        verification.paper()
    return verification.finish()


if __name__ == "__main__":
    raise SystemExit(main())
