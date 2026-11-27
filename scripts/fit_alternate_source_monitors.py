"""Replay a fixed source-only selection method on alternate fitting architectures."""
import argparse

from fit_source_only_monitor import fit_and_freeze
from qualify_single_source_monitor import OUTPUT, TEST_ARCHIVES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=TEST_ARCHIVES, required=True)
    parser.add_argument("--changes-only", action="store_true")
    args = parser.parse_args()
    architecture, changes = args.source, args.changes_only
    root = TEST_ARCHIVES[architecture]
    name = f"{architecture}-{'changes' if changes else 'broad'}"
    fit_and_freeze(OUTPUT / f"{name}-bank.json", OUTPUT / f"{name}-source.json",
                   architecture=architecture, archive_path=root / f"{architecture}-advice-refit-inputs.json",
                   report_path=root / "report.json", changes_only=changes)


if __name__ == "__main__":
    main()
