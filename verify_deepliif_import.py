"""
verify_deepliif_import.py
─────────────────────────
Verification script for DeepLIIF integration.

Verifies the `deepliif` package is in the root directory and attempts
to import `define_G` from `deepliif.models.networks`.

Run from the UNIStainNet project root:
    python verify_deepliif_import.py
"""

import sys
import os

# ── 1. Resolve paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPLIIF_ROOT = os.path.join(SCRIPT_DIR, "deepliif")

print("=" * 60)
print("DeepLIIF Import Verification")
print("=" * 60)
print(f"[INFO] Project root  : {SCRIPT_DIR}")
print(f"[INFO] deepliif root : {DEEPLIIF_ROOT}")

# ── 2. Sanity-check the package folder exists ─────────────────────────────────
if not os.path.isdir(DEEPLIIF_ROOT):
    print(f"\n[ERROR] Directory not found: {DEEPLIIF_ROOT}")
    print("        Make sure the deepliif package is in the project root.")
    sys.exit(1)

# ── 3. Attempt the import ─────────────────────────────────────────────────────
print("\n[INFO] Attempting:  from deepliif.models.networks import define_G")

try:
    from deepliif.models.networks import define_G  # noqa: E402

    # Verify it is callable (not just an object that happened to shadow the name)
    assert callable(define_G), "define_G was imported but is not callable!"

    print("\n[SUCCESS] Import resolved correctly.")
    print(f"          Module  : {define_G.__module__}")
    print(f"          Object  : {define_G}")
    print("\nDeepLIIF generator network is ready for integration. ✓")

except ModuleNotFoundError as exc:
    print(f"\n[FAILED] ModuleNotFoundError: {exc}")
    print("\n── Debugging: contents of ./deepliif ──────────────────────────")

    for entry in sorted(os.listdir(DEEPLIIF_ROOT)):
        full = os.path.join(DEEPLIIF_ROOT, entry)
        kind = "DIR " if os.path.isdir(full) else "FILE"
        print(f"  [{kind}]  {entry}")

    models_pkg = os.path.join(DEEPLIIF_ROOT, "models")
    if os.path.isdir(models_pkg):
        print(f"\n── Debugging: contents of ./deepliif/models ──────")
        for entry in sorted(os.listdir(models_pkg)):
            full = os.path.join(models_pkg, entry)
            kind = "DIR " if os.path.isdir(full) else "FILE"
            print(f"  [{kind}]  {entry}")

    print(f"\n── Current sys.path ────────────────────────────────────────")
    for i, p in enumerate(sys.path):
        print(f"  [{i:02d}]  {p}")

    sys.exit(1)

except Exception as exc:
    print(f"\n[FAILED] Unexpected error during import: {type(exc).__name__}: {exc}")
    raise
