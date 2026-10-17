from pathlib import Path

path = Path("src/msma/s4pca.py")
if not path.exists():
    raise SystemExit(f"File not found: {path.resolve()}")

text = path.read_text(encoding="utf-8")

if '"Interpretation"' in text:
    print("Interpretation section is already present; no change made.")
    raise SystemExit(0)

old = '''            out += ["", "Diagnostic rule", "---------------", n["rule"]]\n        return "\\n".join(out)'''

new = '''            out += ["", "Diagnostic rule", "---------------", n["rule"]]\n            if n["warning"]:\n                out += [\n                    "",\n                    "Interpretation",\n                    "--------------",\n                    (\n                        "Near-zero initial loading candidates were detected. "\n                        "Normalization may amplify platform-dependent floating-point "\n                        "differences in later sparse components. The fitted result "\n                        "remains available, but cross-platform numerical identity "\n                        "may not hold for the flagged components."\n                    ),\n                ]\n        return "\\n".join(out)'''

if old not in text:
    raise SystemExit(
        "Expected insertion point was not found. Open src/msma/s4pca.py and "
        "inspect the detailed-summary block near 'Diagnostic rule'."
    )

path.write_text(text.replace(old, new, 1), encoding="utf-8")
print(f"Patched: {path}")
