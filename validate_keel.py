"""Validate all 42 KEEL files: load, compute stats, diff vs the paper table."""
import numpy as np
from loaders import parse_keel
KEEL_DIR = os.environ.get("CISO_KEEL", "data/keel")
RESULTS  = os.environ.get("CISO_RESULTS", "results")
os.makedirs(RESULTS, exist_ok=True)


# Expected values transcribed from the user's LaTeX table (n, d, IR)
PAPER_TABLE = {
    "abalone19": (4174, 8, 129.4),   "abalone9-18": (731, 8, 16.4),
    "ecoli3": (336, 7, 8.6),         "dermatology-6": (358, 34, 16.9),
    "ecoli-0_vs_1": (220, 7, 1.9),   "ecoli1": (336, 7, 3.3),
    "ecoli4": (336, 7, 15.8),        "flare-F": (1066, 11, 23.8),
    "glass-0-1-2-3_vs_4-5-6": (214, 9, 3.2),
    "glass-0-1-6_vs_2": (192, 9, 10.3),
    "glass-0-1-6_vs_5": (184, 9, 19.4),
    "glass0": (214, 9, 2.1),         "glass1": (214, 9, 1.8),
    "glass2": (214, 9, 11.6),        "glass4": (214, 9, 15.5),
    "glass5": (214, 9, 22.8),        "glass6": (214, 9, 6.3),
    "haberman": (306, 3, 2.8),       "new-thyroid1": (215, 5, 5.1),
    "newthyroid2": (215, 5, 5.1),    "pima": (768, 8, 1.9),
    "segment0": (2308, 19, 6.0),     "shuttle-c0-vs-c4": (1829, 9, 13.9),
    "vehicle0": (846, 18, 3.2),      "vehicle1": (846, 18, 2.9),
    "vehicle2": (846, 18, 2.9),      "vehicle3": (846, 18, 3.0),
    "vowel0": (988, 13, 10.0),       "car-good": (1728, 6, 24.0),
    "winequality-red-4": (1599, 11, 29.1),
    "wisconsin": (683, 9, 1.9),
    "yeast-0-5-6-7-9_vs_4": (528, 8, 9.3),
    "yeast-1-2-8-9_vs_7": (947, 8, 30.6),
    "yeast-1-4-5-8_vs_7": (693, 8, 22.1),
    "yeast-1_vs_7": (459, 7, 14.3),  "yeast-2_vs_4": (514, 8, 9.1),
    "yeast-2_vs_8": (1484, 8, 28.1),
    "yeast1": (1484, 8, 2.5),        "yeast3": (1484, 8, 8.1),
    "yeast4": (1484, 8, 28.1),       "yeast5": (1484, 8, 32.7),
    "yeast6": (1484, 8, 41.4),
}

rows, problems, metas = [], [], []
for name, (en, ed, eir) in sorted(PAPER_TABLE.items()):
    try:
        X, y, m = parse_keel(os.path.join(KEEL_DIR, f"{name}.dat"))
    except Exception as e:
        problems.append(f"{name}: LOAD FAILED -- {type(e).__name__}: {e}")
        continue
    metas.append(m)
    flags = []
    if m["n"] != en:
        flags.append(f"n {m['n']} vs paper {en}")
    if m["d_raw"] != ed:
        flags.append(f"d {m['d_raw']} vs paper {ed}")
    if abs(m["ir"] - eir) > 0.1:
        flags.append(f"IR {m['ir']} vs paper {eir}")
    if not np.all(np.isfinite(X)):
        flags.append("non-finite features")
    if m["positive_rule"] != "keel-declared":
        flags.append(f"positive rule = {m['positive_rule']}")
    status = "OK" if not flags else "DIFF: " + "; ".join(flags)
    if flags:
        problems.append(f"{name}: {status}")
    rows.append((name, m["n"], m["d_raw"], m["d_encoded"], m["n_pos"],
                 m["ir"], m["n_imputed_cells"], status))

print(f"{'dataset':<24}{'n':>6}{'d':>4}{'enc':>5}{'N1':>5}{'IR':>8}"
      f"{'imp':>5}  status")
for r in rows:
    print(f"{r[0]:<24}{r[1]:>6}{r[2]:>4}{r[3]:>5}{r[4]:>5}{r[5]:>8}"
          f"{r[6]:>5}  {r[7]}")
print(f"\nLoaded {len(metas)}/42.")
if problems:
    print("\n=== DISCREPANCIES vs paper table ===")
    for p in problems:
        print(" *", p)
else:
    print("Paper table fully confirmed by the data.")

# minority-size safeguard check (protocol 3: n_splits reduction / exclusion)
tiny = [(m["name"], m["n_pos"]) for m in metas if m["n_pos"] < 5]
print("\nDatasets with N1 < 5 (n_splits will reduce):", tiny or "none")
