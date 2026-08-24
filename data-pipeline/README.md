# Data Pipeline

- Download AFdb: `aria2c` or HF `DeepMind/afdb` (501k)
- PDB: `rsync` from `rsync.wwpdb.org`
- Align: TM-score / lDDT via `openstructure` or Bio.PDB + custom
- Extract: pLDDT from AFdb json, phi/psi via `Bio.PDB`, PAE from AFdb `predicted_aligned_error`
- Disorder: SETH or `metapredict` fallback
- Cluster split: UniRef50 30% identity to prevent leakage
- Manifest: `sequence, target (0-100), plddt, phi, psi, pae` -> `data/manifest.jsonl`
- Upload to HF Dataset `bhumika-tewari-282006/fusion-afdb-quality` with `datasets` library
