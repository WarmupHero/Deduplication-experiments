# Cora

| | |
|---|---|
| Domain | Citations of machine-learning papers, segmented into fields |
| Source | [HPI's Cora dataset page](https://hpi.de/naumann/projects/repeatability/datasets/cora-dataset.html): `cora.tsv` and the duplicate-pair list `cora_DPL.tsv` |
| Licence | No licence named; HPI credits Andrew McCallum, who "permits us to use and modify the data" |
| Records → entities | 1,879 → 182 |
| Entity sizes | 64 singletons; largest entity 238 records |
| Ground truth | `cora_DPL.tsv` (64,578 duplicate pairs), closed transitively |
| Fields kept | the 16 citation columns: address, authors, booktitle, date, editor, institution, journal, month, note, pages, publisher, tech, title, type, volume, year |

Notes:

- The dataset with very large entities, where clustering algorithms are expected to differ most.
- Other copies of Cora (for example pyJedAI's) give 191 entities; this benchmark uses HPI's pair list, which gives 182.
- The segmentation is noisy (author names spill into other fields) and many fields are mostly empty.
- HPI refuses downloads that use Python's default user agent, so the download sends a browser-like one.
- Citations contain unbalanced quotes, so the files are read without quote handling.
