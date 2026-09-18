"""
One-time (per-checkpoint-revision) fetch of the AuroraV1p5 checkpoint +
static-field pickle from Hugging Face, into a local cache directory.

Unlike backends/aifs/aifs-single-2.0 and backends/fcn3/fourcastnet3 (each a
git submodule of a self-contained HF model-package repo), microsoft/aurora
on Hugging Face is a shared monorepo bundling MANY unrelated checkpoint
variants (0.1, several 0.25 variants, wave, air-pollution, ensemble, plus
test pickles) -- a git submodule would force pulling everything (tens of
GB) just to get the ~4.9GB we actually need. The `aurora` package itself
fetches checkpoints via `huggingface_hub.hf_hub_download` (a selective,
single-file, revision-pinned download), so this script just does the same
thing directly, ahead of time, from a location with internet access.

Must be run from a LOGIN node (H100 compute nodes have no internet access --
same constraint as the other two backends' ERA5 IC fetching). The cache
directory (backends/aurora/hf_cache/) is gitignored; AuroraModel reads from
it at runtime via HF_HUB_CACHE / an explicit cache_dir, never re-fetching
if the files are already there.

Run from the repo root: `python backends/aurora/aurora_prefetch_checkpoint.py`
"""

from huggingface_hub import hf_hub_download

REPO = "microsoft/aurora"
REVISION = "a96afd7ee6d65e3bd2d476f3be798a25a56f2296"  # AuroraV1p5.default_checkpoint_revision
CACHE_DIR = "backends/aurora/hf_cache"
FILES = ["aurora-0.25-v1.5.ckpt", "aurora-0.25-v1.5-static.pickle"]

if __name__ == "__main__":
    for fname in FILES:
        path = hf_hub_download(repo_id=REPO, filename=fname, revision=REVISION, cache_dir=CACHE_DIR)
        print(f"{fname} -> {path}")
