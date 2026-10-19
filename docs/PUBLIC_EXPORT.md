# Regenerating the public source snapshot

Run from the source checkout after committing the documentation and code:

```sh
bash scripts/export_public.sh /tmp/SUC-LM-PUBLIC-1.4.7 HEAD
```

Choose a new output directory; the exporter refuses an existing path. The optional second argument selects a committed Git revision and defaults to HEAD. Uncommitted files are not exported. The resulting local repository has one snapshot commit and no inherited private history or configured remote. `PUBLIC_SNAPSHOT.txt` records the source revision.

The export includes current source, six updated model cards, historical compression reports and the qualified runtime 1.4.7 summaries. `.public-exclude` removes raw machine-specific results, internal audits/queues, build manifests and binary packages. Text paths, logins and LAN addresses become public examples; internal optimization progress becomes a concise acceptance summary. A final private-reference scan must pass before the snapshot is committed. Source files in the original checkout are not rewritten.

`SUC-LM-PUBLIC` can have independent commits: never delete its `.git` or replace its history with this generated repository. Review the staged snapshot and integrate its file changes into the existing public checkout separately, preserving any public-only edits. This script does not push to GitHub, upload Hugging Face cards, install packages or alter the existing public checkout. Local installers remain under `SUCStudio/dist` in the private source tree.
