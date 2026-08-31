---
type: moc
---
# Conventions

## Note types (frontmatter `type`)
script · concept · incident · project · moc · experiment

## Status
todo → doing → done. Update in frontmatter; INDEX tables mirror it.

## Naming
- Script notes: `NN-topic.md` (match the .py filename)
- Experiments: dated entries inside a script note's `## Experiments`
- Reusable incidents: `inc-NN-slug.md`, linked from [[error-log]]

## Linking
- Notes: `[[quantization]]` · Code: `[[06_quantization.py]]` · Charts: `![[outlier_sweep.png]]`
- Heading links: `[[failure-modes#Crash]]`

## Growth flows (how the vault stays alive)
1. Ran a TRY experiment → append to the script note's `## Experiments` ([[tpl-experiment]]).
   Numbers break [[baselines]] → investigate → probably an incident.
2. Something broke → ALWAYS a row in [[error-log]]; promote to `inc-*` note ([[tpl-incident]])
   if the lesson is reusable.
3. New script/concept → create from template, link from [[00 INDEX]] + related notes.

## Unsorted stuff
Lands in [[99 Inbox]] first; triage before it grows past ~10 lines.
