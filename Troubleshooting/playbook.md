---
type: moc
---
# Playbook (run in order)
0. Hygiene: pwd · free -h (available col) · pgrep -af python (ghosts?)
1. Run with telemetry: python3 x.py 2>&1 | tee run.log; echo exit=$?  (must end with DONE)
2. Classify the ending → pick mode from [[failure-modes]]
3. Crash: bottom-up read → your frame → smallest fix → check the backward counterpart if shapes changed
4. Block: Ctrl+C → parking spot → GUI/input → Agg / flag-off → kill stragglers
5. Quiet exit: $? → artifact mtime → tail -30 → grep -n "suspect"
6. Silent: expected magnitude → verify the CHECKER first → cross-check duplicate computation → seed rerun
7. Environment: cd with quotes · absolute paths · __version__ + __file__ of every suspect
8. Verify: acceptance criterion defined BEFORE rerunning ([[baselines]])
9. Bank it: add a row to [[error-log]] with mode tag
