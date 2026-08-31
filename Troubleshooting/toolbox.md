---
type: moc
---
# Command Toolbox
| Question | Command | Read |
|---|---|---|
| finished cleanly? | python3 x.py; echo exit=$? | 0 = clean |
| who holds RAM? | ps -eo pid,stat,rss,etime,cmd --sort=-rss | STAT states below |
| interactive | htop (F6 sort RESMEM, F9 kill) | fastest loop |
| kill | pgrep -af name → pkill -f name → kill -9 | look before shooting |
| stopped jobs | jobs → kill -9 %1 | Ctrl+Z victims |
| memory truth | free -h | "available", not "used" |
| headless | MPLBACKEND=Agg python3 x.py | show() becomes no-op |
| guard rail | timeout 300 python3 x.py | auto-kill 5 min |
| code really there? | tail -30 x.py · grep -n "plt.show" x.py | merge bugs die here |
| find anything | find ~ -name "06_quant*" 2>/dev/null | cwd-independent |
| artifact check | ls -la *.png | mtime = when actually written |
| lib identity | python3 -c "import m; print(m.__version__, m.__file__)" | ~/.local shadows /usr/lib |

STATs: S sleeping (waiting on you) · T stopped (Ctrl+Z, all RAM held) · R running ·
Z zombie (can't kill, no RAM) · D unkillable until IO done.
Signals: Ctrl+C=INT (releases) · Ctrl+Z=TSTP (PAUSE, all held) · kill=TERM · kill -9=KILL ·
close terminal=HUP (reaps children).
