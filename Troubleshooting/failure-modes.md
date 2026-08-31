---
type: moc
---
# The Five Failure Modes
| Mode | Signature | First response | Confirm |
|---|---|---|---|
| **Crash** | traceback, prompt returns | read BOTTOM-UP; first frame in YOUR file = crime scene; numbers in the msg are evidence | fix, rerun, check earlier sections unchanged |
| **Block** | no output, process alive, no prompt | Ctrl+C (x2 if needed) → traceback shows parking spot | pgrep/pkill stragglers; free -h |
| **Silent number** | clean run, DONE, number "off" | what magnitude SHOULD print? ([[expected-magnitudes]]) | independent 2nd computation; same-seed rerun |
| **Quiet exit** | prompt back, no DONE, no traceback | echo $? → ls -la mtimes → tail the script | is the code you think is there actually there? |
| **Environment** | "No such file", wrong versions | pwd · which python3 · import x; print(x.__file__) | ~/.local shadows /usr/lib; venv for isolation |

Habits proven in this project: the checker is code too (2.16e+01) · duplicated numbers must
match (0.7632 vs 0.7179) · seed-0 bit-identical reruns = regression test · last printed line
brackets a hang.
