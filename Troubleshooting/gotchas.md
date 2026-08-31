---
type: moc
---
# Language & Library Gotchas
- Matmul dtype strict → dtype= in EVERY constructor
- Matmul: no broadcasting; elementwise ops broadcast, skipping size-1 dims silently
- x[:n] = dim 0 (batch); sequence = x[:, :n]
- x**2 .mean() precedence trap; f-string braces are executable code
- Tensors → matplotlib only as .detach().cpu().numpy(); matplotlib.use("Agg") BEFORE pyplot
- savefig → close → optional show; plt.close(fig) frees memory
- pip --user shadows apt (~/.local vs /usr/lib); venv for isolation; font cache 30–60 s = legitimate
- glibc: freed tensors return to OS only at process exit
- seed 0 ⇒ bit-identical reruns = regression baseline
- fp16: x² underflows ~1e-8; eps + fp32-upcast are load-bearing
