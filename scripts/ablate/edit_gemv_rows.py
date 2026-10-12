# Ablation: gemv_n with N rows in flight (env NROWS, default 4). Regenerates the row kernel of lib/linalg/gemv.cho for N rows.
import os
N = int(os.environ.get("NROWS", "4"))
p = 'lib/linalg/gemv.cho'
s = open(p).read()
i = s.index("fn rows4")
j = s.index("fn row1")
lines = ["fn rows4[&x, &w, &y](xs: &x [float], rs: int, wv: &w [float], ys: &!y [float], i: int, d: int) -> [] int {"]
for k in range(N):
    lines.append("    let r%d = xs[(i + %d) * rs..(i + %d) * rs + d];" % (k, k, k))
lines.append("    let wr = wv[0..d];")
for k in range(N):
    lines.append("    var z%d = 0.0;" % k)
lines.append("    var j = 0;\n    while j < d {\n        let wj = wr[j];")
for k in range(N):
    lines.append("        z%d = z%d + r%d[j] * wj;" % (k, k, k))
lines.append("        j = j + 1;\n    }")
for k in range(N):
    lines.append("    ys[i + %d] = z%d;" % (k, k))
lines.append("    return 0;\n}\n\n")
s = s[:i] + "\n".join(lines) + s[j:]
s = s.replace("while i + 4 <= i1 {\n        rows4(xs, rs, wv, ys, i, d);\n        i = i + 4;", "while i + %d <= i1 {\n        rows4(xs, rs, wv, ys, i, d);\n        i = i + %d;" % (N, N))
open(p, 'w').write(s)
