# Ablation: backward substitution with k ASCENDING (the first draft of the spec): the sum of row i starts with x[i+1], the x
# computed just before, so the whole solve is one serial chain of dependent subtractions. Replaces bwd_fast.
p = 'lib/linalg/chol.cho'
s = open(p).read()
i = s.index("fn bwd_fast")
j = s.index("// The solves.")
new = '''fn bwd_fast[&l, &b, &x](lv: &l [float], ls: int, bv: &b [float], xv: &!x [float], d: int) -> [] int {
    var i = d - 1;
    while i >= 0 {
        var s = bv[i];
        var k = i + 1;
        while k < d {
            s = s - lv[k * ls + i] * xv[k];
            k = k + 1;
        }
        xv[i] = s / lv[i * ls + i];
        i = i - 1;
    }
    return 0;
}

'''
s = s[:i] + new + s[j:]
open(p, 'w').write(s)
