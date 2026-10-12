import sys
L=int(sys.argv[1]); NC=int(sys.argv[2])
out=[]
out.append('''edition 5;
import std.io;
import ml.rng;
fn number_of[&s](text: &s [byte]) -> [] int { var n = 0; var i = 0; while i < len(text) { n = n * 10 + (int_of(text[i]) - '0'); i = i + 1; } return n; }
''')
# chunk fn
lines=[]
lines.append("fn chunk[&x, &y, &o](xs: &x [float], ys: &y [float], at: int, out: &!o [float]) -> [] int {")
lines.append("    let xr = xs[at..at + %d];"%(L*NC))
lines.append("    let yr = ys[at..at + %d];"%(L*NC))
for c in range(NC): lines.append("    var s%d = 0.0;"%c)
lines.append("    var j = 0;\n    while j < %d {"%(L*NC))
for c in range(NC): lines.append("        s%d = s%d + xr[j + %d] * yr[j + %d];"%(c,c,c,c))
lines.append("        j = j + %d;\n    }"%NC)
for c in range(NC): lines.append("    out[%d] = s%d;"%(c,c))
lines.append("    return 0;\n}")
out.append("\n".join(lines))
out.append('''
fn run[&x, &y, &o](xs: &x [float], ys: &y [float], n: int, out: &!o [float]) -> [] float {
    var tot = 0.0;
    var at = 0;
    while at + %d <= n {
        chunk(xs, ys, at, out);
        var k = 0;
        while k < %d { tot = tot + out[k]; k = k + 1; }
        at = at + %d;
    }
    return tot;
}
fn main(world: World) -> [] int {
    let Split { io, ffi, fs, heap, args, net, clock } = split(world);
    release(ffi); release(net); release(fs);
    var n = 0; var reps = 1;
    borrow args as &g in { n = number_of(arg(g, 1)); reps = number_of(arg(g, 2)); }
    var best = 1000000000;
    borrow mut heap as &!hp in {
        borrow mut io as &!ip in {
            borrow clock as &kc in {
                let x = box_slice(hp, n, 0.0);
                let y = box_slice(hp, n, 0.0);
                let o = box_slice(hp, 32, 0.0);
                borrow mut x as &!xw in { let s = contents(xw); var i = 0; while i < n { s[i] = rng.element(1, i); i = i + 1; } }
                borrow mut y as &!yw in { let s = contents(yw); var i = 0; while i < n { s[i] = rng.element(2, i); i = i + 1; } }
                borrow x as &xr in { borrow y as &yr in { borrow mut o as &!ow in {
                    var rep = 0;
                    var acc = 0.0;
                    while rep < 7 {
                        let t0 = clock_ms(kc);
                        var r = 0;
                        while r < reps { acc = acc + run(contents(xr), contents(yr), n, contents(ow)); r = r + 1; }
                        let t1 = clock_ms(kc);
                        if t1 - t0 < best { best = t1 - t0; }
                        rep = rep + 1;
                    }
                    io.print_int(ip, bits_of(acc) & 255);
                    io.newline(ip);
                } } }
                io.write_all(ip, "best_us_per_run ");
                io.print_int(ip, best * 1000 / reps);
                io.newline(ip);
                unbox_slice(hp, x); unbox_slice(hp, y); unbox_slice(hp, o);
            }
        }
    }
    release(args); release(clock); release(heap); release(io);
    return 0;
}
''' % (L*NC, NC, L*NC))
print("\n".join(out))
