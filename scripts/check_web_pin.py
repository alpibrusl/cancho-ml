#!/usr/bin/env python3
"""W1 and W2 (docs/serving-http.md sections 1 and 9): cancho-web, cancho-schema and the compiler's http-server are pinned package dependencies of `ml-serve`, and are not copied here.

    python3 scripts/check_web_pin.py             W1: the pins
    python3 scripts/check_web_pin.py --bins      W2: build/ml and build/ml-predict are the bytes a build without the three dependencies makes (needs the network and the compiler)
    CHECK_REMOTE=1 python3 scripts/check_web_pin.py     also asks GitHub whether the three revs have their stores, and whether the `web` store requires the `schema` rev
    TOML=FILE DOC=FILE ROOT=DIR    read other files (the mutants of scripts/mutants_http.py)

W1 fails when
* cancho.toml lacks [dependencies.web], [dependencies.schema] or [dependencies.server], or one of them has another git URL, another path, or a rev that is not 40 hex digits;
* the revs are not the ones docs/serving-http.md names (the lines "cancho-web rev: `...`", "cancho-schema rev: `...`", "cancho http-server rev: `...`"), or the server rev is not
  the compiler pin of [package] cancho;
* another [dependencies.X] points at one of the three repositories (a store that ml-serve does not import is not pinned);
* a .cho file under lib/, tools/ or tests/ declares module web, schema or http.server (a copy), or a file outside tools/ml-serve imports one of them;
* build/deps exists and lacks an installed module of the three.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = pathlib.Path(os.environ.get("ROOT") or pathlib.Path(__file__).resolve().parent.parent)
TOML = pathlib.Path(os.environ.get("TOML") or ROOT / "cancho.toml")
DOC = pathlib.Path(os.environ.get("DOC") or ROOT / "docs" / "serving-http.md")

DEPS = {
    "web": ("https://github.com/alpibrusl/cancho-web", ".cancho-vcs", "cancho-web rev", "web"),
    "schema": ("https://github.com/alpibrusl/cancho-schema", ".cancho-vcs", "cancho-schema rev", "schema"),
    "server": ("https://github.com/alpibrusl/cancho", "packages/http-server/.cancho-vcs", "cancho http-server rev", "http.server"),
}
MODULES = [v[3] for v in DEPS.values()]


def check():
    problems = []
    proj = tomllib.loads(TOML.read_text())
    deps = proj.get("dependencies", {})
    text = DOC.read_text()
    for name, (url, path, label, module) in DEPS.items():
        d = deps.get(name)
        if d is None:
            problems.append("cancho.toml has no [dependencies.%s]" % name)
            continue
        if d.get("git") != url:
            problems.append("[dependencies.%s]: git is %r, not %s" % (name, d.get("git"), url))
        if d.get("path") != path:
            problems.append("[dependencies.%s]: path is %r, not %s" % (name, d.get("path"), path))
        rev = d.get("rev", "")
        if not re.fullmatch(r"[0-9a-f]{40}", rev):
            problems.append("[dependencies.%s]: rev is not a full commit hash" % name)
            continue
        m = re.search(r"%s: `([0-9a-f]{40})`" % re.escape(label), text)
        if not m:
            problems.append("%s has no '%s: `<40 hex digits>`'" % (DOC.name, label))
        elif m.group(1) != rev:
            problems.append("[dependencies.%s]: rev %s is not the documented %s" % (name, rev, m.group(1)))
    pin = proj.get("package", {}).get("cancho", "")
    if "server" in deps and deps["server"].get("rev") != pin:
        problems.append("[dependencies.server]: rev %s is not the compiler pin %s of [package] cancho" % (deps["server"].get("rev"), pin))
    if "schema" in deps and "web" in deps:
        # `web` records its requirement on schema at SCHEMA_REV of cancho-web's ci.yml, and docs/serving-http.md names that rev with the schema pin
        m = re.search(r"SCHEMA_REV[^`\n]*`([0-9a-f]{40})`", text)
        if m and m.group(1) != deps["schema"].get("rev"):
            problems.append("[dependencies.schema]: rev %s is not the SCHEMA_REV %s that the web store requires" % (deps["schema"].get("rev"), m.group(1)))
    urls = {v[0]: k for k, v in DEPS.items()}
    for name, d in deps.items():
        if d.get("git") in urls and name not in DEPS:
            problems.append("[dependencies.%s] is a store of %s that ml-serve does not import (the documented list is web, schema, server)" % (name, d.get("git")))
        if name in DEPS and d.get("git") not in urls:
            pass
    for entry in proj.get("bin", []):
        for s in entry.get("sources", []):
            if s.startswith("..") or "/../" in s:
                problems.append("a sources entry of [[bin]] %s reaches outside this tree: %s" % (entry["name"], s))
    for sub in ("lib", "tools", "tests"):
        for f in sorted((ROOT / sub).rglob("*.cho")) if (ROOT / sub).exists() else []:
            body = f.read_text()
            for m in MODULES:
                if re.search(r"^module %s;" % re.escape(m), body, re.M):
                    problems.append("%s declares module %s, which is a pinned package's" % (f.relative_to(ROOT), m))
            if "tools/ml-serve/" not in str(f.relative_to(ROOT)) + "/" and not str(f.relative_to(ROOT)).startswith("tools/ml-serve/"):
                for m in MODULES:
                    if re.search(r"^import %s;" % re.escape(m), body, re.M):
                        problems.append("%s imports %s: only tools/ml-serve does" % (f.relative_to(ROOT), m))
    bd = ROOT / "build" / "deps"
    if bd.exists() and list(bd.glob("*.cho")):
        have = "\n".join(p.read_text() for p in bd.glob("*.cho"))
        for m in MODULES:
            if not re.search(r"^module %s;" % re.escape(m), have, re.M):
                problems.append("build/deps has no installed module %s (run cancho install)" % m)
    return problems


def remote(proj):
    problems = []
    if shutil.which("gh") is None:
        print("check_web_pin: remote check skipped (no gh)")
        return problems
    deps = proj["dependencies"]
    for name, (url, path, label, module) in DEPS.items():
        repo = url.replace("https://github.com/", "")
        rev = deps[name]["rev"]
        r = subprocess.run(["gh", "api", "repos/%s/contents/%s/manifest.json?ref=%s" % (repo, path, rev), "--jq", ".name"], capture_output=True, text=True)
        if r.returncode != 0:
            if "Could not resolve" in r.stderr or "dial tcp" in r.stderr or "connect" in r.stderr:
                print("check_web_pin: remote check skipped (offline)")
                return problems
            problems.append("%s has no %s/manifest.json at %s on GitHub" % (repo, path, rev))
    r = subprocess.run(["gh", "api", "repos/alpibrusl/cancho-web/contents/.cancho-vcs/requires/0.json?ref=" + deps["web"]["rev"], "--jq", ".content"], capture_output=True, text=True)
    if r.returncode == 0:
        import base64
        req = json.loads(base64.b64decode(r.stdout))
        got = req["lock"]["origin"]["rev"]
        if got != deps["schema"]["rev"]:
            problems.append("the web store at %s requires schema %s, not the pinned %s" % (deps["web"]["rev"], got, deps["schema"]["rev"]))
    return problems


# The compiler on Linux leaves the name of its temporary file, `cancho-llvm-<pid>-<n>.ll`, in the symbol table of a binary, so two builds of one tree differ in those
# digits (three builds of ml-predict on Linux x86-64 gave three hashes; on macOS two builds gave one). The digits are zeroed before the comparison, which keeps every byte
# length, and nothing else is changed (docs/ci.md section 10).
TMPNAME = re.compile(rb"cancho-llvm-(\d+)-(\d+)\.ll")


def unpid(b):
    return TMPNAME.sub(lambda m: b"cancho-llvm-" + b"0" * len(m.group(1)) + b"-" + b"0" * len(m.group(2)) + b".ll", b)


def pid_width(b):
    m = TMPNAME.search(b)
    return len(m.group(1)) if m else None


# The compiler at cancho 51524d2 numbers the symbols `lexs_<n>` over every module of the build, so a build with three more modules in the package set names (and orders) the same functions
# differently: the symbol and string tables differ (12,086 and 25,106 bytes of an ml of 2,252,128 on Linux x86-64, measured 2026-11-13 over 2,017 such names) and nothing else does:
# `.text`, `.rodata`, `.eh_frame` and every other section, and the program headers, are byte for byte the same. So what W2 compares is the program, not the file: every section but the
# symbol and string tables, and the program headers (ELF), or every section outside __LINKEDIT (Mach-O). A file that is neither is compared whole, as before.
import struct

ELF_SKIP = (b".symtab", b".strtab")


def elf_program(b):
    if len(b) < 64 or b[:4] != b"\x7fELF" or b[4] != 2 or b[5] != 1:
        return None
    try:
        machine, = struct.unpack_from("<H", b, 18)
        entry, phoff, shoff = struct.unpack_from("<QQQ", b, 24)
        phentsize, phnum, shentsize, shnum, shstrndx = struct.unpack_from("<HHHHH", b, 54)
        if (phnum and phentsize != 56) or shentsize != 64 or shnum == 0 or shoff + shnum * 64 > len(b) or phoff + phnum * 56 > len(b) or shstrndx >= shnum:
            return None
        sh = [struct.unpack_from("<IIQQQQIIQQ", b, shoff + 64 * i) for i in range(shnum)]
        names_off, names_size = sh[shstrndx][4], sh[shstrndx][5]
        if names_off + names_size > len(b):
            return None
        names = b[names_off:names_off + names_size]
        secs = []
        for (name, typ, flags, addr, off, size, link, info, align, entsize) in sh:
            end = names.find(b"\0", name)
            if name >= len(names) or end < 0:
                return None
            nm = names[name:end]
            if nm in ELF_SKIP:
                continue
            if typ != 8 and off + size > len(b):
                return None
            secs.append((nm, typ, flags, addr, size, align, entsize, None if typ == 8 else b[off:off + size]))
        phdrs = [struct.unpack_from("<IIQQQQQQ", b, phoff + 56 * i) for i in range(phnum)]
    except struct.error:
        return None
    return ("elf", machine, entry, tuple(phdrs), tuple(secs))


def macho_program(b):
    if len(b) < 32 or struct.unpack_from("<I", b, 0)[0] != 0xFEEDFACF:
        return None
    try:
        cputype, _sub, filetype, ncmds, _sizeofcmds = struct.unpack_from("<iiIII", b, 4)
        off, segs = 32, []
        for _ in range(ncmds):
            cmd, size = struct.unpack_from("<II", b, off)
            if size < 8:
                return None
            if cmd == 0x19:  # LC_SEGMENT_64
                segname = b[off + 8:off + 24].rstrip(b"\0")
                vmaddr, vmsize, _fo, _fs, maxprot, initprot, nsects, _fl = struct.unpack_from("<QQQQiiII", b, off + 24)
                if segname != b"__LINKEDIT":
                    secs = []
                    for k in range(nsects):
                        so = off + 72 + 80 * k
                        sect, seg = b[so:so + 16].rstrip(b"\0"), b[so + 16:so + 32].rstrip(b"\0")
                        addr, ssize, soff, align, _ro, _nr, sflags = struct.unpack_from("<QQIIIII", b, so + 32)
                        zero = (sflags & 0xFF) in (1, 0xC, 0x12)  # S_ZEROFILL, S_GB_ZEROFILL, S_THREAD_LOCAL_ZEROFILL
                        if not zero and soff + ssize > len(b):
                            return None
                        secs.append((sect, seg, addr, ssize, align, sflags, None if zero else b[soff:soff + ssize]))
                    segs.append((segname, vmaddr, vmsize, maxprot, initprot, tuple(secs)))
            off += size
    except struct.error:
        return None
    return ("macho", cputype, filetype, tuple(segs))


def program_of(b):
    return elf_program(b) or macho_program(b)


def same_binary(a, c):
    pa, pc = program_of(a), program_of(c)
    if pa is not None and pc is not None:
        return pa == pc
    if pa is None and pc is None:
        return unpid(a) == unpid(c)
    return False


def _elf(text=b"\x90" * 16, syms=(b"lexs_100", b"lexs_200", b"main"), extra=None, vaddr=0x1000, text_flags=6):
    """A minimal ELF64 little-endian file: .text, .symtab, .strtab, .shstrtab, one program header, and an optional extra section."""
    strtab = b"\0" + b"\0".join(syms) + b"\0"
    symtab = b"\0" * 24
    o = 1
    for n in syms:
        symtab += struct.pack("<IBBHQQ", o, 0x12, 0, 1, 0x1000, 8)
        o += len(n) + 1
    secs = [(b".text", 1, text_flags, text), (b".symtab", 2, 0, symtab), (b".strtab", 3, 0, strtab)]
    if extra is not None:
        secs.append((b".extra", 1, 2, extra))
    shstr, name_off = b"\0", {}
    for nm, *_ in secs + [(b".shstrtab",)]:
        name_off[nm] = len(shstr)
        shstr += nm + b"\0"
    secs.append((b".shstrtab", 3, 0, shstr))
    body, data_off, off = b"", {}, 64 + 56
    for nm, typ, fl, data in secs:
        data_off[nm] = off + len(body)
        body += data
    shoff = 64 + 56 + len(body)
    shdrs = struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    for nm, typ, fl, data in secs:
        shdrs += struct.pack("<IIQQQQIIQQ", name_off[nm], typ, fl, 0x1000 if nm == b".text" else 0, data_off[nm], len(data), 0, 0, 1, 24 if nm == b".symtab" else 0)
    ehdr = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\0" * 8 + struct.pack("<HHIQQQIHHHHHH", 2, 62, 1, 0x1000, 64, shoff, 0, 64, 56, 1, 64, len(secs) + 1, len(secs))
    phdr = struct.pack("<IIQQQQQQ", 1, 5, 120, vaddr, vaddr, len(text), len(text), 0x1000)  # the loaded segment ends with the code, as a real one ends before the symbol table
    return ehdr + phdr + body + shdrs


def _macho(text=b"\x90" * 16, linkedit=b"symbols-and-signature"):
    """A minimal Mach-O 64-bit file: a __TEXT segment with one __text section, and a __LINKEDIT segment that holds the symbols."""
    seg_text = b"__TEXT".ljust(16, b"\0") + struct.pack("<QQQQiiII", 0x1000, 0x1000, 0, 0x1000, 5, 5, 1, 0)
    sect = b"__text".ljust(16, b"\0") + b"__TEXT".ljust(16, b"\0") + struct.pack("<QQIIIIIIII", 0x1000 + 232 + 152, len(text), 232 + 152, 4, 0, 0, 0x80000400, 0, 0, 0)
    c_text = struct.pack("<II", 0x19, 72 + 80) + seg_text + sect
    body_off = 32 + len(c_text) + 72
    sect = sect[:48] + struct.pack("<I", body_off) + sect[52:]
    c_text = struct.pack("<II", 0x19, 72 + 80) + seg_text + sect
    seg_link = b"__LINKEDIT".ljust(16, b"\0") + struct.pack("<QQQQiiII", 0x2000, 0x1000, body_off + len(text), len(linkedit), 1, 1, 0, 0)
    c_link = struct.pack("<II", 0x19, 72) + seg_link
    hdr = struct.pack("<IiiIIIII", 0xFEEDFACF, 0x0100000C, 0, 2, 2, len(c_text) + len(c_link), 0, 0)
    return hdr + c_text + c_link + text + linkedit


def selftest_same_binary():
    """The comparison hides the symbol and string tables (ELF) or the __LINKEDIT segment (Mach-O) and the compiler's process id, and nothing else: it is shown able to fail."""
    base = b"\x7fELF....cancho-llvm-1810468-0.ll\x00lexs_argc\x00rest"
    e = _elf()
    cases = [
        (base, base.replace(b"1810468", b"1510769"), True),  # not a known format: another process id of the same width
        (base, base.replace(b"-0.ll", b"-1.ll"), True),  # another counter
        (base, base.replace(b"rest", b"rexp"), False),  # a real byte differs
        (base, base.replace(b"lexs_argc", b"lexs_argd"), False),
        (base, base.replace(b"1810468", b"99999"), False),  # another width shifts the table: not hidden
        (base, base[:-1], False),
        (b"no name here", b"no name herf", False),
        (e, e, True),
        (e, _elf(syms=(b"lexs_9", b"main", b"lexs_20000000")), True),  # other symbol names, in another order and of other lengths
        (e, _elf(text=b"\x90" * 15 + b"\xcc"), False),  # one byte of the code
        (e, _elf(text=b"\x90" * 17), False),  # a section of another size
        (e, _elf(text_flags=7), False),  # another flag of a section
        (e, _elf(extra=b"x"), False),  # one more section
        (e, _elf(vaddr=0x2000), False),  # another program header
        (e, e[:100], False),  # cut short: not parsed, so compared whole against a parsed one
        (_macho(), _macho(linkedit=b"other symbols and another signature!"), True),
        (_macho(), _macho(text=b"\x90" * 15 + b"\xcc"), False),
    ]
    out = ["same_binary self-test %d: %r" % (i, want) for i, (a, c, want) in enumerate(cases) if same_binary(a, c) != want]
    if not (program_of(e) and program_of(_macho())):
        out.append("same_binary self-test: the synthetic ELF or Mach-O file is not parsed")
    return out


def bins():
    """W2: ml and ml-predict, built from a copy of this tree without the three dependency blocks and without the ml-serve bin, are the same program as build/ml and build/ml-predict
    (every section but the symbol and string tables: see same_binary)."""
    cancho = os.environ.get("CANCHO") or str(ROOT.parent / "cancho-compiler-0207556" / "target" / "release" / "cancho")
    work = pathlib.Path(tempfile.mkdtemp(prefix="webpin-"))
    try:
        for d in ("lib", "tools", "generated", "schemas"):
            shutil.copytree(ROOT / d, work / d)
        text = TOML.read_text()
        for name in DEPS:
            text = re.sub(r"\[dependencies\.%s\]\n(?:[^\[\n].*\n)*" % name, "", text)
        text = re.sub(r"\[\[bin\]\]\nname = \"ml-serve\"\n(?:[^\[\n].*\n)*", "", text)
        (work / "cancho.toml").write_text(text)
        for cmd in ([cancho, "install"], [cancho, "build", "--bin", "ml"], [cancho, "build", "--bin", "ml-predict"]):
            r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
            if r.returncode != 0:
                return ["%s failed: %s" % (" ".join(cmd[1:]), (r.stdout + r.stderr)[-400:])]
        problems = selftest_same_binary()
        for b in ("ml", "ml-predict"):
            a = (ROOT / "build" / b).read_bytes()
            c = (work / "build" / b).read_bytes()
            # A process id of another width moves the symbol table by a byte, which no normalisation can undo; the id is new at every build, so build again until the widths agree.
            # When the width of build/<b> itself is out of reach (a machine whose process ids grew a digit since it was built), the reference is built again from a copy of this
            # tree with all its dependencies, until its width is the one of the build without them.
            for _ in range(4):
                if pid_width(a) == pid_width(c):
                    break
                subprocess.run([cancho, "build", "--bin", b], cwd=work, capture_output=True, text=True)
                c = (work / "build" / b).read_bytes()
            if pid_width(a) != pid_width(c):
                ref = pathlib.Path(tempfile.mkdtemp(prefix="webpin-ref-"))
                try:
                    for d in ("lib", "tools", "generated", "schemas"):
                        shutil.copytree(ROOT / d, ref / d)
                    shutil.copy(TOML, ref / "cancho.toml")
                    subprocess.run([cancho, "install"], cwd=ref, capture_output=True, text=True)
                    for _ in range(8):
                        subprocess.run([cancho, "build", "--bin", b], cwd=ref, capture_output=True, text=True)
                        a = (ref / "build" / b).read_bytes()
                        if pid_width(a) == pid_width(c):
                            break
                finally:
                    shutil.rmtree(ref, ignore_errors=True)
            if not same_binary(a, c):
                problems.append("build/%s is not the same program as the build without the three dependencies (%d and %d bytes; the sections other than the symbol and string tables differ)" % (b, len(a), len(c)))
        return problems
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    problems = check()
    proj = tomllib.loads(TOML.read_text())
    if not problems and os.environ.get("CHECK_REMOTE"):
        problems += remote(proj)
    if "--bins" in sys.argv and not problems:
        problems += bins()
    if problems:
        for p in problems:
            print("check_web_pin: " + p, file=sys.stderr)
        sys.exit(1)
    print("check_web_pin: ok (3 dependencies pinned, none copied%s)" % (", ml and ml-predict unchanged" if "--bins" in sys.argv else ""))


if __name__ == "__main__":
    main()
