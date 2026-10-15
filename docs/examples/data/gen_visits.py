#!/usr/bin/env python3
"""Writes docs/examples/data/visits.csv: 12 customers with 10 visits each (docs/splits.md).

    python3 docs/examples/data/gen_visits.py            # rewrite visits.csv next to this file
    python3 docs/examples/data/gen_visits.py --check    # exit 1 if the committed file differs

The rows are in no order of date and a customer's rows are spread through the file, so a split by group and a split by time both differ
from a split of the first rows: customer, visit_date (an ISO date), basket (a float), items (an integer) and returning (0 or 1: whether the visit
was followed by another within 30 days; it depends on the customer and on the basket, with noise). The generator uses python's random.Random(42)
and nothing else, so the file is reproduced byte for byte.
"""
import datetime
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent


def rows():
    rnd = random.Random(42)
    base = datetime.date(2024, 1, 1)
    taste = {"c%02d" % i: rnd.uniform(-1.0, 1.0) for i in range(1, 13)}
    out = []
    for customer, t in taste.items():
        for _ in range(10):
            day = base + datetime.timedelta(days=rnd.randrange(0, 180))
            basket = round(rnd.uniform(5, 120), 2)
            items = rnd.randint(1, 12)
            score = 1.2 * t + (basket - 60) / 60 + rnd.gauss(0, 0.6)
            out.append((customer, day.isoformat(), basket, items, 1 if score > 0 else 0))
    rnd.shuffle(out)
    return out


def text():
    lines = ["customer,visit_date,basket,items,returning"]
    lines += ["%s,%s,%s,%d,%d" % (c, d, repr(b), i, r) for c, d, b, i, r in rows()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    target = HERE / "visits.csv"
    if "--check" in sys.argv:
        sys.exit(0 if target.read_text() == text() else 1)
    target.write_text(text())
