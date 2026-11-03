#!/usr/bin/env python3
"""Regenerate the kniv v7 golden fixtures from the Python reference runtime.

The Rust provider in `provider/local_onnx/nlp_v7` is a port of
`kniv_v7/runtime.py`; every decoding step there is part of the model's
behaviour, so the only way to know the port is faithful is to compare against
the reference on fixed inputs. This script produces those comparisons.

Usage:

    python scripts/gen-kniv-v7-goldens.py \
        --model /path/to/kniv-v7-xlarge-u-en \
        --runtime /path/to/kniv-nlp-models \
        --variant int8 \
        --out crates/uni-xervo/tests/fixtures/kniv-v7/ultrabert-int8

One file per case is written into `--out`, whose basename is the tag the
Rust test selects with `KNIV_V7_FIXTURE_TAG`. Each tag pins one encoder at
one precision: quantisation moves scores, so an INT8 run is compared against
INT8 goldens, never against fp32 ones.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The cases the Rust golden test replays. Prose cases carry `text`;
# conversation cases carry `turns`.
CASES: list[dict] = [
    {
        "name": "quickstart-conversation",
        "turns": [
            ["user", "I moved to Lisbon last year and started at a bank."],
            ["assistant", "Congratulations on the move!"],
        ],
        "schema": "memory",
    },
    {
        "name": "lovelace-prose",
        "text": "Ada Lovelace was born in London in 1815. She later worked with Charles Babbage on the Analytical Engine.",
        "schema": "memory",
    },
    {
        "name": "no-schema-prose",
        "text": "Marie Curie discovered radium in Paris. She won two Nobel Prizes.",
        "schema": None,
    },
    {
        "name": "negation-and-plans",
        "turns": [
            ["user", "I don't work at Acme anymore. Next month I'm joining Northstar."],
            ["assistant", "That's a big change."],
        ],
        "schema": "memory",
    },
    {
        "name": "clitics-and-urls",
        "text": "She didn't visit www.example.com. It's 3.14 and 50% of the 2nd batch... really?",
        "schema": None,
    },
    {
        "name": "two-paragraphs",
        "text": "Acme Robotics is based in Porto.\n\nThe company hired 200 engineers in 2024.",
        "schema": "memory",
    },
]

# Appendix D of the adoption guide: the schema the memory probe used.
MEMORY_RELATIONS = [
    ("residence", "the place where the person is or has been, resident"),
    ("employer", "person or organization for which the subject works or worked"),
    ("occupation", "occupation of a person"),
    ("owns", "The head holds ownership rights over the tail asset."),
    ("likes", "The head has a favorable attitude toward or takes pleasure in the tail"),
    ("dislikes", "The head feels aversion or an unfavorable attitude toward the tail."),
    ("prefers", "The head favors the tail option or characteristic over relevant alternatives."),
    ("interested in", "The head has curiosity, attention, or an inclination toward the tail subject"),
    ("friend of", "The head and tail have a personal relationship of friendship."),
    ("spouse", "the subject has the object as their spouse (husband, wife, partner, etc.)"),
    ("sibling", "the subject and the object have at least one common parent (brother, sister)"),
    ("member of", "organization, club or musical group to which the subject belongs"),
    ("educated at", "educational institution attended by subject"),
    ("uses", "The head employs the tail tool, resource, method, or component"),
    ("has goal", "The head pursues or is intended to achieve the tail objective or outcome"),
    ("recommends", "The head advises choosing, using, or undertaking the tail."),
    ("scheduled for", "The head event or activity is planned to take place at the tail date or time"),
    ("occurred at", "The head event happened at the tail specified date or time."),
    ("has price", "The head product, service, or booking is offered or charged at the tail amount"),
    ("has count", "The head collection or group contains the tail number of members or items"),
    ("participant in", "event in which a person, organization or creative work was/is a participant"),
    ("located in", "The head is physically situated within the tail place or container."),
    ("created by", "The head work, artifact, or entity was brought into existence by the tail"),
    ("visited", "the head person went to or stayed at the tail place"),
    ("purchased", "the head person bought or acquired the tail item"),
    ("has pet", "the head person owns or keeps the tail animal as a pet"),
    ("plans to", "the head person intends to do or obtain the tail in the future"),
    ("attended", "the head person was present at the tail event, class or appointment"),
    ("had problem with", "the head person experienced an issue or fault with the tail thing"),
    ("completed", "the head person finished the tail task, course, book or activity"),
]
MEMORY_ENTITY_TYPES = [
    "person", "organization", "location", "product", "event", "date", "time",
    "number", "money", "animal", "food", "activity", "vehicle", "device",
    "work of art",
]

SCHEMAS = {
    "memory": {
        "relations": [{"name": n, "description": d} for n, d in MEMORY_RELATIONS],
        "entity_types": [{"name": n} for n in MEMORY_ENTITY_TYPES],
    }
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a kniv v7 release directory")
    ap.add_argument("--runtime", required=True, help="the repo holding kniv_v7/")
    ap.add_argument("--variant", default="int8", help='"", "fp16" or "int8" (shorthand for int8,fp16)')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.runtime)
    from kniv_v7 import Kniv  # noqa: E402

    variant = {"int8": "int8,fp16", "": None}.get(args.variant, args.variant)
    kniv = Kniv(args.model, variant=variant)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.variant or "fp32"

    for case in CASES:
        schema = SCHEMAS.get(case.get("schema") or "", {})
        windows = kniv.analyse(
            text=case.get("text"),
            turns=[tuple(t) for t in case["turns"]] if "turns" in case else None,
            relations=schema.get("relations", []),
            entity_types=schema.get("entity_types"),
        )
        payload = {
            "case": case["name"],
            "variant": tag,
            "run": kniv.meta["run"],
            "input": {k: case[k] for k in ("text", "turns") if k in case},
            "schema": case.get("schema"),
            "windows": windows,
        }
        path = out_dir / f"{case['name']}.json"
        path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
