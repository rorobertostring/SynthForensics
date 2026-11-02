#!/usr/bin/env python3
"""
01_caption_coverage.py — Content coverage of the structured source captions.

Groups the free-text fields of the 1,363 human-validated structured captions
(one per source video) into categories with explicit, case-insensitive
keyword rules, defined below. Single-label fields (scene from `background`,
`shot_type`, `camera_movement`) take the first matching category in the
listed order, and values matching no rule are reported as 'other'. Actions
and the multi-subject flag are multi-label matches over `short_description`
and `dense_description`.

This analysis is not reported in the paper.

Run:
    python 01_caption_coverage.py --captions-dir /path/to/SynthForensics/captions

Output:
    caption_coverage.csv   (field, category, count, pct over all captions)
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# Canonicalization rules: (category, [keywords]); first match wins.
SHOT_RULES = [
    ("close-up", ["close-up", "close up", "closeup"]),
    ("medium", ["medium"]),
    ("wide/long", ["wide", "long shot", "long-shot", "establishing"]),
    ("full-body", ["full body", "full-body", "full shot", "cowboy"]),
    ("aerial", ["aerial", "drone", "bird's-eye", "birds-eye", "overhead"]),
]
CAMERA_RULES = [
    ("static", ["static", "fixed", "stationary", "no camera movement",
                "remains still", "no significant camera"]),
    ("pan", ["pan"]),
    ("tilt", ["tilt"]),
    ("zoom", ["zoom"]),
    ("tracking/dolly", ["tracking", "dolly", "follows", "following"]),
    ("handheld", ["handheld", "hand-held", "shaky", "shake"]),
]
SCENE_RULES = [
    ("news studio/broadcast", ["news studio", "newsroom", "news broadcast",
                               "studio", "broadcast", "anchor desk",
                               "green screen", "weather map"]),
    ("graphic/screen backdrop", ["graphic", "gradient", "abstract", "logo",
                                 "solid color", "solid red", "solid blue",
                                 "digital clock", "patterned screen",
                                 "horizontal lines", "colorful backdrop",
                                 "colorful design", "blurred screen"]),
    ("stage/auditorium", ["stage", "auditorium", "podium", "conference",
                          "lecture", "audience", "theater", "theatre",
                          "press", "panel"]),
    ("office/indoor", ["office", "meeting room", "classroom", "indoor",
                       "room", "hallway", "corridor", "interior", "wall",
                       "curtain", "bookshelf", "whiteboard", "desk"]),
    ("domestic/home", ["kitchen", "living room", "bedroom", "home", "house",
                       "apartment", "couch", "sofa", "domestic"]),
    ("outdoor urban", ["shutter", "corrugated", "street", "city", "urban",
                       "building", "sidewalk", "road", "park ", "square",
                       "outdoor", "outside", "car ", "parking"]),
    ("outdoor nature", ["nature", "forest", "beach", "mountain", "field",
                        "garden", "trees", "grass", "lake", "sea ", "sky"]),
    ("vehicle", ["inside a car", "vehicle interior", "in a car", "bus ",
                 "train ", "cockpit"]),
]
# Multi-label action lexicon over short + dense description. Word-boundary
# regexes: plain substring matching produced false positives ("playing"
# inside "displaying", "sing" inside "using", "walk" inside "sidewalk").
ACTION_LEXICON = [
    ("speaking/talking", r"\bspeak\w*|\btalk\w*|\bsays\b|\bsaying\b|\baddress\w*|\bdeliver\w*|\bnarrat\w*"),
    ("presenting/reporting", r"\bpresent\w*|\breport\w*|\banchor\w*|\bhost\w*|\bbroadcast\w*|\bannounc\w*"),
    ("interviewing", r"\binterview\w*"),
    ("sitting", r"\bsitting\b|\bseated\b|\bsits\b"),
    ("standing", r"\bstanding\b|\bstands\b"),
    ("walking/moving", r"\bwalk(s|ing)?\b|\bmoving\b|\bmoves\b|\bsteps\b|\bstroll\w*|\bapproach\w*|\benter(s|ing)?\b|\bexit(s|ing)?\b|\bleav(es|ing)\b"),
    ("gesturing", r"\bgestur\w*|hand movement|\bpointing\b|\bpoints\b|\bwaving\b|\bnodding\b|\bnods\b|\bshrug\w*"),
    ("smiling/laughing", r"\bsmil\w*|\blaugh\w*|\bgrin\w*"),
    ("hugging/contact", r"\bhug(s|ging)?\b|\bembrac\w*|\bhandshake\b|shaking hands|\bkiss\w*"),
    ("eating/drinking", r"\beating\b|\beats\b|\bdrink\w*|\bsipping\b|\bmeal\b"),
    ("cooking", r"\bcook(s|ing)?\b|preparing food|\bchopping\b"),
    ("reading/writing", r"\breading\b|\breads\b|\bwriting\b|\bwrites\b|\btyping\b|\bnotes\b|\bpaper\b"),
    ("phone/device use", r"\bphone\b|\bdevice\b|\blaptop\b|\bcomputer\b|\btablet\b|\bmicrophone\b"),
    ("performing/playing", r"\bperform\w*|\bplay(s|ing)?\b|\bdanc\w*|\bsing(s|ing|er)?\b|\bmusic\w*|\binstrument\w*|\bsports?\b|\bexercis\w*"),
]
ACTION_LEXICON = [(name, re.compile(pattern)) for name, pattern in ACTION_LEXICON]

MULTI_SUBJECT_RE = re.compile(
    r"two men|two women|two people|two persons|another person|another man|"
    r"another woman|second person|\bgroup\b|\bpeople\b|\bcrowd\b|\bseveral\b|"
    r"both of them|\bcouple\b|\bindividuals\b")


def canonicalize(value: str, rules) -> str:
    text = (value or "").lower()
    for category, keywords in rules:
        if any(k in text for k in keywords):
            return category
    return "other"


def parse_args():
    p = argparse.ArgumentParser(description="Content coverage of the structured captions")
    p.add_argument("--captions-dir", type=Path, required=True,
                   help="Folder with the 1,363 caption JSON files")
    p.add_argument("--output-csv", type=Path,
                   default=Path(__file__).parent / "caption_coverage.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.captions_dir.glob("*.json"))
    assert len(files) == 1363, f"expected 1363 captions, found {len(files)}"

    records = []
    for path in files:
        data = json.loads(path.read_text())
        desc = " ".join([data.get("short_description", ""),
                         data.get("dense_description", "")]).lower()
        rec = {
            "scene": canonicalize(data.get("background"), SCENE_RULES),
            "shot_type": canonicalize(data.get("shot_type"), SHOT_RULES),
            "camera_movement": canonicalize(data.get("camera_movement"), CAMERA_RULES),
        }
        for action, pattern in ACTION_LEXICON:
            rec[f"action:{action}"] = bool(pattern.search(desc))
        rec["multi_subject"] = bool(MULTI_SUBJECT_RE.search(desc))
        records.append(rec)

    df = pd.DataFrame(records)
    n = len(df)

    rows = []
    single_fields = ["scene", "shot_type", "camera_movement"]
    for field in single_fields:
        for category, count in Counter(df[field]).most_common():
            rows.append({"field": field, "category": category,
                         "count": count, "pct": 100 * count / n})
    for col in [c for c in df.columns if c.startswith("action:")] + ["multi_subject"]:
        field = "action" if col.startswith("action:") else "subjects"
        category = col.removeprefix("action:") if col != "multi_subject" else "multi-subject"
        count = int(df[col].sum())
        rows.append({"field": field, "category": category,
                     "count": count, "pct": 100 * count / n})

    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved {args.output_csv.name}: {len(out)} rows over {n} captions")
    print()

    for field in single_fields:
        sub = out[out["field"] == field]
        print(f"{field:16s} " + ", ".join(f"{r['category']} {r['pct']:.1f}%" for _, r in sub.iterrows()))
    print()
    sub = out[out["field"] == "action"].sort_values("pct", ascending=False)
    print("actions (multi-label): " +
          ", ".join(f"{r['category']} {r['pct']:.1f}%" for _, r in sub.iterrows()))
    ms = out[out["field"] == "subjects"]["pct"].iloc[0]
    print(f"multi-subject mentions: {ms:.1f}%")


if __name__ == "__main__":
    main()
