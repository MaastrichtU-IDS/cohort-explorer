#!/usr/bin/env python3
"""Parse icd102019en.xml and produce a compact JSON lookup for the frontend."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

XML_PATH = Path.home() / "Downloads" / "icd102019en.xml"
OUT_PATH = Path(__file__).parent.parent / "frontend" / "public" / "icd10.json"


def main():
    print(f"Parsing {XML_PATH} ...")
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    # Build child → parent map from SubClass relationships
    parent_map: dict[str, str] = {}
    for cls in root.findall("Class"):
        code = cls.get("code", "")
        for subclass in cls.findall("SubClass"):
            child_code = subclass.get("code", "")
            if child_code:
                parent_map[child_code] = code

    entries = []
    for cls in root.findall("Class"):
        code = cls.get("code", "")
        kind = cls.get("kind", "")
        if kind not in ("chapter", "block", "category"):
            continue
        label = ""
        for rubric in cls.findall("Rubric"):
            if rubric.get("kind") == "preferred":
                label_el = rubric.find("Label")
                if label_el is not None:
                    label = "".join(label_el.itertext()).strip()
                break
        if code and label:
            entries.append({"code": code, "label": label, "kind": kind, "parent": parent_map.get(code)})

    entries.sort(key=lambda x: x["code"])
    print(f"Found {len(entries)} category entries.")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, separators=(",", ":"))
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Written to {OUT_PATH} ({size_kb:.1f} KB)")


def augment_with_parents(target_codes: list[str]) -> list[dict]:
    """Given a list of ICD-10 codes, return those codes plus all ancestor
    blocks/chapters, ordered hierarchically (chapters → blocks → categories).

    Reads from the already-generated icd10.json so no XML is needed.
    Codes not found in the data produce a warning and are skipped.
    """
    with open(OUT_PATH, encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    lookup: dict[str, dict] = {e["code"]: e for e in data}

    needed: set[str] = set()
    for code in target_codes:
        code = code.strip()
        if not code:
            continue
        if code not in lookup:
            print(f"  WARNING: code not found in icd10.json, skipping: {code!r}", file=sys.stderr)
            continue
        needed.add(code)
        parent = lookup[code].get("parent")
        while parent:
            needed.add(parent)
            parent = lookup.get(parent, {}).get("parent")

    target_set = set(t.strip() for t in target_codes if t.strip() in lookup)

    # If a target is a chapter, expand its direct block children into the target set
    chapter_targets = [c for c in target_set if lookup[c]["kind"] == "chapter"]
    for chapter_code in chapter_targets:
        for entry in data:
            if entry.get("parent") == chapter_code and entry["kind"] == "block":
                needed.add(entry["code"])
                target_set.add(entry["code"])

    # Build children map (restricted to entries in `needed`)
    children: dict[str, list[str]] = {}
    for code in needed:
        p = lookup[code].get("parent")
        if p and p in needed:
            children.setdefault(p, []).append(code)
    for lst in children.values():
        lst.sort()

    # Collect root entries (no parent inside the needed set)
    roots = sorted(c for c in needed if (lookup[c].get("parent") not in needed))

    # Tree traversal to produce hierarchy-ordered output
    result: list[dict] = []

    def traverse(code: str) -> None:
        entry = dict(lookup[code])
        entry["is_target"] = code in target_set
        result.append(entry)
        for child in children.get(code, []):
            traverse(child)

    for r in roots:
        traverse(r)

    return result


def generate_parents_map() -> None:
    """Starting from the blocks/chapters in the target list, find ALL their descendant
    category codes in the full icd10.json, then write a parents map for each.

    Format per line:
        CODE: parent1, parent2, ...  (immediate parent first, chapter last)

    Output file: frontend/public/icd10-parents-map.txt
    """
    targeted_path = OUT_PATH.parent / "icd10-targeted.json"
    with open(targeted_path, encoding="utf-8") as f:
        targeted: list[dict] = json.load(f)

    with open(OUT_PATH, encoding="utf-8") as f:
        full_data: list[dict] = json.load(f)

    lookup: dict[str, dict] = {e["code"]: e for e in full_data}

    # Collect the block and chapter codes that were in the original target set
    target_blocks_chapters = {
        e["code"] for e in targeted
        if e.get("is_target") and e.get("kind") in ("block", "chapter")
    }
    print(f"Target blocks/chapters: {sorted(target_blocks_chapters)}")

    # Build a parent → children map over the full data
    children_map: dict[str, list[str]] = {}
    for entry in full_data:
        p = entry.get("parent")
        if p:
            children_map.setdefault(p, []).append(entry["code"])

    # Find all descendant categories of a given code
    def get_all_category_descendants(root: str) -> list[str]:
        result = []
        stack = [root]
        while stack:
            code = stack.pop()
            for child in children_map.get(code, []):
                entry = lookup.get(child, {})
                if entry.get("kind") == "category":
                    result.append(child)
                else:
                    stack.append(child)
        return sorted(result)

    # Gather all category descendants of target blocks/chapters
    category_codes: set[str] = set()
    for code in target_blocks_chapters:
        category_codes.update(get_all_category_descendants(code))
    print(f"Expanded to {len(category_codes)} category codes")

    # Build parents map
    out_path = OUT_PATH.parent / "icd10-parents-map.txt"
    lines: list[str] = []
    for code in sorted(category_codes):
        ancestors: list[str] = []
        parent = lookup.get(code, {}).get("parent")
        while parent:
            ancestors.append(parent)
            parent = lookup.get(parent, {}).get("parent")
        if ancestors:
            lines.append(f"{code}: {', '.join(ancestors)}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written {len(lines)} entries to {out_path}")


def augment_main():
    target_codes = [
        "E10-E14", "E10", "E11", "O24.4", "E66",
        "IX",
        "I10", "I11", "I20", "I21", "I25", "I42", "I50", "I63", "I64", "I48", "I70",
        "O90.3", "O14", "O15",
        "J44", "J45", "J80",
        "C50", "C61", "C92", "C91",
        "N17", "N18",
    ]

    print("Augmenting with parents...")
    entries = augment_with_parents(target_codes)

    print(f"\n{'CODE':<14} {'KIND':<10} {'TARGET':<8} LABEL")
    print("-" * 80)
    for e in entries:
        marker = "  ✓" if e["is_target"] else ""
        print(f"  {e['code']:<12} {e['kind']:<10} {marker:<8} {e['label']}")

    out_path = OUT_PATH.parent / "icd10-targeted.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\nWritten {len(entries)} entries to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "augment":
        augment_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "parents-map":
        generate_parents_map()
    else:
        main()
