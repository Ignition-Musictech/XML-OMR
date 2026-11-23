from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path("/mnt/e/super_omr")
OUT = ROOT / "tests/20260808_dataset_final"
ROWS = OUT / "row_samples"
GRID_V4 = ROOT / "tests/20260805_crop_repair/grids_v4"
DATA = ROOT / "data"
HOMR_SRC = ROOT / "omr_methods/homr/homr_src"


def gt_path(piece: str) -> Path | None:
    for cand in (f"{piece}.mxl", f"{piece}_gd.musicxml",
                 f"{piece}.musicxml"):
        p = DATA / piece / cand
        if p.exists():
            return p
    return None


def load_homr_chain():
    try:
        sys.path.insert(0, str(HOMR_SRC))
        from homr.circle_of_fifths import strip_naturals
        from training.datasets.convert_lieder import (
            MeasureCutter, contains_only_supported_clefs, is_grandstaff)
        from training.datasets.music_xml_parser import (
            music_xml_file_to_tokens)
        from training.transformer.training_vocabulary import (
            calc_ratio_of_tuplets, token_lines_to_str)
        return {"strip_naturals": strip_naturals,
                "MeasureCutter": MeasureCutter,
                "supported_clefs": contains_only_supported_clefs,
                "is_grandstaff": is_grandstaff,
                "xml_to_tokens": music_xml_file_to_tokens,
                "tuplet_ratio": calc_ratio_of_tuplets,
                "to_str": token_lines_to_str}
    except Exception as e:
        print(f"[phase B unavailable] homr imports failed: {e}")
        return None


def gt_measure_list(gm) -> list[int] | None:
    if gm is None:
        return None
    if isinstance(gm, list):
        return [int(x) for x in gm]
    return [int(gm)]


def export_excerpt(sc, gex_pair, start: int, end: int,
                   out_xml: Path) -> str | None:
    import copy

    from music21 import clef as m21clef
    from music21 import key, meter

    gex, score_exporter_cls = gex_pair
    ex = sc.measures(start, end)
    try:
        out_xml.write_bytes(gex.parse(ex))
        return None
    except Exception:
        pass
    try:
        for pi, part in enumerate(ex.parts):
            mm = part.getElementsByClass("Measure")
            if not mm:
                continue
            first = mm[0]
            src = sc.parts[pi].getElementsByClass("Measure")[start - 1]
            for cls in (m21clef.Clef, meter.TimeSignature,
                        key.KeySignature):
                if not first.getElementsByClass(cls):
                    ctx = src.getContextByClass(cls)
                    if ctx is not None:
                        first.insert(0.0, copy.deepcopy(ctx))
        se = score_exporter_cls(ex)
        se.makeNotation = False
        out_xml.write_text(ET.tostring(se.parse(), encoding="unicode"),
                           encoding="utf-8")
        return None
    except Exception as e:
        return f"excerpt_export_fail: {str(e)[:120]}"


def main() -> None:
    from music21 import converter
    from music21.musicxml.m21ToXml import (GeneralObjectExporter,
                                           ScoreExporter)

    ROWS.mkdir(parents=True, exist_ok=True)
    homr = load_homr_chain()

    manifest = OUT / "dataset_final_manifest.jsonl"
    if not manifest.exists():
        manifest = OUT / "dataset_1480_manifest.jsonl"
    slots_by_row: dict = defaultdict(list)
    with open(manifest) as f:
        for line in f:
            r = json.loads(line)
            slots_by_row[(r["piece"], r["page"], r["sys"])].append(r)

    grids: dict = {}
    for piece_dir in sorted(GRID_V4.iterdir()):
        if not piece_dir.is_dir() or piece_dir.name == "_audit":
            continue
        for gp in sorted(piece_dir.glob("page-*.grid.json")):
            g = json.loads(gp.read_text())
            grids[(piece_dir.name, g["page"])] = g
            g["_img"] = None

    scores: dict = {}
    gex = GeneralObjectExporter()
    index: list[dict] = []
    n_ok = n_row_skip = n_tok = n_tok_skip = 0

    for (piece, page, sysi), slots in sorted(slots_by_row.items()):
        slots.sort(key=lambda s: s["idx_in_sys"])
        rec = {"piece": piece, "page": page, "sys": sysi,
               "n_slots": len(slots)}
        g = grids.get((piece, page))
        expected = (len(g["systems"][sysi]["measures"])
                    if g and sysi < len(g["systems"]) else None)
        rec["expected_slots"] = expected

        gms: list[int] = []
        ok = expected is not None and len(slots) == expected and \
            [s["idx_in_sys"] for s in slots] == list(range(expected))
        if not ok:
            rec["status"] = "skip_incomplete_row"
            index.append(rec)
            n_row_skip += 1
            continue
        for s in slots:
            ms = gt_measure_list(s["gt_m"])
            if ms is None:
                ok = False
                break
            gms.extend(ms)
        if not ok or gms != list(range(gms[0], gms[0] + len(gms))):
            rec["status"] = "skip_gt_not_contiguous"
            index.append(rec)
            n_row_skip += 1
            continue
        rec["gt_start"], rec["gt_end"] = gms[0], gms[-1]

        stem = f"{piece}_p{page}_sys{sysi}"
        png_out = ROWS / f"{stem}.png"
        xml_out = ROWS / f"{stem}.musicxml"
        tok_out = ROWS / f"{stem}.tokens"

        if not png_out.exists():
            if g["_img"] is None:
                g["_img"] = cv2.imread(g["png"], cv2.IMREAD_GRAYSCALE)
            x0 = min(s["x0"] for s in slots)
            x1 = max(s["x1"] for s in slots)
            y0 = min(s["y0"] for s in slots)
            y1 = max(s["y1"] for s in slots)
            crop = g["_img"][max(0, y0):y1, max(0, x0):x1]
            if crop.size == 0:
                rec["status"] = "skip_empty_crop"
                index.append(rec)
                n_row_skip += 1
                continue
            cv2.imwrite(str(png_out), crop)

        if not xml_out.exists():
            if piece not in scores:
                sc = converter.parse(str(gt_path(piece)))
                for part in sc.parts:
                    for i, m in enumerate(
                            part.getElementsByClass("Measure")):
                        m.number = i + 1
                scores[piece] = sc
            err = export_excerpt(scores[piece], (gex, ScoreExporter),
                                 gms[0], gms[-1], xml_out)
            if err:
                rec["status"] = err
                index.append(rec)
                n_row_skip += 1
                continue

        rec["status"] = "ok"
        rec["png"] = str(png_out)
        rec["musicxml"] = str(xml_out)
        n_ok += 1

        if homr and not tok_out.exists():
            try:
                voices = homr["xml_to_tokens"](str(xml_out))
                if len(voices) != 1 or not voices[0]:
                    raise ValueError(f"voices={len(voices)}")
                voice = voices[0]
                if not homr["is_grandstaff"](voice):
                    raise ValueError("not_grandstaff")
                cutter = homr["MeasureCutter"](voice)
                symbols = cutter.extract_measures(len(voice))
                rec["tuplet_ratio"] = round(
                    homr["tuplet_ratio"](symbols), 4)
                if not homr["supported_clefs"](symbols):
                    raise ValueError("unsupported_clefs")
                symbols = homr["strip_naturals"](symbols)
                tok_out.write_text(homr["to_str"](symbols),
                                   encoding="utf-8")
                rec["tokens"] = str(tok_out)
                n_tok += 1
            except Exception as e:
                rec["homr_tokens"] = f"skip: {str(e)[:100]}"
                n_tok_skip += 1
        elif homr and tok_out.exists():
            rec["tokens"] = str(tok_out)
            n_tok += 1
        index.append(rec)

    with open(OUT / "row_index.jsonl", "w") as f:
        for r in index:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"rows ok {n_ok}, skipped {n_row_skip}, "
          f"homr tokens {n_tok} (token-skip {n_tok_skip})")
    by_piece: dict = defaultdict(int)
    for r in index:
        if r["status"] == "ok":
            by_piece[r["piece"]] += 1
    for p in sorted(by_piece):
        print(f"  {p}: {by_piece[p]} rows")


if __name__ == "__main__":
    main()
