from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import pypdfium2 as pdfium

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def inspect_pdf(pdf_path: Path) -> dict:
    pdf = pdfium.PdfDocument(str(pdf_path))
    n = len(pdf)
    counts = {"text": 0, "path": 0, "image": 0}
    for i in range(min(n, 3)):
        for o in pdf[i].get_objects():
            if   o.type == 1: counts["text"]  += 1
            elif o.type == 2: counts["path"]  += 1
            elif o.type == 3: counts["image"] += 1
    kind = "scanned" if counts["image"] >= 1 and counts["path"] == 0 else "vector"
    w_pt, h_pt = pdf[0].get_size()
    return {
        "pages":      n,
        "kind":       kind,
        "width_in":   round(w_pt / 72.0, 2),
        "height_in":  round(h_pt / 72.0, 2),
        "obj_counts": counts,
    }


def render_pdf(pdf_path: Path, dpi: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("page-*.png"):
        old.unlink()
    pdf = pdfium.PdfDocument(str(pdf_path))
    paths = []
    for i in range(len(pdf)):
        bmp = pdf[i].render(scale=dpi / 72.0, grayscale=True)
        out = out_dir / f"page-{i + 1:02d}.png"
        bmp.to_pil().save(out, optimize=True)
        paths.append(out)
    return paths


def run_homr_on_image(image_path: Path) -> Path | None:
    from homr.main import (
        ProcessingConfig,
        XmlGeneratorArguments,
        process_image,
        download_weights,
    )
    import onnxruntime as ort

    has_gpu = "CUDAExecutionProvider" in ort.get_available_providers()

    download_weights(has_gpu)

    config = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        use_gpu_inference=has_gpu,
    )
    xml_args = XmlGeneratorArguments(large_page=False, metronome=None, tempo=None)

    expected_xml = image_path.with_suffix(".musicxml")
    try:
        process_image(str(image_path), config, xml_args)
    except Exception as e:
        print(f"    HOMR error on {image_path.name}: {e}")
        return None

    return expected_xml if expected_xml.exists() else None


def _detect_ns(root: ET.Element) -> str:
    if root.tag.startswith("{"):
        return root.tag.split("}")[0] + "}"
    return ""


def merge_page_xmls(page_xmls: list[Path], out_path: Path) -> tuple[int, int]:
    trees = [ET.parse(p) for p in page_xmls]
    roots = [t.getroot() for t in trees]
    ns = _detect_ns(roots[0])

    base_root = roots[0]
    base_part = base_root.find(f"{ns}part")
    if base_part is None:
        parts = base_root.findall(f".//{ns}part")
        if not parts:
            raise RuntimeError("First page MusicXML has no <part>")
        base_part = parts[0]

    for root in roots[1:]:
        part = root.find(f"{ns}part")
        if part is None:
            part_list = root.findall(f".//{ns}part")
            part = part_list[0] if part_list else None
        if part is None:
            continue
        for m in part.findall(f"{ns}measure"):
            base_part.append(m)

    for i, m in enumerate(base_part.findall(f"{ns}measure"), 1):
        m.set("number", str(i))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(base_root).write(
        str(out_path), encoding="utf-8", xml_declaration=True
    )
    return _score_stats(base_root, ns)


def _score_stats(root: ET.Element, ns: str = "") -> tuple[int, int]:
    measures = 0
    notes = 0
    for p in root.findall(f".//{ns}part"):
        ms = p.findall(f"{ns}measure")
        measures = max(measures, len(ms))
        for m in ms:
            for n in m.findall(f".//{ns}note"):
                if (n.find(f"{ns}pitch") is not None
                        or n.find(f"{ns}unpitched") is not None):
                    notes += 1
    return measures, notes


def _detail_stats(root: ET.Element, ns: str = "") -> dict:
    per_staff: dict[str, int] = {}
    grace_count = 0
    for n in root.findall(f".//{ns}note"):
        if n.find(f"{ns}pitch") is None and n.find(f"{ns}unpitched") is None:
            continue
        s = n.findtext(f"{ns}staff") or "1"
        per_staff[s] = per_staff.get(s, 0) + 1
        if n.find(f"{ns}grace") is not None:
            grace_count += 1

    ts = []
    for t in root.findall(f".//{ns}time"):
        beats = t.findtext(f"{ns}beats") or "?"
        bt = t.findtext(f"{ns}beat-type") or "?"
        ts.append(f"{beats}/{bt}")

    ks = [k.findtext(f"{ns}fifths") or "?" for k in root.findall(f".//{ns}key")]
    return {
        "per_staff": per_staff,
        "grace_notes": grace_count,
        "time_sigs": ts[:5],
        "key_sigs": ks[:5],
    }


def process_pdf(pdf: Path, dpi: int, output_dir: Path | None = None) -> dict:
    print(f"\n{'=' * 70}\n>>> {pdf.name}  (HOMR)\n{'=' * 70}")

    info = inspect_pdf(pdf)
    print(f"[inspect] pages={info['pages']}  kind={info['kind']}  "
          f"size={info['width_in']}x{info['height_in']} in  "
          f"objs={info['obj_counts']}")

    if output_dir is None:
        output_dir = pdf.parent
    work_dir = output_dir / "homr_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    pages = render_pdf(pdf, dpi, work_dir)
    print(f"[render]  {len(pages)} page(s) at {dpi} DPI")

    t0 = time.time()
    page_xmls: list[Path] = []
    failed: list[str] = []
    for png in pages:
        print(f"[homr]    {png.name} ...", end=" ", flush=True)
        xml = run_homr_on_image(png)
        if xml is not None:
            print(f"OK")
            page_xmls.append(xml)
        else:
            print(f"FAILED")
            failed.append(png.name)
    dt = time.time() - t0

    if not page_xmls:
        return {"pdf": pdf, "status": "ALL_PAGES_FAILED", "out": None, "info": info}

    stem = pdf.stem
    out_xml = output_dir / f"{stem}_homr.musicxml"

    if len(page_xmls) == 1:
        import shutil
        shutil.copy2(page_xmls[0], out_xml)
        root = ET.parse(out_xml).getroot()
        ns = _detect_ns(root)
        measures, notes = _score_stats(root, ns)
        mode = "single-page"
    else:
        measures, notes = merge_page_xmls(page_xmls, out_xml)
        root = ET.parse(out_xml).getroot()
        ns = _detect_ns(root)
        mode = f"merged ({len(page_xmls)}/{len(pages)} pages)"

    print(f"[merge]   {measures} measures, {notes} notes in {dt:.1f}s  [{mode}]")

    details = _detail_stats(root, ns)

    return {
        "pdf":         pdf,
        "out":         out_xml,
        "status":      "OK" if not failed else "PARTIAL",
        "mode":        mode,
        "kind":        info["kind"],
        "pages":       info["pages"],
        "measures":    measures,
        "notes":       notes,
        "grace_notes": details["grace_notes"],
        "per_staff":   details["per_staff"],
        "time_sigs":   details["time_sigs"],
        "key_sigs":    details["key_sigs"],
        "elapsed_s":   round(dt, 1),
        "failed_pages": failed,
    }


def process_images(images: list[Path], output_dir: Path) -> dict:
    print(f"\n{'=' * 70}\n>>> {len(images)} image(s)  (HOMR)\n{'=' * 70}")

    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    page_xmls: list[Path] = []
    failed: list[str] = []
    for img in images:
        print(f"[homr]    {img.name} ...", end=" ", flush=True)
        xml = run_homr_on_image(img)
        if xml is not None:
            print(f"OK")
            page_xmls.append(xml)
        else:
            print(f"FAILED")
            failed.append(img.name)
    dt = time.time() - t0

    if not page_xmls:
        return {"pdf": None, "status": "ALL_PAGES_FAILED", "out": None}

    stem = images[0].stem.rsplit("-", 1)[0] if len(images) > 1 else images[0].stem
    out_xml = output_dir / f"{stem}_homr.musicxml"

    if len(page_xmls) == 1:
        import shutil
        shutil.copy2(page_xmls[0], out_xml)
        root = ET.parse(out_xml).getroot()
        ns = _detect_ns(root)
        measures, notes = _score_stats(root, ns)
    else:
        measures, notes = merge_page_xmls(page_xmls, out_xml)
        root = ET.parse(out_xml).getroot()
        ns = _detect_ns(root)

    details = _detail_stats(root, ns)
    print(f"[done]    {measures} measures, {notes} notes in {dt:.1f}s")

    return {
        "pdf":         None,
        "out":         out_xml,
        "status":      "OK" if not failed else "PARTIAL",
        "mode":        f"images ({len(page_xmls)}/{len(images)})",
        "pages":       len(images),
        "measures":    measures,
        "notes":       notes,
        "grace_notes": details["grace_notes"],
        "per_staff":   details["per_staff"],
        "time_sigs":   details["time_sigs"],
        "key_sigs":    details["key_sigs"],
        "elapsed_s":   round(dt, 1),
        "failed_pages": failed,
    }


def write_report(r: dict) -> None:
    if r.get("out") is None:
        src = r.get("pdf") or "images"
        print(f"\n[FAIL] {src}: {r['status']}")
        return

    rep_path = r["out"].with_suffix(".report.txt")
    lines = [
        f"Source           : {r.get('pdf', 'direct images')}",
        f"Engine           : HOMR (transformer-based)",
        f"Status           : {r['status']}",
        f"Mode             : {r['mode']}",
        f"Pages            : {r.get('pages', '?')}",
        f"Measures         : {r['measures']}",
        f"Pitched notes    : {r['notes']}",
        f"Grace notes      : {r.get('grace_notes', 0)}",
        f"Notes per staff  : {r.get('per_staff', {})}   (1=top, 2=bottom)",
        f"Time signatures  : {r.get('time_sigs', [])}",
        f"Key sig (fifths) : {r.get('key_sigs', [])}",
        f"Elapsed          : {r['elapsed_s']} s",
        f"MusicXML         : {r['out'].name}",
    ]
    if r.get("failed_pages"):
        lines.append(f"Pages FAILED     : {r['failed_pages']}")
    rep_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[done] -> {r['out'].name}  "
          f"({r['measures']} measures / {r['notes']} notes / {r['elapsed_s']}s)")


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _autocrop_rect(img) -> tuple[int, int, int, int]:
    import cv2
    import numpy as np
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    dominant = int(np.argmax(hist.flatten()))
    thresh = cv2.threshold(gray, dominant - 30, 255, cv2.THRESH_BINARY)[1]
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, np.ones((9, 9), np.uint8))
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    full = (0, 0, img.shape[1], img.shape[0])
    best, area_best = None, 0.0
    for c in contours:
        a = cv2.contourArea(c)
        if a > area_best:
            area_best, best = a, c
    if best is None:
        return full
    x, y, w, h = cv2.boundingRect(best)
    if x < img.shape[1] * 0.25 or y < img.shape[0] * 0.25:
        return full
    return (int(x), int(y), int(w), int(h))


def _rbox(b) -> dict:
    return {"center": [round(float(b.center[0]), 2), round(float(b.center[1]), 2)],
            "size": [round(float(b.size[0]), 2), round(float(b.size[1]), 2)],
            "angle": round(float(b.angle), 2)}


def _bbox(b) -> dict:
    return {"box": [int(v) for v in b.box],
            "center": [round(float(b.center[0]), 2), round(float(b.center[1]), 2)]}


def export_geometry(image_path: Path, out_json: Path,
                    staff_sample_px: int = 20) -> dict:
    import json
    import cv2
    import numpy as np
    from homr.autocrop import autocrop
    from homr.bar_line_detection import detect_bar_lines, prepare_bar_line_image
    from homr.bounding_boxes import create_rotated_bounding_boxes
    from homr.brace_dot_detection import (
        find_braces_brackets_and_grand_staff_lines, prepare_brace_dot_image)
    from homr.main import (ProcessingConfig, download_weights,
                           load_and_preprocess_predictions, predict_symbols)
    from homr.model import Clef, Note
    from homr.note_detection import add_notes_to_staffs, combine_noteheads_with_stems
    from homr.staff_detection import break_wide_fragments, detect_staff
    import onnxruntime as ort

    result: dict = {"error": None}
    try:
        has_gpu = "CUDAExecutionProvider" in ort.get_available_providers()
        download_weights(has_gpu)
        config = ProcessingConfig(enable_debug=False, enable_cache=False,
                                  write_staff_positions=False,
                                  read_staff_positions=False,
                                  selected_staff=-1, use_gpu_inference=has_gpu)

        raw = cv2.imread(str(image_path))
        if raw is None:
            result["error"] = f"cannot read {image_path}"
            return result
        cx, cy, cw, ch = _autocrop_rect(raw)
        cropped = autocrop(raw)
        if cropped.shape[:2] != (ch, cw):
            result["error"] = (f"autocrop rect mismatch: rect {(cw, ch)} vs "
                               f"array {cropped.shape[1::-1]}")
            return result
        scale = cw / 1920.0

        predictions, debug = load_and_preprocess_predictions(
            str(image_path), config.enable_debug, config.enable_cache,
            config.use_gpu_inference)
        symbols = predict_symbols(debug, predictions)
        symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)
        noteheads_with_stems = combine_noteheads_with_stems(
            symbols.noteheads, symbols.stems_rest)
        if len(noteheads_with_stems) == 0:
            result["error"] = "no noteheads found"
            return result
        unit = float(np.median([n.notehead.size[1] for n in noteheads_with_stems]))
        all_noteheads = [n.notehead for n in noteheads_with_stems]
        all_stems = [n.stem for n in noteheads_with_stems if n.stem is not None]
        bar_candidates = [line for line in symbols.bar_lines
                          if not line.is_overlapping_with_any(all_noteheads)
                          and not line.is_overlapping_with_any(all_stems)]
        bar_line_boxes = detect_bar_lines(bar_candidates, unit)
        staffs = detect_staff(debug, predictions.staff, symbols.staff_fragments,
                              symbols.clefs_keys, bar_line_boxes)
        if len(staffs) == 0:
            result["error"] = "no staffs found"
            return result
        brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
        brace_dot = create_rotated_bounding_boxes(brace_dot_img, skip_merging=True,
                                                  max_size=(100, -1))
        add_notes_to_staffs(staffs, noteheads_with_stems, predictions.symbols,
                            predictions.notehead)
        multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)

        def staff_dict(st) -> dict:
            pts = [p for i, p in enumerate(st.grid)
                   if i % max(1, staff_sample_px) == 0 or i == len(st.grid) - 1]
            return {
                "min_x": round(float(st.min_x), 2), "max_x": round(float(st.max_x), 2),
                "min_y": round(float(st.min_y), 2), "max_y": round(float(st.max_y), 2),
                "unit_size": round(float(st.average_unit_size), 3),
                "is_grandstaff": bool(st.is_grandstaff),
                "n_lines": len(st.grid[0].y) if st.grid else 0,
                "track": [{"x": round(float(p.x), 1),
                           "y": [round(float(v), 1) for v in p.y]} for p in pts],
                "notes": [{"center": [round(float(n.center[0]), 2),
                                      round(float(n.center[1]), 2)],
                           "size": [round(float(n.box.size[0]), 2),
                                    round(float(n.box.size[1]), 2)],
                           "position": int(n.position),
                           "stem": _rbox(n.stem) if n.stem is not None else None,
                           "stem_direction": (str(n.stem_direction.name)
                                              if n.stem_direction is not None else None)}
                          for n in st.symbols if isinstance(n, Note)],
                "clefs": [_bbox(c.box) for c in st.symbols if isinstance(c, Clef)],
                "bar_lines": [_rbox(b) for b in bar_line_boxes
                              if st.min_y - 0.5 * unit <= b.center[1] <= st.max_y + 0.5 * unit
                              or (b.top_left[1] <= st.min_y and b.bottom_left[1] >= st.max_y)],
            }

        result.update({
            "image": str(image_path),
            "image_size": [int(raw.shape[1]), int(raw.shape[0])],
            "transform": {"crop": [cx, cy, cw, ch], "work_width": 1920,
                          "work_size": [int(predictions.original.shape[1]),
                                        int(predictions.original.shape[0])],
                          "scale": round(scale, 6)},
            "unit_size": round(unit, 3),
            "systems": [{"staffs": [staff_dict(s) for s in ms.staffs],
                         "connections": [_rbox(c) for c in ms.connections]}
                        for ms in multi_staffs],
            "page_bar_lines": [_rbox(b) for b in bar_line_boxes],
            "page_noteheads": [{"center": [round(float(n.notehead.center[0]), 2),
                                           round(float(n.notehead.center[1]), 2)],
                                "size": [round(float(n.notehead.size[0]), 2),
                                         round(float(n.notehead.size[1]), 2)],
                                "stem": _rbox(n.stem) if n.stem is not None else None}
                               for n in noteheads_with_stems],
            "page_clefs_keys": [_rbox(b) for b in symbols.clefs_keys],
            "page_stems_rests": [_rbox(b) for b in symbols.stems_rest],
            "page_brace_dot": [_rbox(b) for b in brace_dot],
        })
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        try:
            debug.clean_debug_files_from_previous_runs()
        except Exception:
            pass
        return result
    except Exception as e:
        result["error"] = f"geometry export failed: {type(e).__name__}: {e}"
        return result


def expand_inputs(args: list[str]) -> tuple[list[Path], list[Path]]:
    pdfs: list[Path] = []
    imgs: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            pdfs.extend(sorted(p.glob("*.pdf")))
            for ext in IMAGE_EXTS:
                imgs.extend(sorted(p.glob(f"*{ext}")))
        elif p.is_file():
            if p.suffix.lower() == ".pdf":
                pdfs.append(p)
            elif p.suffix.lower() in IMAGE_EXTS:
                imgs.append(p)
            else:
                print(f"WARNING: skipping {a} (unsupported format)")
        else:
            print(f"WARNING: skipping {a} (not found)")
    return pdfs, imgs


def main():
    ap = argparse.ArgumentParser(
        description="PDF/Image -> MusicXML via HOMR (transformer-based OMR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+",
                    help="PDF files, image files, or directories")
    ap.add_argument("--dpi", type=int, default=300,
                    help="PDF render DPI (default 300)")
    ap.add_argument("--output-dir", "-o", default=None,
                    help="Output directory (default: next to input)")
    ap.add_argument("--export-geometry", default=None, metavar="OUT.json",
                    help="write page geometry to OUT.json")
    args = ap.parse_args()

    if args.export_geometry:
        if len(args.inputs) != 1:
            sys.exit("--export-geometry takes exactly one page image")
        r = export_geometry(Path(args.inputs[0]).resolve(),
                            Path(args.export_geometry))
        if r["error"]:
            sys.exit(f"ERROR: {r['error']}")
        n_sys = len(r["systems"])
        print(f"[geometry] systems: {n_sys}  staffs: "
              f"{[len(s['staffs']) for s in r['systems']]}  "
              f"bar lines: {len(r['page_bar_lines'])}  "
              f"noteheads: {len(r['page_noteheads'])}  -> {args.export_geometry}")
        return

    pdfs, imgs = expand_inputs(args.inputs)
    if not pdfs and not imgs:
        sys.exit("No PDFs or images to process.")

    summary: list[dict] = []

    for pdf in pdfs:
        out_dir = Path(args.output_dir) if args.output_dir else pdf.parent
        try:
            r = process_pdf(pdf.resolve(), args.dpi, out_dir)
        except Exception as e:
            r = {"pdf": pdf, "out": None, "status": f"EXCEPTION: {e}"}
        write_report(r)
        summary.append(r)

    if imgs:
        out_dir = Path(args.output_dir) if args.output_dir else imgs[0].parent
        try:
            r = process_images([i.resolve() for i in imgs], out_dir)
        except Exception as e:
            r = {"pdf": None, "out": None, "status": f"EXCEPTION: {e}"}
        write_report(r)
        summary.append(r)

    print(f"\n\n{'=' * 70}\nSUMMARY ({len(summary)} job(s))\n{'=' * 70}")
    print(f"{'Source':<50}{'STATUS':<10}{'measures':>10}{'notes':>10}")
    for r in summary:
        name = str(r.get("pdf", "images") or "images")
        name = Path(name).name[:48] if name != "images" else "images"
        meas = r.get("measures", "-")
        notes = r.get("notes", "-")
        print(f"{name:<50}{r['status']:<10}{str(meas):>10}{str(notes):>10}")


if __name__ == "__main__":
    main()
