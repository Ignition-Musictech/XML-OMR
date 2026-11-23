from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path("/mnt/e/super_omr")
OUT = ROOT / "tests/20260808_dataset_final"
HOMR_SRC = ROOT / "omr_methods/homr/homr_src"

sys.path.insert(0, str(HOMR_SRC))

from homr.circle_of_fifths import strip_naturals
from training.datasets.convert_lieder import (
    MeasureCutter, contains_only_supported_clefs, is_grandstaff)
from training.datasets.music_xml_parser import (
    music_xml_file_to_tokens)
from training.transformer.training_vocabulary import (
    calc_ratio_of_tuplets, token_lines_to_str)


def main() -> None:
    rows = [json.loads(ln) for ln in open(OUT / "row_index.jsonl")]
    n_tok = n_skip = n_prev = 0
    for rec in rows:
        if rec.get("status") != "ok":
            continue
        stem = f"{rec['piece']}_p{rec['page']}_sys{rec['sys']}"
        tok_out = OUT / "row_samples" / f"{stem}.tokens"
        if tok_out.exists():
            rec["tokens"] = str(tok_out)
            n_prev += 1
            continue
        try:
            txt = Path(rec["musicxml"]).read_text(encoding="utf-8")
            clean = re.sub(r"<miscellaneous>.*?</miscellaneous>", "",
                           txt, flags=re.S)
            with tempfile.NamedTemporaryFile(
                    "w", suffix=".musicxml", delete=False,
                    encoding="utf-8") as tf:
                tf.write(clean)
                tmp_xml = tf.name
            voices = music_xml_file_to_tokens(tmp_xml)
            if len(voices) != 1 or not voices[0]:
                raise ValueError(f"voices={len(voices)}")
            voice = voices[0]
            if not is_grandstaff(voice):
                raise ValueError("not_grandstaff")
            cutter = MeasureCutter(voice)
            symbols = cutter.extract_measures(len(voice))
            rec["tuplet_ratio"] = round(calc_ratio_of_tuplets(symbols), 4)
            if not contains_only_supported_clefs(symbols):
                raise ValueError("unsupported_clefs")
            symbols = strip_naturals(symbols)
            tok_out.write_text(token_lines_to_str(symbols),
                               encoding="utf-8")
            rec["tokens"] = str(tok_out)
            rec.pop("homr_tokens", None)
            n_tok += 1
        except Exception as e:
            rec["homr_tokens"] = f"skip: {str(e)[:100]}"
            n_skip += 1
    with open(OUT / "row_index.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"tokens written {n_tok}, already had {n_prev}, "
          f"skipped {n_skip}")


if __name__ == "__main__":
    main()
