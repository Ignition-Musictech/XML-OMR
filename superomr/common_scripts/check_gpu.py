from __future__ import annotations

import glob
import os
import site
import sys

_REEXEC_FLAG = "SUPER_OMR_GPU_CHECK_REEXEC"


def ensure_lib_path() -> None:
    if os.environ.get(_REEXEC_FLAG):
        return
    dirs: list[str] = []
    for sp in site.getsitepackages():
        dirs += sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
    if not dirs:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(dirs) + (f":{cur}" if cur else "")
    os.environ[_REEXEC_FLAG] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def check_torch() -> dict:
    try:
        import torch
    except Exception as exc:
        return {"error": f"torch import failed: {exc}", "detail": ""}
    try:
        if not torch.cuda.is_available():
            return {"error": "torch.cuda.is_available() is False",
                    "detail": f"torch {torch.__version__}"}
        x = torch.randn(64, 64, device="cuda")
        float((x @ x).sum().item())
        name = torch.cuda.get_device_name(0)
        return {"error": None,
                "detail": f"torch {torch.__version__} on {name}"}
    except Exception as exc:
        return {"error": f"torch GPU run failed: {exc}", "detail": ""}


def _homr_encoder_path() -> str | None:
    try:
        from homr.transformer.configs import Config
        p = str(Config().filepaths.encoder_path_fp16)
        return p if os.path.exists(p) else None
    except Exception:
        return None


_ORT_PROBE = r"""
import json, os, sys
out = {"error": None, "detail": ""}
try:
    import onnxruntime as ort
    out["detail"] = f"onnxruntime {ort.__version__}"
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        out["error"] = ("this onnxruntime build has no CUDAExecutionProvider "
                        "(the CPU-only 'onnxruntime' package is installed "
                        "instead of 'onnxruntime-gpu')")
    else:
        model = None
        try:
            from homr.transformer.configs import Config
            cand = str(Config().filepaths.encoder_path_fp16)
            model = cand if os.path.exists(cand) else None
        except Exception:
            model = None
        if model is None:
            out["error"] = ("homr's ONNX weights are not on disk yet, so the "
                            "CUDA session could not be proven. Run one page "
                            "through homr to fetch them, then re-run this check")
        else:
            sess = ort.InferenceSession(
                model,
                providers=[("CUDAExecutionProvider",
                            {"cudnn_conv_algo_search": "DEFAULT"})])
            got = sess.get_providers()
            if "CUDAExecutionProvider" not in got:
                out["error"] = ("onnxruntime silently fell back to CPU "
                                f"(session providers: {got}). The usual cause "
                                "is a CUDA major-version mismatch between "
                                "onnxruntime-gpu and torch; see requirements.txt")
except Exception as exc:
    out["error"] = f"onnxruntime probe failed: {exc}"
sys.stderr.flush()
print("__ORT_RESULT__" + json.dumps(out))
"""


def check_onnxruntime() -> dict:
    import json
    import subprocess
    try:
        p = subprocess.run([sys.executable, "-c", _ORT_PROBE],
                           capture_output=True, text=True, timeout=300)
    except Exception as exc:
        return {"error": f"onnxruntime probe could not run: {exc}",
                "detail": ""}
    for line in p.stdout.splitlines():
        if line.startswith("__ORT_RESULT__"):
            try:
                return json.loads(line[len("__ORT_RESULT__"):])
            except Exception:
                break
    tail = (p.stderr or p.stdout or "").strip().splitlines()
    return {"error": "onnxruntime probe produced no result"
                     + (f" ({tail[-1][:160]})" if tail else ""),
            "detail": ""}


def check_all() -> dict:
    t = check_torch()
    o = check_onnxruntime()
    errs = [r["error"] for r in (t, o) if r["error"]]
    return {"error": "; ".join(errs) if errs else None,
            "torch": t, "onnxruntime": o}


def main() -> None:
    ensure_lib_path()
    r = check_all()
    for label, key in (("torch", "torch"),
                       ("onnxruntime (homr)", "onnxruntime")):
        sub = r[key]
        mark = "OK  " if sub["error"] is None else "FAIL"
        print(f"[{mark}] {label:<22} {sub['detail'] or sub['error']}")
        if sub["error"] and sub["detail"]:
            print(f"       {sub['error']}")
    if r["error"]:
        print("\nGPU preflight FAILED.")
        sys.exit(1)
    print("\nGPU preflight passed: both runtimes are on the GPU.")


if __name__ == "__main__":
    main()
