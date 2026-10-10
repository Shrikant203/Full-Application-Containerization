#!/usr/bin/env python3
"""
test_api.py - black-box tests for the running API (standard library only).

Usage:  python test_api.py --url http://localhost:5000 --out api_test_results.json
Works against a local server, gunicorn, or the Docker container (port 5000 published).
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request


def call(base, path, method="GET", body=None, raw=None, headers=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {"Content-Type": "application/json"} if data is not None else {}
    h.update(headers or {})
    req = urllib.request.Request(base + path, data=data, method=method, headers=h)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            status, text = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        status, text = e.code, e.read().decode()
    ms = (time.perf_counter() - t0) * 1000
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = text
    return status, parsed, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:5000")
    ap.add_argument("--out", default="api_test_results.json")
    a = ap.parse_args()
    base, checks = a.url.rstrip("/"), []

    def check(name, ok, detail):
        checks.append({"test": name, "passed": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    s, b, _ = call(base, "/health")
    check("GET /health returns 200 and status ok", s == 200 and b.get("status") == "ok", f"HTTP {s} {json.dumps(b)}")
    s, b, _ = call(base, "/")
    check("GET / lists the endpoints", s == 200 and "endpoints" in b, f"HTTP {s}, endpoints={list(b.get('endpoints', {}))}")
    s, info, _ = call(base, "/model-info")
    check("GET /model-info returns model metadata", s == 200 and info.get("parameters") == 4810,
          f"HTTP {s}, parameters={info.get('parameters')}, test_accuracy={info.get('test_accuracy')}")
    s, samples, _ = call(base, "/samples")
    check("GET /samples returns 10 labelled examples", s == 200 and len(samples) == 10, f"HTTP {s}, {len(samples)} samples")

    correct, rows = 0, []
    for smp in samples:
        s, r, _ = call(base, "/predict", "POST", {"pixels": smp["pixels"]})
        ok = s == 200 and r.get("digit") == smp["label"]
        correct += ok
        rows.append({"true": smp["label"], "predicted": r.get("digit"), "confidence": r.get("confidence")})
    check("POST /predict classifies the 10 samples", correct == 10, f"{correct}/10 correct")
    for row in rows: print(f"        true {row['true']} -> predicted {row['predicted']} (confidence {row['confidence']})")

    s, r, _ = call(base, "/predict", "POST", {"instances": [x["pixels"] for x in samples]})
    batch_ok = s == 200 and r.get("count") == 10 and [p["digit"] for p in r["predictions"]] == [x["label"] for x in samples]
    check("POST /predict batch of 10 instances", batch_ok, f"HTTP {s}, count={r.get('count')}, inference_ms={r.get('inference_ms')}")

    s, r, _ = call(base, "/predict", "POST", {"pixels": [1, 2, 3]})
    check("Wrong input length is rejected with 400", s == 400, f"HTTP {s} {json.dumps(r)}")
    s, r, _ = call(base, "/predict", "POST", {"pixels": ["a"] * 64})
    check("Non-numeric pixels are rejected with 400", s == 400, f"HTTP {s} {json.dumps(r)}")
    s, r, _ = call(base, "/predict", "POST", {"pixels": [99] * 64})
    check("Out-of-range pixels are rejected with 400", s == 400, f"HTTP {s} {json.dumps(r)}")
    s, r, _ = call(base, "/predict", "POST", raw=b"not json")
    check("Invalid JSON is rejected with 400", s == 400, f"HTTP {s} {json.dumps(r)}")
    s, r, _ = call(base, "/nope")
    check("Unknown route returns JSON 404", s == 404 and isinstance(r, dict), f"HTTP {s} {json.dumps(r)}")
    s, r, _ = call(base, "/predict")
    check("GET on /predict returns JSON 405", s == 405 and isinstance(r, dict), f"HTTP {s} {json.dumps(r)}")

    lat = [call(base, "/predict", "POST", {"pixels": samples[i % 10]["pixels"]})[2] for i in range(50)]
    summary = {"requests": 50, "mean_ms": round(statistics.mean(lat), 2), "median_ms": round(statistics.median(lat), 2),
               "max_ms": round(max(lat), 2)}
    check("Latency of 50 sequential /predict calls measured", True, json.dumps(summary))

    passed = sum(c["passed"] for c in checks)
    print(f"\n{passed}/{len(checks)} checks passed")
    json.dump({"base_url": base, "passed": passed, "total": len(checks), "checks": checks, "predictions": rows,
               "latency": summary}, open(a.out, "w"), indent=2)
    sys.exit(0 if passed == len(checks) else 1)


if __name__ == "__main__":
    main()
