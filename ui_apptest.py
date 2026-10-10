"""
ui_apptest.py - integration test of the Streamlit UI against the REAL API.

It runs Streamlit's AppTest (the UI script executed in-process, driven like a user) and the UI
talks to the Flask API over HTTP. Inside the Docker setup it is executed in the ui container:
    docker compose exec -T ui python - < tests/ui_apptest.py
Environment: APP_PATH (default /app/streamlit_app.py), API_URL (set by the container).
"""
import json
import os
import sys

from streamlit.testing.v1 import AppTest

APP = os.environ.get("APP_PATH", "/app/streamlit_app.py")
good_api = os.environ.get("API_URL", "http://localhost:5000")
checks = []


def check(name, ok, detail):
    checks.append({"test": name, "passed": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def metric(at, label):
    return next((m.value for m in at.metric if m.label == label), None)


at = AppTest.from_file(APP, default_timeout=30).run()
check("UI page renders without an exception", not at.exception, f"exceptions={len(at.exception)}")
check("UI title is shown", any("Digit Classifier" in t.value for t in at.title), [t.value for t in at.title])
check("Sidebar reports that the API is reachable", any("API reachable" in s.value for s in at.sidebar.success),
      [s.value for s in at.sidebar.success])
check("Sidebar shows the model accuracy from the API", metric(at.sidebar, "Model test accuracy") is not None,
      metric(at.sidebar, "Model test accuracy"))

correct, rows = 0, []
for d in range(10):
    at.selectbox(key="digit").select(d).run()
    at.button(key="predict_sample").click().run()
    pred, conf = metric(at, "Predicted digit"), metric(at, "Confidence")
    ok = pred == str(d)
    correct += ok
    rows.append({"true": d, "predicted": pred, "confidence": conf})
    print(f"        digit {d} -> UI shows predicted {pred}, confidence {conf}")
check("UI classifies all 10 sample digits through the API", correct == 10, f"{correct}/10 correct")

sample = [s for s in at.session_state["samples"] if s["label"] == 3][0]["pixels"]
at.text_area(key="pixels_text").set_value(", ".join(str(int(v)) for v in sample)).run()
at.button(key="predict_text").click().run()
check("Pixel-value tab predicts a pasted image of the digit 3", metric(at, "Predicted digit") == "3",
      f"predicted {metric(at, 'Predicted digit')}")

at.text_area(key="pixels_text").set_value("1, 2, 3").run()
at.button(key="predict_text").click().run()
errs = [e.value for e in at.error]
check("Invalid input shows the API error message in the UI",
      any("64 pixel values" in e for e in errs) and not at.exception, errs[:1])

os.environ["API_URL"] = "http://127.0.0.1:9"      # nothing listens here: simulate an API outage
down = AppTest.from_file(APP, default_timeout=30).run()
os.environ["API_URL"] = good_api
errs = [e.value for e in down.sidebar.error]
check("UI shows an error (no crash) when the API is unreachable", bool(errs) and not down.exception,
      (errs[0][:70] if errs else "no error shown"))

passed = sum(c["passed"] for c in checks)
print(f"\n{passed}/{len(checks)} UI integration checks passed")
sys.exit(0 if passed == len(checks) else 1)
