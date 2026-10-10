#!/usr/bin/env python3
"""
task11_docker.py - runs the complete Docker workflow for Task 11 (Full Application
Containerization: model + Flask API + Streamlit UI) and records the REAL output of every command.

Steps: verify Docker/Compose -> validate config and build both images -> start the application with
docker compose -> test API, UI and the container integration -> inspect containers and network ->
resilience checks (restart / stop / start the API) -> clean up.

Usage:
    1. Start Docker Desktop and wait for "Engine running".
    2. Run from the project folder (the one containing docker-compose.yml):
           python3 task11_docker.py
Outputs:
    docker_command_log.txt      plain-text log of every command, output and exit code
    docker_results.json         structured results (used to build the report)
    api_test_results.json       results of tests/test_api.py against the API container
    ui_integration_test.txt     output of the UI integration test run inside the ui container
    image_size_summary.txt      sizes of the two images
Host ports used: 8000 (API) and 8501 (Streamlit UI). They must be free.
"""
import json
import os
import socket
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

TABLE = "table {{.Names}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}"


def wait(name, tries=60):
    return ("for i in $(seq 1 %d); do s=$(docker inspect --format '{{.State.Health.Status}}' %s); "
            "echo \"check $i: $s\"; [ \"$s\" = healthy ] && break; sleep 2; done; [ \"$s\" = healthy ]" % (tries, name))


PY_HEALTH = "\"import urllib.request; print(urllib.request.urlopen('http://api:5000/health').read().decode())\""
PY_MODS = ("\"import importlib.util as u; print('flask installed:', u.find_spec('flask') is not None); "
           "print('streamlit installed:', u.find_spec('streamlit') is not None); print('numpy installed:', u.find_spec('numpy') is not None)\"")

A = "1. Verify Docker and Compose"
B = "2. Validate the configuration and build the images"
C = "3. Start the full application with Docker Compose"
D = "4. Test the application"
E = "5. Inspect the containers and the network"
F = "6. Resilience and dependency checks"
G = "7. Cleanup"

# (section, title, command, critical)
STEPS = [
    (A, "Docker version", "docker --version && docker version --format 'Client: {{.Client.Version}} | Server: {{.Server.Version}}'", True),
    (A, "Docker Compose version", "docker compose version", True),
    (B, "Services defined in docker-compose.yml", "docker compose config --services", True),
    (B, "Build both images (api and ui)", "docker compose build", True),
    (B, "List the images", "docker image ls --filter 'reference=dl-*' --format 'table {{.Repository}}\\t{{.Tag}}\\t{{.ID}}\\t{{.Size}}'", False),
    (B, "Size of the API image in bytes", "docker image inspect dl-api:1.0 --format '{{.Size}}'", False),
    (B, "Size of the UI image in bytes", "docker image inspect dl-ui:1.0 --format '{{.Size}}'", False),
    (B, "Layer history of the UI image", "docker history dl-ui:1.0", False),
    (C, "Start both containers (docker compose up -d)", "docker compose up -d", True),
    (C, "Wait until the API container is healthy", wait("dl-api"), True),
    (C, "Wait until the UI container is healthy", wait("dl-ui"), True),
    (C, "List the running containers", "docker ps --filter name=dl- --format '" + TABLE + "'", False),
    (D, "API test suite (tests/test_api.py)", "python3 tests/test_api.py --url http://localhost:8000 --out api_test_results.json", False),
    (D, "UI health endpoint", "curl -s http://localhost:8501/_stcore/health", False),
    (D, "UI main page", "curl -s -o /dev/null -w 'HTTP status: %{http_code}\\n' http://localhost:8501/", False),
    (D, "UI container reaches the API container by service name", "docker compose exec -T ui python -c " + PY_HEALTH, False),
    (D, "UI integration test inside the ui container (Streamlit AppTest against the real API)", "docker compose exec -T ui python - < tests/ui_apptest.py 2>&1 | grep -v -i 'warning\\|ScriptRunContext'; exit ${PIPESTATUS[0]}", False),
    (E, "User inside both containers (should not be root)", "docker exec dl-api id && docker exec dl-ui id", False),
    (E, "Container configuration and health", "docker inspect dl-api dl-ui --format '{{.Name}}: User={{.Config.User}} | Health={{.State.Health.Status}} | Image={{.Config.Image}} | Restart={{.HostConfig.RestartPolicy.Name}}'", False),
    (E, "Containers attached to the private network", "docker network inspect dl-app-net --format '{{range .Containers}}{{.Name}} -> {{.IPv4Address}}{{println}}{{end}}'", False),
    (E, "API address configured in the UI container", "docker exec dl-ui printenv API_URL", False),
    (E, "Installed components: api container", "docker exec dl-api python -c " + PY_MODS, False),
    (E, "Installed components: ui container", "docker exec dl-ui python -c " + PY_MODS, False),
    (E, "Container logs (last lines of both services)", "docker compose logs --tail 6", False),
    (E, "Resource usage of both containers", "docker stats --no-stream dl-api dl-ui", False),
    (F, "Restart the API container; the UI can reach it again", "docker restart dl-api && " + wait("dl-api") + " && docker compose exec -T ui python -c " + PY_HEALTH, False),
    (F, "Stop the API; the UI container keeps running", "docker compose stop api && sleep 2 && curl -s http://localhost:8501/_stcore/health && echo && docker ps -a --filter name=dl- --format 'table {{.Names}}\\t{{.Status}}'", False),
    (F, "Start the API again and repeat the UI integration test", "docker compose start api && " + wait("dl-api") + " && docker compose exec -T ui python - < tests/ui_apptest.py 2>&1 | grep -v -i 'warning\\|ScriptRunContext' | tail -4", False),
    (G, "Stop and remove the containers and the network (images are kept)", "docker compose down && docker ps -a --format 'table {{.Names}}\\t{{.Status}}'", False),
]


def run_cmd(cmd, timeout=2400):
    t0 = datetime.now()
    try:
        p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
        rc = p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "Command timed out.", 124
    return {"command": cmd, "output": out.rstrip(), "returncode": rc, "time": t0.strftime("%Y-%m-%d %H:%M:%S")}


def port_in_use(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def save(results):
    json.dump(results, open("docker_results.json", "w"), indent=2)
    with open("docker_command_log.txt", "w") as f:
        f.write(f"Task 11 Docker command log - generated {datetime.now():%Y-%m-%d %H:%M:%S}\n" + "=" * 78 + "\n")
        for i, r in enumerate(results, 1):
            f.write(f"\n[{i}] {r['section']} - {r['title']}  ({r['time']})\n$ {r['command']}\n{r['output']}\n"
                    f"(exit code {r['returncode']})\n" + "-" * 78 + "\n")


def main():
    if run_cmd("docker info", 60)["returncode"] != 0:
        sys.exit("Docker is not running. Start Docker Desktop, wait for 'Engine running', then run again.")
    for f in ("docker-compose.yml", "api/Dockerfile", "ui/Dockerfile", "api/app.py", "ui/streamlit_app.py",
              "api/model/model.npz", "tests/test_api.py", "tests/ui_apptest.py"):
        if not os.path.exists(f):
            sys.exit(f"Missing file: {f}. Run this script from the project folder.")
    # remove leftovers of an earlier run so that names and ports are free
    run_cmd("docker compose down --remove-orphans; docker rm -f dl-api dl-ui; true")
    busy = [p for p in (8000, 8501) if port_in_use(p)]
    if busy:
        sys.exit(f"Host port(s) {busy} are already in use by another program or container. "
                 f"Stop it (see 'docker ps') and run this script again.")

    results = []
    for i, (section, title, cmd, critical) in enumerate(STEPS, 1):
        r = run_cmd(cmd)
        r.update({"section": section, "title": title})
        results.append(r)
        print(f"\n### [{i}/{len(STEPS)}] {title}\n$ {cmd}\n{r['output'][-3000:]}\n(exit code {r['returncode']})")
        if critical and r["returncode"] != 0:
            save(results)
            print(f"\nSTOPPED: the critical step '{title}' failed. Nothing after it can work.\n"
                  f"Please send me the file docker_command_log.txt (it contains the error).")
            run_cmd("docker compose down --remove-orphans; true")
            sys.exit(1)

    def out_of(title):
        return next(x for x in results if x["title"] == title)["output"]

    def num(title):
        d = "".join(ch for ch in out_of(title).split("\n")[0] if ch.isdigit())
        return int(d) if d else None
    a, u = num("Size of the API image in bytes"), num("Size of the UI image in bytes")
    if a and u:
        txt = (f"dl-api:1.0 (Flask + gunicorn + NumPy model) : {a:>13,} bytes = {a / 1e6:8.1f} MB\n"
               f"dl-ui:1.0  (Streamlit UI)                   : {u:>13,} bytes = {u / 1e6:8.1f} MB\n")
        open("image_size_summary.txt", "w").write(txt)
        print("\n" + txt)
    open("ui_integration_test.txt", "w").write(
        out_of("UI integration test inside the ui container (Streamlit AppTest against the real API)") + "\n")
    save(results)
    ok = sum(1 for r in results if r["returncode"] == 0)
    print(f"\nDone: {ok}/{len(results)} commands succeeded. Files: docker_command_log.txt, docker_results.json, "
          f"api_test_results.json, ui_integration_test.txt, image_size_summary.txt")


if __name__ == "__main__":
    main()
