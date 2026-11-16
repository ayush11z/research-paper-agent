"""Single server for both modes (replaces the old Node/Express server).

  /            deep search UI (public/index.html), backed by POST /search (SSE)
  /live.html   live meeting mode, backed by the /ws/live WebSocket

Run:  .venv/bin/python app.py      (or: uvicorn app:app --port 3000)
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, Request, WebSocket  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from live.engine import LiveSession, Services  # noqa: E402

services = Services()


@asynccontextmanager
async def lifespan(_app):
    await services.start()
    print(f"Live mode ready: {json.dumps(services.status)}", flush=True)
    yield
    await services.close()


app = FastAPI(lifespan=lifespan)


@app.get("/api/status")
async def status():
    return services.status


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    await ws.accept()
    await LiveSession(ws, services).run()


@app.post("/search")
async def search(request: Request):
    """Deep search: runs citation_agent.py and streams its output as server-sent events."""
    body = await request.json()
    if not body.get("thesis") or not body.get("relatedWork"):
        return JSONResponse({"error": "thesis and relatedWork are required"}, status_code=400)

    env = {
        **os.environ,
        "THESIS": body["thesis"],
        "RELATED_WORK": body["relatedWork"],
        "FIELD": body.get("field") or "computer science",
        "TOP_N": str(body.get("topN") or 8),
        "KEYWORDS": body.get("keywords") or "",
    }

    def event(type_, data):
        return f"data: {json.dumps({'type': type_, 'data': data})}\n\n"

    async def stream():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "citation_agent.py", "--server", cwd=ROOT, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        queue: asyncio.Queue = asyncio.Queue()

        async def pump(reader, is_err):
            async for line in reader:
                await queue.put((is_err, line.decode(errors="replace").rstrip("\n")))
            await queue.put(None)

        pumps = [asyncio.create_task(pump(proc.stdout, False)),
                 asyncio.create_task(pump(proc.stderr, True))]
        try:
            open_streams = 2
            while open_streams:
                item = await queue.get()
                if item is None:
                    open_streams -= 1
                    continue
                is_err, line = item
                if not line.strip():
                    continue
                if is_err:
                    yield event("log", line)
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    yield event("log", line)
                    continue
                if isinstance(parsed, dict) and parsed.get("type") in ("gaps", "contradictions"):
                    yield event(parsed["type"], parsed["data"])
                else:
                    yield event("result", parsed)
            yield event("done", {"code": await proc.wait()})
        finally:
            for p in pumps:
                p.cancel()
            if proc.returncode is None:
                proc.kill()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=ROOT / "public", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "3000")))
