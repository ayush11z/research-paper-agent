const express = require("express");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

// Manually parse .env so we don't need the dotenv package
function loadDotenv(envPath) {
  try {
    const lines = fs.readFileSync(envPath, "utf8").split("\n");
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const idx = trimmed.indexOf("=");
      if (idx === -1) continue;
      const key = trimmed.slice(0, idx).trim();
      let val = trimmed.slice(idx + 1).trim();
      if ((val.startsWith('"') && val.endsWith('"')) ||
          (val.startsWith("'") && val.endsWith("'"))) {
        val = val.slice(1, -1);
      }
      if (!(key in process.env)) process.env[key] = val;
    }
  } catch {
    // .env is optional
  }
}
loadDotenv(path.join(__dirname, ".env"));

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

app.post("/search", (req, res) => {
  const { thesis, relatedWork, field, topN, keywords } = req.body;

  if (!thesis || !relatedWork) {
    return res.status(400).json({ error: "thesis and relatedWork are required" });
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();          // flush 200 + headers immediately so the client knows the stream is open
  req.socket.setTimeout(0);   // disable socket idle timeout for long-running streams

  const send = (type, data) => {
    if (!res.writableEnded) res.write(`data: ${JSON.stringify({ type, data })}\n\n`);
  };

  const py = spawn("python3", ["citation_agent.py", "--server"], {
    cwd: __dirname,
    env: {
      ...process.env,
      THESIS: thesis,
      RELATED_WORK: relatedWork,
      FIELD: field || "computer science",
      TOP_N: String(topN || 8),
      KEYWORDS: keywords || "",
      GEMINI_API_KEY: process.env.GEMINI_API_KEY || "",
    },
  });

  const handleLine = (line) => {
    console.log("LINE:", line);
    try {
      const parsed = JSON.parse(line);
      if (parsed.type === "gaps" || parsed.type === "contradictions") {
        send(parsed.type, parsed.data);
      } else {
        send("result", parsed);
      }
    } catch { send("log", line); }
  };

  let stdoutBuffer = "";
  py.stdout.on("data", (data) => {
    stdoutBuffer += data.toString();
    const lines = stdoutBuffer.split("\n");
    stdoutBuffer = lines.pop();
    lines.filter(Boolean).forEach(handleLine);
  });

  py.stderr.on("data", (data) => {
    console.error("STDERR:", data.toString());
    send("log", data.toString());
  });

  py.on("close", (code) => {
    console.log("Python exited with code:", code);
    if (stdoutBuffer.trim()) {
      stdoutBuffer.split("\n").filter(Boolean).forEach(handleLine);
    }
    send("done", { code });
    res.end();
  });

  res.on("close", () => py.kill());
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
