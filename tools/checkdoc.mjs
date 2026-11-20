// Extract every markdown link target from the playbook and check it.
import { readFileSync } from "node:fs";

const md = readFileSync("SLM-Project-Playbook.md", "utf8");
const urls = [...new Set([...md.matchAll(/\]\((https?:\/\/[^)\s]+)\)/g)].map((m) => m[1]))];
console.log(`unique urls: ${urls.length}`);

const results = [];
async function check(u) {
  try {
    const r = await fetch(u, {
      redirect: "follow",
      headers: { "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-bot" },
      signal: AbortSignal.timeout(30000),
    });
    results.push({ u, status: r.status, final: r.url });
    console.log(`${r.status}\t${u}`);
  } catch (e) {
    results.push({ u, status: 0, err: String(e.cause?.code || e.message).slice(0, 50) });
    console.log(`0\t${u}\t${e.cause?.code || e.message}`);
  }
}
const q = [...urls];
await Promise.all(
  Array.from({ length: 6 }, async () => {
    while (q.length) await check(q.shift());
  })
);
const bad = results.filter((r) => r.status !== 200);
console.log(`\nchecked ${results.length}; non-200: ${bad.length}`);
for (const b of bad) console.log(`BAD ${b.status} ${b.u} ${b.err || ""}`);
