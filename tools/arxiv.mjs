// Fetch arXiv abstracts by id list.
const ids = process.argv.slice(2);
if (!ids.length) {
  console.error("usage: node tools/arxiv.mjs <id> [id...]");
  process.exit(2);
}
const url = `http://export.arxiv.org/api/query?id_list=${ids.join(",")}&max_results=${ids.length}`;
const r = await fetch(url, { headers: { "user-agent": "research-bot" } });
const xml = await r.text();
const entries = xml.split("<entry>").slice(1);
for (const e of entries) {
  const get = (tag) => {
    const m = e.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`));
    return m ? m[1].replace(/\s+/g, " ").trim() : "";
  };
  console.log(`\n### ${get("title")}\nID: ${get("id")}\n${get("summary")}`);
}
