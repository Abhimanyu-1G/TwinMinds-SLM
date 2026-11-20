// Query the Hugging Face models API for a search term.
// usage: node tools/hf.mjs <search> [limit] [sort]
const [, , search, limit = "8", sort = "downloads"] = process.argv;
const url = `https://huggingface.co/api/models?search=${encodeURIComponent(
  search
)}&limit=${limit}&sort=${sort}&direction=-1`;
const r = await fetch(url, { headers: { "user-agent": "research-bot" } });
if (!r.ok) {
  console.log("HTTP", r.status);
  process.exit(1);
}
const list = await r.json();
console.log(`== ${search} (${list.length})`);
for (const m of list) {
  const tags = (m.tags || []).filter((t) => /^(text-generation|image-text|audio|.*-.*)$/.test(t)).slice(0, 6).join(",");
  console.log(`${m.id} | dl=${m.downloads} | likes=${m.likes} | created=${(m.createdAt || "").slice(0, 10)} | ${tags}`);
}
