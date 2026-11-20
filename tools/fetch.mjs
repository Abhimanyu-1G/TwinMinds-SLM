// Minimal page fetcher: node tools/fetch.mjs <url> <outfile> [--raw]
import { writeFileSync } from "node:fs";

const [, , url, outfile, ...flags] = process.argv;
if (!url || !outfile) {
  console.error("usage: node fetch.mjs <url> <outfile> [--raw]");
  process.exit(2);
}

const decode = (s) =>
  s
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x27;/g, "'")
    .replace(/&mdash;/g, "—")
    .replace(/&ndash;/g, "–")
    .replace(/&hellip;/g, "…");

function toText(html) {
  let h = html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<svg[\s\S]*?<\/svg>/gi, " ")
    .replace(/<noscript[\s\S]*?<\/noscript>/gi, " ")
    .replace(/<!--[\s\S]*?-->/g, " ");
  h = h.replace(/<(h[1-6])[^>]*>/gi, "\n\n## ");
  h = h.replace(/<\/(h[1-6])>/gi, "\n");
  h = h.replace(/<(p|div|section|article|li|tr|br|pre|blockquote)[^>]*>/gi, "\n");
  h = h.replace(/<\/(p|div|section|article|li|tr|pre|blockquote)>/gi, "\n");
  h = h.replace(/<a [^>]*href="([^"]+)"[^>]*>/gi, " [$1] ");
  h = h.replace(/<[^>]+>/g, " ");
  h = decode(h);
  h = h
    .split("\n")
    .map((l) => l.replace(/[ \t\r\f\v]+/g, " ").trim())
    .filter((l) => l.length > 1)
    .join("\n");
  h = h.replace(/\n{3,}/g, "\n\n");
  return h.trim();
}

const res = await fetch(url, {
  headers: {
    "user-agent":
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    accept: "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
  },
  redirect: "follow",
});
const ct = res.headers.get("content-type") || "";
const body = await res.text();
const out = flags.includes("--raw") || !/html/i.test(ct) ? body : toText(body);
writeFileSync(outfile, out, "utf8");
console.log(JSON.stringify({ status: res.status, ct, bytes: body.length, chars: out.length, outfile }));
