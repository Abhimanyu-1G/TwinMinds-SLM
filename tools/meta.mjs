// Print key metadata for a list of HF model ids.
const ids = [
  "HuggingFaceTB/SmolLM3-3B",
  "Qwen/Qwen3.5-0.8B",
  "Qwen/Qwen3.5-4B",
  "Qwen/Qwen3.5-9B",
  "google/gemma-4-E2B-it",
  "google/gemma-4-E4B-it",
  "google/gemma-4-12B-it",
  "ibm-granite/granite-4.2-3b",
  "microsoft/Phi-4-mini-instruct",
  "allenai/Olmo-3-7B-Instruct",
  "openbmb/MiniCPM5-2B",
  "mistralai/Ministral-3-8B-Instruct-2512",
  "microsoft/bitnet-b1.58-2B-4T",
  "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
  "meta-llama/Llama-3.2-3B-Instruct",
];

for (const id of ids) {
  try {
    const r = await fetch(`https://huggingface.co/api/models/${id}`, {
      headers: { "user-agent": "research-bot" },
    });
    if (!r.ok) {
      console.log(`${id}\tHTTP ${r.status}`);
      continue;
    }
    const m = await r.json();
    const lic = (m.tags || []).filter((t) => t.startsWith("license:")).join(",") || "?";
    const params = m.safetensors?.total ?? "?";
    const ctx = m.config?.max_position_embeddings ?? m.config?.text_config?.max_position_embeddings ?? "?";
    const arch = (m.config?.architectures || []).join(",");
    console.log(
      `${id}\tparams=${params}\tctx=${ctx}\t${lic}\tdl=${m.downloads}\tlikes=${m.likes}\t${arch}\tcreated=${(m.createdAt || "").slice(0, 10)}`
    );
  } catch (e) {
    console.log(`${id}\tERR ${e.message}`);
  }
}
