// Read config.json fields for model ids.
const ids = [
  "HuggingFaceTB/SmolLM3-3B",
  "Qwen/Qwen3.5-0.8B",
  "Qwen/Qwen3.5-4B",
  "google/gemma-4-E2B-it",
  "google/gemma-4-E4B-it",
  "ibm-granite/granite-4.2-3b",
  "microsoft/Phi-4-mini-instruct",
  "allenai/Olmo-3-7B-Instruct",
  "openbmb/MiniCPM5-2B",
  "mistralai/Ministral-3-8B-Instruct-2512",
  "meta-llama/Llama-3.2-3B-Instruct",
];
for (const id of ids) {
  try {
    const r = await fetch(`https://huggingface.co/${id}/raw/main/config.json`, {
      headers: { "user-agent": "research-bot" },
    });
    if (!r.ok) {
      console.log(`${id}\tHTTP ${r.status}`);
      continue;
    }
    const c = await r.json();
    const t = c.text_config || c;
    const ctx = t.max_position_embeddings ?? c.max_position_embeddings ?? "?";
    const layers = t.num_hidden_layers ?? "?";
    const hidden = t.hidden_size ?? "?";
    const heads = t.num_attention_heads ?? "?";
    const kv = t.num_key_value_heads ?? "?";
    const vocab = t.vocab_size ?? "?";
    console.log(`${id}\tctx=${ctx}\tlayers=${layers}\thidden=${hidden}\theads=${heads}\tkv=${kv}\tvocab=${vocab}\tarch=${c.architectures?.[0]}`);
  } catch (e) {
    console.log(`${id}\tERR ${e.message}`);
  }
}
