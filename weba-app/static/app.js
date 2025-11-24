const form = document.getElementById("infer-form");
const out = document.getElementById("output");
const statusEl = document.getElementById("status");
const btn = document.getElementById("run-btn");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    btn.disabled = true;
    out.textContent = "";
    statusEl.textContent = "Generating…";

    const payload = {
        prompt_text: document.getElementById("prompt").value || "What is the purpose of life?",
        max_new_tokens: parseInt(document.getElementById("tokens").value, 10) || 120,
        top_k: parseInt(document.getElementById("topk").value, 10) || 0, // 0 disables
        temperature: parseFloat(document.getElementById("temperature").value) || 0.8,
    };

    try {
        const res = await fetch("generate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        out.textContent = data.text || "";
        statusEl.textContent = "Done.";
    } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
    } finally {
        btn.disabled = false;
    }
});
