const $ = (s) => document.querySelector(s);
let types = [], active = null;

document.querySelectorAll("nav button").forEach((b) => (b.onclick = () => show(b.dataset.tab)));
function show(tab) {
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.id === tab));
  if (tab === "log") loadLog();
  if (tab === "design") loadDesign();
}

const depth = (l) => (l === "l2" ? "layer 2" : "layers 2+3");

async function loadTypes(select) {
  types = await (await fetch("/api/types")).json();
  $("#types").innerHTML = types.map((t) =>
    `<button class="type" data-id="${t.id}"><b>${t.name}</b><small>${t.size} px · ${depth(t.layers)} · ${t.n_train} good images</small></button>`).join("");
  document.querySelectorAll(".type").forEach((b) => (b.onclick = () => { pick(b.dataset.id); show("inspect"); }));
  pick(select || (active && active.id) || types[0].id);
}

async function pick(id) {
  active = types.find((t) => t.id === id);
  document.querySelectorAll(".type").forEach((b) => b.classList.toggle("on", b.dataset.id === id));
  $("#activeName").textContent = active.name;
  const s = await (await fetch(`/api/samples/${id}`)).json();
  $("#samples").innerHTML = s.length ? s.map((x) =>
    `<button data-url="${x.url}" data-label="${x.label}"><img src="${x.url}" alt="${x.label}"><small>${x.label.replace(/_/g, " ")}</small></button>`).join("")
    : `<p class="hint">No samples for this part. Drop your own image.</p>`;
  document.querySelectorAll("#samples button").forEach((b) => (b.onclick = async () => {
    const blob = await (await fetch(b.dataset.url)).blob();
    inspect(new File([blob], b.dataset.url.split("/").pop(), { type: blob.type }), b.dataset.label);
  }));
  $("#result").innerHTML = `<div class="empty">The heatmap and verdict appear here.</div>`;
}

async function inspect(file, label = "") {
  $("#result").innerHTML = `<div class="empty">Inspecting…</div>`;
  const fd = new FormData();
  fd.append("type", active.id); fd.append("image", file); fd.append("label", label);
  const res = await fetch("/api/inspect", { method: "POST", body: fd });
  const r = await res.json();
  if (!res.ok) { $("#result").innerHTML = `<div class="empty">${r.detail}</div>`; return; }
  const src = URL.createObjectURL(file);
  const pct = Math.min(100, (r.score / (r.threshold * 1.6)) * 100), mark = (1 / 1.6) * 100;
  let truth = "";
  if (label) {
    const defect = label !== "good", flagged = r.verdict === "FAIL";
    truth = defect === flagged
      ? `<div class="truth ok">Known label: ${label.replace(/_/g, " ")}. Correct call.</div>`
      : `<div class="truth miss">Known label: ${label.replace(/_/g, " ")}. ${defect ? "Missed defect." : "False alarm."}</div>`;
  }
  $("#result").innerHTML = `
    <span class="verdict ${r.verdict}">${r.verdict}</span>
    <div class="pair"><div><img src="${src}" alt="input"><small>Input</small></div>
      <div><img src="${r.overlay}" alt="heatmap"><small>Heatmap: red = most unlike the good parts</small></div></div>
    <div class="bar"><i style="width:${pct}%;background:${r.verdict === "FAIL" ? "var(--red)" : "var(--green)"}"></i><s style="left:${mark}%"></s></div>
    <div class="nums"><span>Score <b>${r.score.toFixed(2)}</b></span><span>Threshold <b>${r.threshold.toFixed(2)}</b></span>
      <span>Time <b>${r.ms.toFixed(0)} ms</b></span><span>Part <b>${active.name}</b></span></div>${truth}`;
}

const drop = $("#drop");
drop.querySelector("input").onchange = (e) => e.target.files[0] && inspect(e.target.files[0]);
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) inspect(e.dataTransfer.files[0]); });

$("#teachFiles").onchange = (e) => ($("#teachCount").textContent = `${e.target.files.length} images chosen`);
$("#teachBtn").onclick = async () => {
  const files = $("#teachFiles").files, name = $("#teachName").value.trim();
  if (!name) { $("#teachMsg").textContent = "Give the part a name."; return; }
  if (files.length < 5) { $("#teachMsg").textContent = "Choose at least 5 good images."; return; }
  const fd = new FormData();
  fd.append("name", name);
  [...files].forEach((f) => fd.append("images", f));
  $("#teachBtn").disabled = true;
  $("#teachMsg").textContent = `Learning what a good ${name} looks like from ${files.length} images…`;
  const res = await fetch("/api/teach", { method: "POST", body: fd });
  const r = await res.json();
  $("#teachBtn").disabled = false;
  if (!res.ok) { $("#teachMsg").textContent = r.detail; return; }
  $("#teachMsg").textContent = `Done in ${r.seconds} s from ${r.n_train} good images. Threshold ${r.threshold}.`;
  await loadTypes(r.id);
};

async function loadLog() {
  const { rows, counts } = await (await fetch("/api/log")).json();
  $("#counts").innerHTML = `<div><b>${counts.total}</b>inspected</div><div><b>${counts.pass}</b>passed</div><div><b>${counts.fail}</b>failed</div>`;
  $("#logRows").innerHTML = rows.map((r) => `<tr><td>${r.time.replace("T", " ")}</td><td>${r.type}</td><td>${r.file || ""}</td>
    <td>${r.score.toFixed(2)}</td><td>${r.threshold.toFixed(2)}</td><td class="${r.verdict}">${r.verdict}</td><td>${r.ms.toFixed(0)}</td>
    <td>${(r.label || "").replace(/_/g, " ")}</td></tr>`).join("");
}

const f3 = (v) => Number(v).toFixed(3);
async function loadDesign() {
  const d = await (await fetch("/api/design")).json();
  const c = d.chosen, h = d.heavy, s = d.smallest, a = d.all_categories;
  const tpe = d.search && d.search.results["tpe@40"], rnd = d.search && d.search.results["random@40"];
  let html = `<div class="tiles">
    <div class="tile t1"><b>${c.footprint_mb} MB</b><span>our design vs ${h.footprint_mb} MB for WideResNet-50, same held-out AUROC (${f3(c.test_auroc)})</span></div>
    <div class="tile t2"><b>${(tpe.mean * 100).toFixed(1)}%</b><span>of the best trade-off found in 40 of 300 tries (random search: ${(rnd.mean * 100).toFixed(1)}%)</span></div>
    <div class="tile t3"><b>${d.edge.model_mb.onnx_int8} MB</b><span>INT8 model, down from ${d.edge.model_mb.onnx_fp32} MB, accuracy kept</span></div>
    <div class="tile t4"><b>${f3(a.mean_image_auroc)}</b><span>mean AUROC on all 15 MVTec AD types, one fixed design</span></div></div>
  <div class="grid2"><div class="panel"><h2>How the design was chosen</h2>
    <p class="hint">An ML optimiser searched 300 versions of the detector for accuracy, speed and size.</p>
    <img class="chart" src="/results/design/design_space.png" alt="design space">
    <table><thead><tr><th>Design (held-out images)</th><th>AUROC</th><th>ms</th><th>MB</th></tr></thead><tbody>
      <tr><td>WideResNet-50, 256 px</td><td>${f3(h.test_auroc)}</td><td>${h.latency_ms}</td><td>${h.footprint_mb}</td></tr>
      <tr><td><b>Ours: ResNet-18, 224 px</b></td><td><b>${f3(c.test_auroc)}</b></td><td><b>${c.latency_ms}</b></td><td><b>${c.footprint_mb}</b></td></tr>
      <tr><td>Smallest within 0.01</td><td>${f3(s.test_auroc)}</td><td>${s.latency_ms}</td><td>${s.footprint_mb}</td></tr></tbody></table>
    <p class="note">Laptop CPU, 4 threads. Not yet measured on a Raspberry Pi.</p></div>
  <div class="panel"><h2>Settings per part type</h2>
    <table><thead><tr><th>Part</th><th>Input</th><th>Features</th><th>Threshold rule</th><th>Memory</th></tr></thead><tbody>
    ${d.types.map((t) => `<tr><td>${t.name}</td><td>${t.size} px</td><td>${depth(t.layers)}</td><td>${t.rule}</td><td>${t.bank_mb} MB</td></tr>`).join("")}
    </tbody></table>`;
  if (d.weak_types) {
    html += `<h3>Weak types: design search per type (held-out AUROC)</h3><table><thead><tr><th>Part</th><th>Default</th><th>Best found</th><th>Design</th></tr></thead><tbody>
    ${Object.entries(d.weak_types).map(([k, v]) => `<tr><td>${k}</td><td>${f3(v.default.held_out_auroc)}</td><td>${f3(v.pick.held_out_auroc)}</td>
      <td>${v.pick.size} px · ${depth(v.pick.layers)} · ${v.pick.coreset * 100}%</td></tr>`).join("")}</tbody></table>`;
  }
  if (d.threshold && d.threshold.rules) {
    const r = d.threshold.rules[d.threshold.chosen_rule], m = d.threshold.rules.max;
    html += `<h3>Threshold rule (held-out halves, all 15 types)</h3><table><thead><tr><th>Rule</th><th>Missed</th><th>False alarms</th></tr></thead><tbody>
      <tr><td>max (before)</td><td>${m.held_out.missed}/${m.held_out.defective}</td><td>${m.held_out.false_alarms}/${m.held_out.good}</td></tr>
      <tr><td><b>${d.threshold.chosen_rule} (now)</b></td><td>${r.held_out.missed}/${r.held_out.defective}</td><td>${r.held_out.false_alarms}/${r.held_out.good}</td></tr></tbody></table>`;
  }
  $("#designBody").innerHTML = html + `</div></div>`;
}

loadTypes();
