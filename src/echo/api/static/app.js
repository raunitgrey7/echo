/* echo dashboard: vanilla JS, no build step. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = { job: null, segments: [], activeIndex: -1 };
  const SPEAKER_COLORS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"];

  function apiKey() { return localStorage.getItem("echo.apiKey") || $("api-key").value || ""; }
  function headers() { const h = {}; const k = apiKey(); if (k) h["X-API-Key"] = k; return h; }
  async function api(path, options) {
    const res = await fetch(path, Object.assign({ headers: headers() }, options || {}));
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) { /* plain body */ }
      throw new Error(res.status + ": " + detail);
    }
    return res;
  }
  function fmt(t) {
    t = Math.max(0, t || 0);
    const m = Math.floor(t / 60), s = Math.floor(t % 60);
    return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
  }
  function speakerColor(name, speakers) {
    const i = Math.max(0, speakers.indexOf(name));
    return "var(" + SPEAKER_COLORS[i % SPEAKER_COLORS.length] + ")";
  }
  function seek(t) { const p = $("player"); p.currentTime = Math.max(0, t); p.play().catch(() => {}); }

  async function loadHealth() {
    try {
      const h = await (await fetch("/health")).json();
      $("health").textContent = h.backend + " / " + h.model + " / " + h.jobs + " jobs";
      $("key-field").hidden = !h.auth_required;
      $("demo-btn").disabled = !h.demo_available;
    } catch (e) { $("health").textContent = "server unreachable"; }
  }

  async function loadJobs() {
    try {
      const jobs = await (await api("/v1/jobs")).json();
      const ul = $("jobs"); ul.innerHTML = "";
      if (!jobs.length) { ul.innerHTML = "<li class='empty'>No jobs yet. Upload a WAV or run the demo.</li>"; return; }
      jobs.forEach((j) => {
        const li = document.createElement("li");
        li.className = state.job && state.job.id === j.id ? "active" : "";
        li.innerHTML = "<strong>" + escapeHtml(j.title) + "</strong><small>" + fmt(j.duration) + " · " + j.n_speakers + " speaker" + (j.n_speakers === 1 ? "" : "s") + " · " + j.n_actions + " actions · " + j.backend + "</small>";
        li.onclick = () => openJob(j.id);
        ul.appendChild(li);
      });
    } catch (e) { $("jobs").innerHTML = "<li class='empty'>" + escapeHtml(e.message) + "</li>"; }
  }

  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

  async function submitJob(formData, label) {
    const btn = $("analyse-btn"); btn.disabled = true; $("demo-btn").disabled = true;
    $("progress").textContent = label + " (whisper runs on CPU, about 1 s per 15 s of audio)";
    const started = performance.now();
    try {
      if ($("api-key").value) localStorage.setItem("echo.apiKey", $("api-key").value);
      const res = formData ? await api("/v1/jobs", { method: "POST", body: formData }) : await api("/v1/demo", { method: "POST" });
      const job = await res.json();
      $("progress").textContent = "Done in " + ((performance.now() - started) / 1000).toFixed(1) + " s (pipeline RTF " + job.real_time_factor.toFixed(3) + ")";
      render(job); loadJobs(); loadHealth();
    } catch (e) { $("progress").textContent = e.message; }
    finally { btn.disabled = false; $("demo-btn").disabled = false; }
  }

  $("upload-form").onsubmit = (ev) => {
    ev.preventDefault();
    const fd = new FormData();
    fd.append("file", $("file").files[0]);
    if ($("title").value) fd.append("title", $("title").value);
    submitJob(fd, "Analysing " + $("file").files[0].name);
  };
  $("demo-btn").onclick = () => submitJob(null, "Synthesising and analysing the demo meeting");

  async function openJob(id) { try { render(await (await api("/v1/jobs/" + id)).json()); loadJobs(); } catch (e) { $("progress").textContent = e.message; } }

  function render(job) {
    state.job = job; state.segments = job.transcript.segments; state.activeIndex = -1;
    $("content").hidden = false;
    $("job-title").textContent = job.title;
    $("job-meta").textContent = fmt(job.duration) + " · " + job.transcript.speakers.length + " speaker(s) · " + job.backend + " " + job.model + " · speech " + Math.round(job.speech_ratio * 100) + "% · RTF " + job.real_time_factor.toFixed(3);
    const player = $("player");
    player.src = "/v1/jobs/" + job.id + "/audio" + (apiKey() ? "" : "");
    ["srt", "vtt", "txt"].forEach((f) => { $("dl-" + f).href = "/v1/jobs/" + job.id + "/transcript?format=" + f; });
    $("dl-notes").href = "/v1/jobs/" + job.id + "/notes";
    $("dl-json").href = "/v1/jobs/" + job.id;
    renderTimeline(job); renderTranscript(job); renderNotes(job); renderActions(job);
    $("hits").innerHTML = ""; $("query").value = "";
  }

  function renderTimeline(job) {
    const tl = $("timeline"); tl.innerHTML = "";
    const dur = job.duration || 1;
    const speakers = job.transcript.speakers;
    job.transcript.segments.forEach((s) => {
      const d = document.createElement("div"); d.className = "region";
      d.style.left = (100 * s.start / dur) + "%"; d.style.width = Math.max(0.3, 100 * (s.end - s.start) / dur) + "%";
      d.style.background = speakerColor(s.speaker, speakers); d.title = s.speaker + " " + fmt(s.start);
      tl.appendChild(d);
    });
    job.keywords.slice(0, 8).forEach((k) => k.timestamps.forEach((t) => {
      const m = document.createElement("div"); m.className = "kw"; m.style.left = (100 * t / dur) + "%"; m.title = k.phrase + " at " + fmt(t); tl.appendChild(m);
    }));
    const cur = document.createElement("div"); cur.className = "cursor"; cur.id = "cursor"; tl.appendChild(cur);
    tl.onclick = (ev) => { const r = tl.getBoundingClientRect(); seek(dur * (ev.clientX - r.left) / r.width); };
    const legend = $("legend"); legend.innerHTML = "";
    speakers.forEach((sp) => { const e = document.createElement("span"); e.innerHTML = "<i class='swatch' style='background:" + speakerColor(sp, speakers) + "'></i>" + escapeHtml(sp); legend.appendChild(e); });
    const kw = document.createElement("span"); kw.className = "kwlist"; kw.innerHTML = "Keywords: ";
    job.keywords.slice(0, 8).forEach((k) => { const e = document.createElement("span"); e.textContent = k.phrase + " (" + k.count + ")"; e.onclick = () => seek(k.timestamps[0]); kw.appendChild(e); });
    legend.appendChild(kw);
  }

  function renderTranscript(job) {
    const box = $("transcript"); box.innerHTML = "";
    const speakers = job.transcript.speakers;
    job.transcript.segments.forEach((s, i) => {
      const row = document.createElement("div"); row.className = "seg"; row.id = "seg-" + i;
      const words = s.words.length ? s.words.map((w, j) => "<span class='word' data-start='" + w.start + "' data-end='" + w.end + "' id='w-" + i + "-" + j + "'>" + escapeHtml(w.text) + "</span>").join(" ") : escapeHtml(s.text);
      row.innerHTML = "<span class='t'>" + fmt(s.start) + "</span><span class='who' style='color:" + speakerColor(s.speaker, speakers) + "'>" + escapeHtml(s.speaker) + "</span><span class='txt'>" + words + "</span>";
      row.onclick = (ev) => { const w = ev.target.closest(".word"); seek(w ? parseFloat(w.dataset.start) : s.start); };
      box.appendChild(row);
    });
  }

  function renderNotes(job) {
    const n = job.notes, box = $("notes"); box.innerHTML = "";
    if (!n.summary.length) { box.innerHTML = "<p class='empty'>No sentences to summarise.</p>"; return; }
    const head = document.createElement("p"); head.innerHTML = "<strong>Key terms:</strong> " + escapeHtml(n.title_terms.join(", ")) + " · " + n.n_sentences + " sentences, " + Math.round(n.compression * 100) + "% kept"; box.appendChild(head);
    const addList = (title, bullets) => {
      const h = document.createElement("h3"); h.textContent = title; box.appendChild(h);
      const ul = document.createElement("ul");
      bullets.forEach((b) => { const li = document.createElement("li"); li.innerHTML = "<span class='ts'>" + fmt(b.start) + "</span>" + escapeHtml(b.text); li.onclick = () => seek(b.start); ul.appendChild(li); });
      box.appendChild(ul);
    };
    addList("Summary", n.summary);
    n.sections.forEach((sec) => addList(sec.title + " (" + fmt(sec.start) + " to " + fmt(sec.end) + ")", sec.bullets));
  }

  function renderActions(job) {
    const tbody = $("actions").querySelector("tbody"); tbody.innerHTML = "";
    if (!job.actions.length) { tbody.innerHTML = "<tr><td colspan='4' class='empty'>No action items detected.</td></tr>"; return; }
    job.actions.forEach((a) => {
      const tr = document.createElement("tr");
      tr.innerHTML = "<td>" + fmt(a.start) + "</td><td>" + escapeHtml(a.owner) + "</td><td>" + escapeHtml(a.due || "") + "</td><td>" + escapeHtml(a.text) + "<br><small class='empty'>" + escapeHtml(a.speaker + " · cues: " + a.cues.join(", ")) + "</small></td>";
      tr.onclick = () => seek(a.start); tbody.appendChild(tr);
    });
  }

  $("search-form").onsubmit = async (ev) => {
    ev.preventDefault();
    if (!state.job) return;
    try {
      const data = await (await api("/v1/jobs/" + state.job.id + "/search?q=" + encodeURIComponent($("query").value) + "&k=8")).json();
      const ol = $("hits"); ol.innerHTML = "";
      if (!data.hits.length) ol.innerHTML = "<li class='empty'>No matches.</li>";
      data.hits.forEach((h) => { const li = document.createElement("li"); li.innerHTML = "<span class='ts'>" + fmt(h.start) + "</span><strong>" + escapeHtml(h.speaker) + ":</strong> " + escapeHtml(h.text) + "<span class='score'>score " + h.score.toFixed(2) + "</span>"; li.onclick = () => seek(h.start); ol.appendChild(li); });
    } catch (e) { $("hits").innerHTML = "<li class='empty'>" + escapeHtml(e.message) + "</li>"; }
  };

  document.querySelectorAll(".tab").forEach((b) => b.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".panel").forEach((p) => p.hidden = p.id !== "panel-" + b.dataset.tab);
  });

  $("player").addEventListener("timeupdate", () => {
    const t = $("player").currentTime, dur = state.job ? state.job.duration : 0;
    const cur = $("cursor"); if (cur && dur) cur.style.left = (100 * t / dur) + "%";
    let idx = -1;
    for (let i = 0; i < state.segments.length; i++) { const s = state.segments[i]; if (t >= s.start && t < s.end + 0.15) { idx = i; break; } }
    if (idx !== state.activeIndex) {
      document.querySelectorAll(".seg.active").forEach((e) => e.classList.remove("active"));
      if (idx >= 0) { const el = $("seg-" + idx); el.classList.add("active"); el.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
      state.activeIndex = idx;
    }
    document.querySelectorAll(".word.active").forEach((e) => e.classList.remove("active"));
    if (idx >= 0) {
      const s = state.segments[idx];
      for (let j = 0; j < s.words.length; j++) { if (t >= s.words[j].start && t <= s.words[j].end + 0.1) { const w = $("w-" + idx + "-" + j); if (w) w.classList.add("active"); break; } }
    }
  });

  loadHealth(); loadJobs();
})();
