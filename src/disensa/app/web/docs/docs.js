(function () {
  "use strict";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // highlight the section being read in the index
  const links = [...document.querySelectorAll(".toc a")];
  const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const targets = [...byId.keys()].map((id) => document.getElementById(id)).filter(Boolean);
  function update() {
    let current = targets[0];
    for (const t of targets) { if (t.getBoundingClientRect().top <= 120) current = t; }
    if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 4) current = targets[targets.length - 1];
    links.forEach((a) => a.classList.toggle("active", a === byId.get(current.id)));
  }
  window.addEventListener("scroll", update, { passive: true });
  update();
  // phone: the index folds away
  const toc = document.getElementById("toc"), toggle = document.getElementById("toc-toggle");
  toggle.addEventListener("click", () => { const open = !toc.classList.contains("open"); toc.classList.toggle("open", open); toggle.setAttribute("aria-expanded", String(open)); });
  links.forEach((a) => a.addEventListener("click", () => { toc.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); }));
})();
