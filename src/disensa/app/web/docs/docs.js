(function () {
  "use strict";
  // formulas: \( … \) inline, \[ … \] on their own line (KaTeX, vendored under /vendor/katex)
  if (window.renderMathInElement) {
    renderMathInElement(document.querySelector("article"), {
      delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
      throwOnError: false,
    });
  }
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // highlight the section being read in the index
  const links = [...document.querySelectorAll(".toc a")];
  const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const targets = [...byId.keys()].map((id) => document.getElementById(id)).filter(Boolean);
  const sections = [...document.querySelectorAll(".toc ol.top > li")];
  function setActive(id) {
    const active = byId.get(id);
    links.forEach((a) => a.classList.toggle("active", a === active));
    // open the top-level section that contains the active link
    sections.forEach((li) => li.classList.toggle("current", li.contains(active)));
  }
  // after a click in the index, highlight the target at once and ignore the sections
  // scrolled past on the way, until the scroll ends
  let locked = null, release = 0;
  function update() {
    if (locked) return;
    let current = targets[0];
    for (const t of targets) { if (t.getBoundingClientRect().top <= 120) current = t; }
    if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 4) current = targets[targets.length - 1];
    setActive(current.id);
  }
  function unlock() { locked = null; clearTimeout(release); update(); }
  links.forEach((a) => a.addEventListener("click", () => {
    locked = a.getAttribute("href").slice(1);
    setActive(locked);
    clearTimeout(release);
    release = setTimeout(unlock, 1200); // fallback if the scrollend event never fires
  }));
  window.addEventListener("scrollend", () => { if (locked) unlock(); });
  window.addEventListener("scroll", update, { passive: true });
  update();
  // phone: the index folds away
  const toc = document.getElementById("toc"), toggle = document.getElementById("toc-toggle");
  toggle.addEventListener("click", () => { const open = !toc.classList.contains("open"); toc.classList.toggle("open", open); toggle.setAttribute("aria-expanded", String(open)); });
  links.forEach((a) => a.addEventListener("click", () => { toc.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); }));
})();

// keep term popups inside the viewport: anchor under the term, then shift left if they would overflow
document.querySelectorAll(".term-wrap").forEach((wrap) => {
  const pop = wrap.querySelector(".term-pop");
  const place = () => {
    pop.style.left = "0px";
    const r = pop.getBoundingClientRect();
    const over = r.right - (document.documentElement.clientWidth - 16);
    const under = 16 - r.left;
    if (over > 0) pop.style.left = `${-Math.min(over, r.left - 16)}px`;
    else if (under > 0) pop.style.left = `${under}px`;
  };
  wrap.addEventListener("mouseenter", place);
  wrap.addEventListener("focusin", place);
});
