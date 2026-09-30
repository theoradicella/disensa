// <site-header current="lab|docs"></site-header>: the floating header shared by every page.
// Loaded in <head> (not deferred), so the header is filled in as soon as the tag is parsed.
(function () {
  "use strict";
  const LOGO = `
    <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
      <circle cx="11" cy="11" r="2.4" fill="#104281"/>
      <circle cx="5" cy="7" r="1.6" fill="#2a78d6"/><circle cx="16.5" cy="5.5" r="1.4" fill="#2a78d6" opacity=".75"/>
      <circle cx="17" cy="15" r="1.8" fill="#2a78d6" opacity=".9"/><circle cx="6.5" cy="16.5" r="1.3" fill="#2a78d6" opacity=".6"/>
      <circle cx="11" cy="3" r="1" fill="#2a78d6" opacity=".5"/><circle cx="11.5" cy="19.5" r="1" fill="#2a78d6" opacity=".5"/>
    </svg>`;
  const GITHUB = `<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>`;
  const PYPI = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" aria-hidden="true"><path d="M12 2.8 20 7.2v9.6L12 21.2 4 16.8V7.2z"/><path d="M4 7.2 12 11.6l8-4.4M12 11.6v9.6"/></svg>`;
  const LINKS = [["lab", "/lab"], ["docs", "/docs"]];

  class SiteHeader extends HTMLElement {
    connectedCallback() {
      const current = this.getAttribute("current");
      const nav = LINKS.map(([label, href]) =>
        `<a class="nav-link" href="${href}"${label === current ? ' aria-current="page"' : ""}>${label}</a>`).join("\n");
      this.innerHTML = `
        <header class="topbar">
          <a class="mark" href="/" aria-label="Home">${LOGO}</a>
          <nav class="nav">
            ${nav}
            <span class="nav-sep" aria-hidden="true"></span>
            <a class="icon-link" href="https://github.com/theoradicella/disensa" target="_blank" rel="noopener" aria-label="GitHub repository" title="GitHub">${GITHUB}</a>
            <a class="icon-link" href="https://pypi.org/project/disensa/" target="_blank" rel="noopener" aria-label="Python package on PyPI" title="PyPI">${PYPI}</a>
          </nav>
        </header>`;
    }
  }
  customElements.define("site-header", SiteHeader);
})();
