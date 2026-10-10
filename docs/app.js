/**
 * Cohort Monitoring Package — Interactive UI Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Feature Showcase Tab Switching
  const showcaseTabBtns = document.querySelectorAll('.tab-btn[data-tab]');
  const showcasePanels = document.querySelectorAll('.showcase-panel');

  showcaseTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');

      showcaseTabBtns.forEach(b => b.classList.remove('active'));
      showcasePanels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }
    });
  });

  // 1b. Clinical Visualization Tab Switching
  const visTabBtns = document.querySelectorAll('.vis-tab-btn[data-vis-tab]');
  const visPanels = document.querySelectorAll('.vis-tab-panel');

  visTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-vis-tab');

      visTabBtns.forEach(b => b.classList.remove('active'));
      visPanels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }
    });
  });

  // 2. Deployment Snippet Tab Switching
  const deployTabs = document.querySelectorAll('.deploy-tab[data-deploy]');
  const deploySnippets = document.querySelectorAll('.deploy-snippet');

  deployTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetSnippetId = tab.getAttribute('data-deploy');

      deployTabs.forEach(t => t.classList.remove('active'));
      deploySnippets.forEach(s => s.classList.remove('active'));

      tab.classList.add('active');
      const targetSnippet = document.getElementById(targetSnippetId);
      if (targetSnippet) {
        targetSnippet.classList.add('active');
      }
    });
  });

  // 3. Copy to Clipboard Functionality
  const copyButtons = document.querySelectorAll('.copy-btn');

  copyButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      const codeElement = btn.closest('.terminal-box, .deploy-card, .code-snippet-box')?.querySelector('code, pre');
      const textToCopy = btn.getAttribute('data-clipboard-text') || codeElement?.innerText;

      if (!textToCopy) return;

      try {
        await navigator.clipboard.writeText(textToCopy.trim());
        const originalHtml = btn.innerHTML;
        btn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
          <span style="color:#10b981;">Copied!</span>
        `;
        setTimeout(() => {
          btn.innerHTML = originalHtml;
        }, 2000);
      } catch (err) {
        console.error('Failed to copy text:', err);
      }
    });
  });

  // 4. Smooth Anchor Scrolling with Header Offset
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const targetId = this.getAttribute('href');
      if (targetId === '#' || targetId === '') return;
      const targetEl = document.querySelector(targetId);
      if (targetEl) {
        e.preventDefault();
        const headerOffset = 84;
        const elementPosition = targetEl.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

        window.scrollTo({
          top: offsetPosition,
          behavior: 'smooth'
        });
      }
    });
  });

  // 5. Scroll-spy for Top Navigation
  const navLinks = document.querySelectorAll('.nav-menu .nav-link');
  const sections = Array.from(navLinks)
    .map(link => {
      const id = link.getAttribute('href');
      if (id && id.startsWith('#') && id.length > 1) {
        const el = document.querySelector(id);
        return el ? { link, el } : null;
      }
      return null;
    })
    .filter(Boolean);

  if ('IntersectionObserver' in window && sections.length > 0) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const activeId = '#' + entry.target.id;
          navLinks.forEach(link => {
            if (link.getAttribute('href') === activeId) {
              link.classList.add('active');
            } else {
              link.classList.remove('active');
            }
          });
        }
      });
    }, {
      rootMargin: '-25% 0px -65% 0px'
    });

    sections.forEach(s => observer.observe(s.el));
  }
});
