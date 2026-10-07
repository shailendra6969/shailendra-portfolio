/* ═══════════════════════════════════════════════════════════
   Ask my AI + portfolio interactive features
   ----------------------------------------------------------------
   Retrieval is keyword/TF-IDF over confirmed data in data/facts.json.
   No API key, no external requests except GitHub's public API.
   Works offline.
   ═══════════════════════════════════════════════════════════ */

const PA = (function () {
  'use strict';

  const CFG = {
    LLM_ENDPOINT: null,
    TOP_K: 4,
    GATE: 0.055,
    MIN_COVERAGE: 0.5,
    AUTOPLAY_QUESTION: 'What is your strongest model result?',
    RETRIEVE_MS: 320,
    GROUND_MS: 300,
    ANSWER_MS: 340
  };

  const STOP = new Set(('a an and are as at be but by can could did do does for from had has have ' +
    'he her his how i if in into is it its me my no nor not of on or our ours she so than that the ' +
    'their them then there these they this to too up us was we were what when where which who whom ' +
    'why will with would you your yours am about after before over under again once s t just don should now').split(' '));

  const ALIASES = {
    built: ['built', 'created', 'made', 'developed', 'repository', 'project'],
    build: ['built', 'created', 'repository', 'project'],
    projects: ['project', 'repository', 'notebook'],
    project: ['repository', 'notebook'],
    work: ['worked', 'experience', 'role'],
    worked: ['worked', 'experience', 'annotation'],
    experience: ['experience', 'company', 'companies', 'role', 'employer'],
    company: ['experience', 'companies'],
    companies: ['experience', 'company'],
    employer: ['experience', 'companies'],
    role: ['experience', 'worked'],
    job: ['experience', 'companies'],
    cv: ['computer', 'vision', 'video', 'manufacturing'],
    vision: ['computer', 'video', 'manufacturing', 'opencv'],
    ml: ['machine', 'learning', 'model', 'classifier'],
    ai: ['assistant', 'retrieval', 'grounding'],
    model: ['classifier', 'accuracy', 'recall', 'evaluation'],
    models: ['classifiers', 'accuracy', 'recall', 'evaluation'],
    best: ['strongest', 'accuracy', 'recall'],
    strongest: ['accuracy', 'recall', 'best'],
    result: ['accuracy', 'recall', 'evaluation'],
    results: ['accuracy', 'recall', 'evaluation'],
    cloud: ['aws', 'google', 'datasets'],
    aws: ['cloud', 'google'],
    gcp: ['cloud', 'google'],
    google: ['cloud'],
    email: ['contact', 'address'],
    contact: ['email', 'github', 'experience'],
    github: ['repository', 'notebook'],
    repo: ['repository', 'notebook'],
    skills: ['stack', 'tags', 'python', 'opencv'],
    skill: ['stack', 'tags', 'python', 'opencv'],
    stack: ['python', 'opencv', 'tensorflow'],
    churn: ['bank', 'customer', 'exited', 'prediction'],
    customers: ['customer', 'churn', 'bank'],
    video: ['manufacturing', 'hours', 'footage'],
    hours: ['10,000', 'video', 'manufacturing'],
    data: ['dataset', 'validation', 'quality', 'cleaning'],
    dataset: ['validation', 'quality', 'video'],
    quality: ['validation', 'dataset', 'cleaning'],
    validation: ['dataset', 'annotation', 'quality'],
    annotation: ['validation', 'label', 'quality'],
    tools: ['python', 'opencv', 'tensorflow', 'aws'],
    technologies: ['python', 'opencv', 'tensorflow', 'aws'],
    education: ['degree', 'university', 'college', 'pharmacy', 'coursework'],
    degree: ['education', 'university', 'pharmacy', 'bachelor'],
    agentic: ['ai', 'rag', 'langgraph', 'llm', 'llms', 'claude'],
    rag: ['retrieval', 'vector', 'search', 'langgraph', 'agentic', 'database'],
    langgraph: ['agentic', 'rag', 'llm', 'workflow'],
    llm: ['agentic', 'rag', 'ai', 'evaluation', 'models'],
    llms: ['agentic', 'rag', 'ai', 'evaluation', 'models']
  };

  const INTENTS = [
    { cat: 'experience', terms: ['experience', 'company', 'companies', 'worked', 'employer', 'role', 'career', 'job'] },
    { cat: 'computer_vision', terms: ['vision', 'cv', 'video', 'manufacturing', 'annotation', 'opencv', 'cnn', 'footage', 'hours'] },
    { cat: 'machine_learning', terms: ['model', 'models', 'classifier', 'accuracy', 'recall', 'churn', 'prediction', 'ml', 'evaluation'] },
    { cat: 'cloud', terms: ['cloud', 'aws', 'google', 'gcp', 'infrastructure'] },
    { cat: 'ai_systems', terms: ['ai', 'retrieval', 'grounding', 'refusal', 'assistant', 'agentic', 'rag', 'langgraph', 'llm'] },
    { cat: 'education', terms: ['education', 'degree', 'university', 'college', 'pharmacy', 'bachelor', 'study', 'graduated'] }
  ];

  const INTRO = {
    computer_vision: 'On the manufacturing computer vision and data quality work:',
    machine_learning: 'On the machine learning side:',
    cloud: 'On infrastructure:',
    ai_systems: 'On AI systems:',
    experience: 'On experience:',
    analytics: 'On analytics:',
    education: 'On education and background:'
  };

  const REFUSAL_DEFAULT = 'That is not something I have built or can show evidence for yet.';

  function tokenize(s) {
    return String(s || '')
      .toLowerCase()
      .replace(/[^a-z0-9+,]+/g, ' ')
      .split(' ')
      .filter(Boolean);
  }

  function contentTerms(s) {
    return tokenize(s).filter(t => !STOP.has(t) && t.length > 1);
  }

  function buildIndex(facts) {
    const docs = facts.map(function (f) {
      const text = [f.claim, (f.tags || []).join(' '), f.category || '', f.skill_group || '', f.id].join(' ');
      return { fact: f, tokens: tokenize(text) };
    });
    const df = Object.create(null);
    docs.forEach(function (d) {
      new Set(d.tokens).forEach(function (t) { df[t] = (df[t] || 0) + 1; });
    });
    const N = docs.length || 1;
    docs.forEach(function (d) {
      const tf = Object.create(null);
      d.tokens.forEach(function (t) { tf[t] = (tf[t] || 0) + 1; });
      const w = Object.create(null);
      let sum = 0;
      Object.keys(tf).forEach(function (t) {
        const idf = Math.log((N + 1) / ((df[t] || 0) + 1)) + 1;
        const v = (1 + Math.log(tf[t])) * idf;
        w[t] = v; sum += v * v;
      });
      d.w = w;
      d.norm = Math.sqrt(sum) || 1;
    });
    return { docs: docs, df: df, N: N };
  }

  function intentsFor(hard) {
    const bonus = Object.create(null);
    INTENTS.forEach(function (it) {
      if (it.terms.some(function (t) { return hard.indexOf(t) !== -1; })) bonus[it.cat] = 1.35;
    });
    return bonus;
  }

  function scoreAll(index, question) {
    const hard = contentTerms(question);
    const q = Object.create(null);
    hard.forEach(function (t) {
      q[t] = (q[t] || 0) + 1;
      (ALIASES[t] || []).forEach(function (a) {
        if (!q[a]) q[a] = (q[a] || 0) + 0.6;
      });
    });
    let qn = 0;
    Object.keys(q).forEach(function (t) { qn += q[t] * q[t]; });
    qn = Math.sqrt(qn) || 1;
    const bonus = intentsFor(hard);

    return index.docs.map(function (d) {
      let dot = 0;
      Object.keys(q).forEach(function (t) { if (d.w[t]) dot += q[t] * d.w[t]; });
      let s = dot / (qn * d.norm);
      if (!(s > 0)) s = 0;
      if (s > 0 && bonus[d.fact.category]) s *= bonus[d.fact.category];
      if (hard.indexOf(d.fact.id) !== -1) s = Math.max(s, 0.95);
      const covered = {};
      hard.forEach(function (t) {
        if (d.w[t]) { covered[t] = 1; return; }
        if ((ALIASES[t] || []).some(function (a) { return d.w[a]; })) covered[t] = 1;
      });
      const cov = hard.length ? Object.keys(covered).length / hard.length : 0;
      return { fact: d.fact, score: s, coverage: cov };
    }).sort(function (a, b) { return b.score - a.score; });
  }

  function answerGitHubQuestion(qLower, gitHubRepos, manualProjects) {
    if (!gitHubRepos || !gitHubRepos.length) return null;

    const BANNER = 'Listed from GitHub, not reviewed here.';
    const LIVE_FACT = {
      id: 'github-live',
      claim: 'Public GitHub repositories auto-discovered via GitHub API.',
      evidence_type: 'live ↗',
      link: 'https://github.com/shailendra6969'
    };

    // 1. Refusal check: questions asking about metrics/benchmarks/accuracy/architecture/deployment on unreviewed github projects
    if (/(accuracy|metric|benchmark|f1|recall|loss|precision|architecture|deployment|production usage|sla|scale|latency|client)/i.test(qLower) &&
        /(github|auto-listed|repositories|repo)/i.test(qLower) &&
        !/(churn|manufacturing|video|10,000)/i.test(qLower)) {
      return {
        refused: true,
        text: 'I do not have verified evidence for metrics, deployment, or internal architecture of unreviewed GitHub repositories. Details are available only for the hand-written case studies.',
        facts: [],
        trace: { candidates: 0, kept: 0, top: 0, reason: 'unreviewed github details withheld' }
      };
    }

    // 2. Count: "how many projects", "how many repos", "project count", "number of projects"
    if (/(how many (projects?|repos?)|project count|total (projects?|repos?)|number of (projects?|repos?))/i.test(qLower)) {
      const manualCount = manualProjects ? manualProjects.length : 4;
      const text = 'Project count breakdown:';
      const claims = [
        'I have ' + manualCount + ' verified case studies with detailed technical documentation and evidence.',
        'I have ' + gitHubRepos.length + ' public repositories auto-discovered on GitHub with the portfolio topic (' + BANNER + ').'
      ];
      return {
        refused: false,
        text: text,
        claims: claims,
        facts: [LIVE_FACT],
        via: 'github-live',
        trace: { candidates: gitHubRepos.length, kept: gitHubRepos.length, top: 1.0, reason: 'github-count' }
      };
    }

    // 3. Latest project: "latest project", "newest project", "most recent project"
    if (/(latest|newest|most recent) (project|repo)/i.test(qLower)) {
      const sorted = gitHubRepos.slice().sort(function (a, b) { return new Date(b.pushed_at || 0) - new Date(a.pushed_at || 0); });
      const newest = sorted[0];
      const name = (newest.name || '').replace(/[-_]+/g, ' ');
      const text = 'Latest project on GitHub (' + BANNER + '):';
      const claims = [
        name + (newest.language ? ' [' + newest.language + ']' : '') + ' — ' + (newest.description || 'Active repository') + ' (Pushed: ' + (newest.pushed_at ? String(newest.pushed_at).slice(0, 10) : 'recently') + ').'
      ];
      const fact = {
        id: 'github-live',
        claim: newest.description || name,
        evidence_type: 'live ↗',
        link: newest.html_url || 'https://github.com/shailendra6969'
      };
      return {
        refused: false,
        text: text,
        claims: claims,
        facts: [fact],
        via: 'github-live',
        trace: { candidates: 1, kept: 1, top: 1.0, reason: 'github-latest' }
      };
    }

    // 4. Python projects: "which projects use python", "python projects"
    if (/(python projects?|projects?.*use.*python|built with python|using python)/i.test(qLower)) {
      const pyRepos = gitHubRepos.filter(function (r) {
        return (r.language || '').toLowerCase() === 'python' || (r.topics || []).some(function (t) { return t.toLowerCase() === 'python'; });
      });
      const text = 'Projects using Python:';
      const claims = [];
      claims.push('Curated Case Studies: Manufacturing CV pipeline (TensorFlow/OpenCV), Bank Churn prediction (scikit-learn), Precision Pharma analytics, and Grammar Scoring engine.');
      if (pyRepos.length) {
        claims.push('Discovered GitHub Python repositories (' + BANNER + '): ' + pyRepos.map(function (r) { return r.name.replace(/[-_]+/g, ' '); }).join(', ') + '.');
      }
      return {
        refused: false,
        text: text,
        claims: claims,
        facts: [LIVE_FACT],
        via: 'github-live',
        trace: { candidates: pyRepos.length, kept: pyRepos.length, top: 1.0, reason: 'github-python' }
      };
    }

    // 5. JavaScript projects: "which projects use javascript", "javascript projects"
    if (/(javascript projects?|projects?.*use.*javascript|built with javascript|using javascript|js projects?)/i.test(qLower)) {
      const jsRepos = gitHubRepos.filter(function (r) {
        return (r.language || '').toLowerCase() === 'javascript' || (r.topics || []).some(function (t) { return t.toLowerCase().indexOf('javascript') !== -1; });
      });
      const text = 'Projects using JavaScript (' + BANNER + '):';
      const claims = [];
      if (jsRepos.length) {
        claims.push('Discovered GitHub JavaScript repositories: ' + jsRepos.map(function (r) { return r.name.replace(/[-_]+/g, ' '); }).join(', ') + '.');
      } else {
        claims.push('No public repositories with primary language JavaScript are currently tagged with the portfolio topic on GitHub.');
      }
      return {
        refused: false,
        text: text,
        claims: claims,
        facts: [LIVE_FACT],
        via: 'github-live',
        trace: { candidates: jsRepos.length, kept: jsRepos.length, top: 1.0, reason: 'github-javascript' }
      };
    }

    // 6. Topics: "what topics do your projects have", "project topics"
    if (/(what topics|which topics|project topics|topics.*projects?)/i.test(qLower)) {
      const topicSet = new Set();
      gitHubRepos.forEach(function (r) {
        (r.topics || []).forEach(function (t) {
          const tl = t.toLowerCase();
          if (tl !== 'portfolio' && tl !== 'featured') topicSet.add(t);
        });
      });
      const topicList = Array.from(topicSet).sort();
      const text = 'Topics across GitHub projects (' + BANNER + '):';
      const claims = [
        topicList.length ? 'Available repository topics: ' + topicList.join(', ') : 'Repositories are currently tagged with the portfolio topic.'
      ];
      return {
        refused: false,
        text: text,
        claims: claims,
        facts: [LIVE_FACT],
        via: 'github-live',
        trace: { candidates: topicList.length, kept: topicList.length, top: 1.0, reason: 'github-topics' }
      };
    }

    // 7. General list: "what projects do you have", "list projects", "what are your projects"
    if (/(what projects|list.*projects|all projects|tell me.*projects)/i.test(qLower) &&
        !/(accuracy|result|model|recall)/i.test(qLower)) {
      const text = 'Summary of projects (' + BANNER + '):';
      const claims = [
        'Verified Case Studies: Manufacturing CV pipeline, Bank Churn ML prediction, AI Decision Intelligence Platform (in development), and Precision Pharma.',
        'Auto-discovered GitHub repositories (' + gitHubRepos.length + ' repos): ' + gitHubRepos.slice(0, 8).map(function (r) {
          return r.name.replace(/[-_]+/g, ' ') + (r.language ? ' [' + r.language + ']' : '');
        }).join('; ') + (gitHubRepos.length > 8 ? ' and ' + (gitHubRepos.length - 8) + ' more.' : '.')
      ];
      return {
        refused: false,
        text: text,
        claims: claims,
        facts: [LIVE_FACT],
        via: 'github-live',
        trace: { candidates: gitHubRepos.length, kept: gitHubRepos.length, top: 1.0, reason: 'github-list' }
      };
    }

    return null;
  }

  function createAssistant(facts, refusalMessage, manualProjects) {
    const all = facts.slice();
    const confirmed = all.filter(function (f) { return f.status === 'confirmed'; });
    const index = buildIndex(all);
    const refusal = refusalMessage || REFUSAL_DEFAULT;
    let gitHubRepos = [];

    function setGitHubProjects(repos) {
      gitHubRepos = Array.isArray(repos) ? repos.slice() : [];
    }

    function ask(question) {
      const qLower = String(question || '').trim().toLowerCase();

      // Check if question asks about GitHub projects specifically
      if (gitHubRepos && gitHubRepos.length > 0) {
        const ghAnswer = answerGitHubQuestion(qLower, gitHubRepos, manualProjects);
        if (ghAnswer) {
          return ghAnswer;
        }
      }
      const ranked = scoreAll(index, question);
      const withheld = ranked[0];
      if (withheld && withheld.fact.status !== 'confirmed' && withheld.score >= CFG.GATE) {
        return {
          refused: true,
          text: refusal,
          facts: [],
          trace: {
            candidates: ranked.length,
            kept: 0,
            top: withheld.score,
            reason: 'top match withheld'
          }
        };
      }

      const kept = ranked.filter(function (r) {
        return r.fact.status === 'confirmed' &&
          r.score >= CFG.GATE &&
          r.coverage >= CFG.MIN_COVERAGE;
      }).slice(0, CFG.TOP_K);

      if (!kept.length) {
        return {
          refused: true,
          text: refusal,
          facts: [],
          trace: {
            candidates: ranked.length,
            kept: 0,
            top: ranked[0] ? ranked[0].score : 0,
            reason: 'no verified evidence found'
          }
        };
      }

      return {
        refused: false,
        text: null,
        facts: kept.map(function (k) { return k.fact; }),
        trace: {
          candidates: ranked.length,
          kept: kept.length,
          top: kept[0].score,
          reason: 'ok'
        }
      };
    }

    return { ask: ask, confirmed: confirmed, all: all, refusal: refusal, setGitHubProjects: setGitHubProjects };
  }

  function templateAnswer(facts) {
    if (!facts.length) return null;
    if (facts.length === 1) {
      return { intro: INTRO[facts[0].category] || 'Here is the evidence:', claims: [facts[0].claim] };
    }
    const order = ['computer_vision', 'machine_learning', 'cloud', 'ai_systems', 'analytics', 'experience'];
    const byCat = Object.create(null);
    facts.forEach(function (f) { (byCat[f.category] = byCat[f.category] || []).push(f); });
    const claims = [];
    order.forEach(function (c) {
      if (byCat[c]) byCat[c].forEach(function (f) { claims.push(f.claim); });
    });
    facts.forEach(function (f) { if (claims.indexOf(f.claim) === -1) claims.push(f.claim); });
    return { intro: 'Here is what the evidence supports:', claims: claims };
  }

  async function generate(question, result) {
    if (result.refused) return { text: result.text, facts: [], via: 'refusal' };
    if (result.via === 'github-live') {
      return {
        text: result.text,
        claims: result.claims || [],
        facts: result.facts || [],
        via: 'github-live'
      };
    }
    const t = templateAnswer(result.facts);
    return {
      text: t ? t.intro : result.facts[0].claim,
      claims: t ? t.claims : [],
      facts: result.facts,
      via: 'template'
    };
  }

  return {
    CFG: CFG,
    createAssistant: createAssistant,
    generate: generate
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = PA;

/* ═══════════════════════════════ BROWSER INITIALIZATION ═══════════════════════════════ */
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  (function () {
    'use strict';

    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const $ = function (s, r) { return (r || document).querySelector(s); };
    const wait = function (ms) { return new Promise(function (r) { setTimeout(r, reduce ? 0 : ms); }); };

    let paletteAsk = null;

    /* Year */
    const y = $('#year'); if (y) y.textContent = String(new Date().getFullYear());

    /* Nav sticky */
    const nav = $('#nav');
    if (nav && 'IntersectionObserver' in window) {
      const sentinel = document.createElement('div');
      sentinel.style.cssText = 'position:absolute;top:0;height:1px;width:1px';
      document.body.appendChild(sentinel);
      new IntersectionObserver(function (e) {
        nav.classList.toggle('is-stuck', !e[0].isIntersecting);
      }).observe(sentinel);
    }

    /* Scroll reveals */
    const reveals = document.querySelectorAll('.reveal');
    if (reduce || !('IntersectionObserver' in window)) {
      reveals.forEach(function (el) { el.classList.add('is-in'); });
    } else {
      const io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting) { en.target.classList.add('is-in'); io.unobserve(en.target); }
        });
      }, { rootMargin: '0px 0px -8% 0px', threshold: 0.06 });
      reveals.forEach(function (el) { io.observe(el); });
    }

    /* Failsafe reveals */
    setTimeout(function () {
      document.querySelectorAll('.hero__title .line > span').forEach(function (s) {
        if (parseFloat(getComputedStyle(s).opacity) < 0.99) {
          s.style.animation = 'none';
          s.style.opacity = '1';
          s.style.transform = 'none';
        }
      });
      document.querySelectorAll('.reveal:not(.is-in)').forEach(function (el) {
        el.classList.add('is-in');
      });
    }, 2200);

    /* Counter animation */
    function countUp(el) {
      const to = parseInt(el.getAttribute('data-count'), 10);
      if (!isFinite(to)) return;
      const fmt = function (n) { return n.toLocaleString('en-US'); };
      if (reduce) { el.textContent = fmt(to); return; }
      const dur = 900, t0 = performance.now();
      function frame(t) {
        const p = Math.min(1, (t - t0) / dur);
        const eased = 1 - Math.pow(1 - p, 3);
        el.textContent = fmt(Math.round(to * eased));
        if (p < 1) requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    }
    const counter = document.querySelector('[data-count]');
    if (counter) {
      if (!('IntersectionObserver' in window)) countUp(counter);
      else {
        const cio = new IntersectionObserver(function (e) {
          if (e[0].isIntersecting) { countUp(counter); cio.disconnect(); }
        }, { threshold: 0.4 });
        cio.observe(counter);
      }
    }

    /* Intro bar */
    (function () {
      const intro = document.getElementById('intro');
      if (!intro) return;
      if (reduce) { intro.remove(); return; }
      setTimeout(function () { intro.classList.add('is-done'); }, 940);
      setTimeout(function () { if (intro.parentNode) intro.remove(); }, 1180);
    })();

    /* ══════════════ LIGHTBOX CONTROLLER ══════════════ */
    const Lightbox = (function () {
      const modal = document.getElementById('lightbox');
      if (!modal) return { open: function () {}, register: function () {} };

      const img = document.getElementById('lbImg');
      const caption = document.getElementById('lightboxCaption');
      const scaleEl = document.getElementById('lbScale');
      const prevBtn = document.getElementById('lbPrev');
      const nextBtn = document.getElementById('lbNext');
      const zoomInBtn = document.getElementById('lbZoomIn');
      const zoomOutBtn = document.getElementById('lbZoomOut');
      const closeBtn = document.getElementById('lbClose');
      const viewport = document.getElementById('lbViewport');

      let zoomables = [];
      let currentIndex = -1;
      let scale = 1.0;
      let panX = 0, panY = 0;
      let isDragging = false;
      let startX = 0, startY = 0;
      let triggerEl = null;
      let lastTouchDist = 0;

      function updateTransform() {
        if (!img) return;
        img.style.transform = 'translate(' + panX + 'px, ' + panY + 'px) scale(' + scale + ')';
        if (scaleEl) scaleEl.textContent = scale.toFixed(1) + 'x';
        if (viewport) viewport.style.cursor = scale > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default';
      }

      function setScale(newScale) {
        scale = Math.max(1.0, Math.min(3.0, newScale));
        if (scale === 1.0) { panX = 0; panY = 0; }
        updateTransform();
      }

      function showItem(idx) {
        if (idx < 0 || idx >= zoomables.length) return;
        currentIndex = idx;
        const target = zoomables[currentIndex];
        const src = target.getAttribute('data-zoom') || target.getAttribute('src');
        const cap = target.getAttribute('data-caption') || target.getAttribute('alt') || '';

        if (img) {
          img.src = src;
          img.alt = cap;
        }
        if (caption) caption.textContent = cap;
        setScale(1.0);
      }

      function open(el) {
        refreshZoomables();
        triggerEl = el || document.activeElement;
        const found = zoomables.indexOf(el);
        currentIndex = found !== -1 ? found : 0;
        modal.hidden = false;
        document.body.style.overflow = 'hidden';
        showItem(currentIndex);
        modal.focus();
        document.addEventListener('keydown', onKeyDown, true);
      }

      function close() {
        if (modal.hidden) return;
        modal.hidden = true;
        document.body.style.overflow = '';
        document.removeEventListener('keydown', onKeyDown, true);
        setScale(1.0);
        if (triggerEl && triggerEl.focus) {
          try { triggerEl.focus({ preventScroll: true }); } catch (e) { triggerEl.focus(); }
        }
      }

      function next() {
        if (!zoomables.length) return;
        showItem((currentIndex + 1) % zoomables.length);
      }

      function prev() {
        if (!zoomables.length) return;
        showItem((currentIndex - 1 + zoomables.length) % zoomables.length);
      }

      function onKeyDown(e) {
        if (modal.hidden) return;
        if (e.key === 'Escape') { e.preventDefault(); close(); return; }
        if (e.key === 'ArrowRight') { e.preventDefault(); next(); return; }
        if (e.key === 'ArrowLeft') { e.preventDefault(); prev(); return; }
        if (e.key === '+' || e.key === '=') { e.preventDefault(); setScale(scale + 0.3); return; }
        if (e.key === '-' || e.key === '_') { e.preventDefault(); setScale(scale - 0.3); return; }

        /* Focus trap */
        if (e.key === 'Tab') {
          const focusables = Array.prototype.slice.call(modal.querySelectorAll('button:not([disabled])'));
          if (!focusables.length) return;
          const first = focusables[0];
          const last = focusables[focusables.length - 1];
          if (e.shiftKey && document.activeElement === first) {
            e.preventDefault(); last.focus();
          } else if (!e.shiftKey && document.activeElement === last) {
            e.preventDefault(); first.focus();
          }
        }
      }

      /* Viewport wheel zoom */
      if (viewport) {
        viewport.addEventListener('wheel', function (e) {
          e.preventDefault();
          const delta = e.deltaY < 0 ? 0.25 : -0.25;
          setScale(scale + delta);
        }, { passive: false });

        /* Viewport drag / pan */
        viewport.addEventListener('mousedown', function (e) {
          if (scale <= 1.0) return;
          isDragging = true;
          startX = e.clientX - panX;
          startY = e.clientY - panY;
          updateTransform();
        });

        window.addEventListener('mousemove', function (e) {
          if (!isDragging) return;
          panX = e.clientX - startX;
          panY = e.clientY - startY;
          updateTransform();
        });

        window.addEventListener('mouseup', function () {
          if (!isDragging) return;
          isDragging = false;
          updateTransform();
        });

        /* Touch support: pinch zoom & pan */
        viewport.addEventListener('touchstart', function (e) {
          if (e.touches.length === 2) {
            const dx = e.touches[0].clientX - e.touches[1].clientX;
            const dy = e.touches[0].clientY - e.touches[1].clientY;
            lastTouchDist = Math.hypot(dx, dy);
          } else if (e.touches.length === 1 && scale > 1.0) {
            isDragging = true;
            startX = e.touches[0].clientX - panX;
            startY = e.touches[0].clientY - panY;
          }
        }, { passive: true });

        viewport.addEventListener('touchmove', function (e) {
          if (e.touches.length === 2) {
            const dx = e.touches[0].clientX - e.touches[1].clientX;
            const dy = e.touches[0].clientY - e.touches[1].clientY;
            const dist = Math.hypot(dx, dy);
            if (lastTouchDist > 0) {
              const diff = dist - lastTouchDist;
              setScale(scale + diff * 0.01);
            }
            lastTouchDist = dist;
          } else if (e.touches.length === 1 && isDragging) {
            panX = e.touches[0].clientX - startX;
            panY = e.touches[0].clientY - startY;
            updateTransform();
          }
        }, { passive: true });

        viewport.addEventListener('touchend', function () {
          isDragging = false;
          lastTouchDist = 0;
        });

        /* Double-click toggle */
        viewport.addEventListener('dblclick', function () {
          setScale(scale > 1.2 ? 1.0 : 2.0);
        });
      }

      if (zoomInBtn) zoomInBtn.addEventListener('click', function () { setScale(scale + 0.3); });
      if (zoomOutBtn) zoomOutBtn.addEventListener('click', function () { setScale(scale - 0.3); });
      if (prevBtn) prevBtn.addEventListener('click', prev);
      if (nextBtn) nextBtn.addEventListener('click', next);
      if (closeBtn) closeBtn.addEventListener('click', close);
      modal.querySelectorAll('[data-close]').forEach(function (el) {
        el.addEventListener('click', close);
      });

      function refreshZoomables() {
        zoomables = Array.prototype.slice.call(document.querySelectorAll('[data-zoom]'));
      }

      function register() {
        refreshZoomables();
        zoomables.forEach(function (el) {
          if (el._lbBound) return;
          el._lbBound = true;
          el.style.cursor = 'zoom-in';
          el.setAttribute('tabindex', '0');
          el.setAttribute('role', 'button');
          el.setAttribute('aria-label', (el.getAttribute('data-caption') || 'View image') + ' in full size');
          el.addEventListener('click', function (e) {
            e.preventDefault();
            open(el);
          });
          el.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              open(el);
            }
          });
        });
      }

      return { open: open, close: close, register: register };
    })();

    /* ══════════════ HERO SCANNER ══════════════ */
    (function () {
      const stage = document.getElementById('scanStage');
      if (!stage) return;
      const countEl = document.getElementById('scanCount');
      const phaseEl = document.getElementById('scanPhase');
      const captionEl = document.getElementById('scanCaption');
      const COLS = 4, ROWS = 3, N = COLS * ROWS;

      let tiles = [];
      let rejectedCount = 0;
      let timers = [], running = false, inView = false;
      let seen = 0, acc = 0, rej = 0;

      const pad2 = function (n) { return (n < 10 ? '0' : '') + n; };
      const setCount = function () { if (countEl) countEl.textContent = pad2(seen) + '/' + N; };
      const say = function (s) { if (phaseEl) phaseEl.textContent = s; };
      const clearTimers = function () { timers.forEach(clearTimeout); timers = []; };
      const later = function (fn, ms) { timers.push(setTimeout(fn, ms)); };

      fetch('assets/scanner/labels.json')
        .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
        .then(function (data) {
          const frames = (data && data.frames) || [];
          initScannerTiles(frames);
        })
        .catch(function () {
          initScannerTiles([]);
        });

      function initScannerTiles(frames) {
        stage.innerHTML = '';
        tiles = [];
        rejectedCount = 0;
        const frag = document.createDocumentFragment();

        for (let i = 0; i < N; i++) {
          const frame = frames[i] || {};
          const frameNum = pad2(i + 1);
          const isBad = frame.status === 'REJECTED' || (!frames.length && (i === 2 || i === 5 || i === 10));
          if (isBad) rejectedCount++;

          const t = document.createElement('div');
          t.className = 'scan' + (isBad ? ' is-bad' : '');

          // Image lazy loader with fallback to CSS shapes
          const imgSrc = 'assets/scanner/frame-' + frameNum + '.webp';
          const img = document.createElement('img');
          img.className = 'scan__img';
          img.src = imgSrc;
          img.alt = 'QC frame ' + frameNum;
          img.loading = 'lazy';
          img.onerror = function () { this.remove(); };
          t.appendChild(img);

          // Bounding box from normalized coordinates
          const box = document.createElement('span');
          box.className = 'scan__box';
          if (frame.bbox) {
            box.style.left = (frame.bbox.x * 100) + '%';
            box.style.top = (frame.bbox.y * 100) + '%';
            box.style.width = (frame.bbox.w * 100) + '%';
            box.style.height = (frame.bbox.h * 100) + '%';
          }
          t.appendChild(box);

          const tag = document.createElement('span');
          tag.className = 'scan__tag mono';
          tag.textContent = isBad ? 'REJECTED' : 'ACCEPTED';
          t.appendChild(tag);

          const idn = document.createElement('span');
          idn.className = 'scan__id mono';
          idn.textContent = frame.id || ('f' + ('00' + (i + 1)).slice(-3));
          t.appendChild(idn);

          // Make tile zoomable in lightbox
          t.setAttribute('data-zoom', imgSrc);
          t.setAttribute('data-caption', 'Inspection ' + (frame.id || frameNum) + ' · ' + (frame.class || (isBad ? 'defect' : 'normal')) + ' · ' + (isBad ? 'REJECTED' : 'ACCEPTED'));

          frag.appendChild(t);
          tiles.push(t);
        }
        stage.appendChild(frag);
        Lightbox.register();

        if (reduce) staticView();
        else start();
      }

      function staticView() {
        tiles.forEach(function (t) { t.classList.add('is-hit', 'is-tagged'); });
        seen = N; acc = N - rejectedCount; rej = rejectedCount;
        setCount(); say('static view');
      }

      function cycle() {
        clearTimers();
        tiles.forEach(function (t) { t.classList.remove('is-hit', 'is-tagged'); });
        seen = acc = rej = 0; setCount();
        stage.classList.remove('is-scanning');
        void stage.offsetWidth;
        stage.classList.add('is-scanning');
        say('scan pass');

        const SWEEP = 2200, colMs = SWEEP / COLS;
        for (let i = 0; i < N; i++) {
          (function (i) {
            const at = (i % COLS) * colMs + 140;
            later(function () { if (tiles[i]) tiles[i].classList.add('is-hit'); }, at);
            later(function () {
              const t = tiles[i];
              if (!t) return;
              t.classList.add('is-tagged');
              seen++;
              if (t.classList.contains('is-bad')) rej++; else acc++;
              setCount();
              say('ok ' + pad2(acc) + ' \u00b7 no ' + pad2(rej));
            }, at + 420);
          })(i);
        }
        later(function () { stage.classList.remove('is-scanning'); say('pass complete'); }, SWEEP + 700);
        later(function () {
          tiles.forEach(function (t) { t.classList.remove('is-hit', 'is-tagged'); });
          seen = acc = rej = 0; setCount();
          say('standby');
        }, SWEEP + 3300);
        later(cycle, SWEEP + 4200);
      }

      function stop() {
        clearTimers();
        stage.classList.remove('is-scanning');
        running = false;
      }
      function start() {
        if (running || reduce) return;
        running = true; cycle();
      }

      if ('IntersectionObserver' in window) {
        const sio = new IntersectionObserver(function (e) {
          inView = e[0].isIntersecting;
          if (inView) start(); else stop();
        }, { threshold: 0.15 });
        sio.observe(stage);
        document.addEventListener('visibilitychange', function () {
          if (document.hidden) stop();
          else if (inView) start();
        });
      }
    })();

    /* ══════════════ COMMAND PALETTE ══════════════ */
    (function () {
      const pal = document.getElementById('palette');
      const form = document.getElementById('paletteForm');
      const input = document.getElementById('paletteInput');
      const box = document.getElementById('paletteChips');
      const btn = document.getElementById('paletteBtn');
      if (!pal || !form || !input || !box) return;
      let lastFocus = null, isOpen = false;

      Array.prototype.forEach.call(document.querySelectorAll('#chips .chip'), function (c) {
        box.appendChild(c.cloneNode(true));
      });

      function focusables() {
        return Array.prototype.slice.call(pal.querySelectorAll('button, input, a[href]'))
          .filter(function (e) { return e.getClientRects().length > 0; });
      }
      function onKey(e) {
        if (e.key === 'Escape') { e.preventDefault(); close(); return; }
        if (e.key !== 'Tab') return;
        const f = focusables();
        if (!f.length) return;
        const first = f[0], last = f[f.length - 1];
        if (!pal.contains(document.activeElement)) { e.preventDefault(); first.focus(); return; }
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
      function open() {
        if (isOpen) return;
        isOpen = true;
        lastFocus = document.activeElement;
        pal.hidden = false;
        requestAnimationFrame(function () { pal.classList.add('is-open'); });
        input.value = '';
        document.addEventListener('keydown', onKey, true);
        try { input.focus({ preventScroll: true }); } catch (err) { input.focus(); }
      }
      function close() {
        if (!isOpen) return;
        isOpen = false;
        pal.classList.remove('is-open');
        document.removeEventListener('keydown', onKey, true);
        if (reduce) pal.hidden = true;
        else setTimeout(function () { pal.hidden = true; }, 260);
        if (lastFocus && lastFocus.focus) {
          try { lastFocus.focus({ preventScroll: true }); } catch (err) { lastFocus.focus(); }
        }
      }
      function ask(q) {
        close();
        const sec = document.getElementById('ask-my-ai');
        if (sec && sec.scrollIntoView) {
          try { sec.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' }); }
          catch (err) { sec.scrollIntoView(); }
        }
        if (paletteAsk) setTimeout(function () { paletteAsk(q); }, reduce ? 0 : 450);
      }

      document.addEventListener('keydown', function (e) {
        if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
          e.preventDefault();
          if (isOpen) close(); else open();
        }
      });
      if (btn) btn.addEventListener('click', open);
      Array.prototype.forEach.call(pal.querySelectorAll('[data-close]'), function (el) {
        el.addEventListener('click', close);
      });
      form.addEventListener('submit', function (e) {
        e.preventDefault();
        const q = input.value.trim();
        if (!q) { input.focus(); return; }
        input.value = '';
        ask(q);
      });
      box.addEventListener('click', function (e) {
        const b = e.target.closest('.chip');
        if (b) ask(b.textContent.trim());
      });
    })();

    /* ══════════════ SCROLLYTELLING: CV PIPELINE ══════════════ */
    (function () {
      const wrap = document.getElementById('cvScrolly');
      if (!wrap) return;
      const steps = Array.prototype.slice.call(wrap.querySelectorAll('.pipe-step'));
      if (!steps.length) return;
      let ticking = false;

      function paint(prog) {
        wrap.style.setProperty('--prog', prog.toFixed(4));
        const active = prog <= 0 ? -1 : Math.min(steps.length - 1, Math.floor(prog * steps.length));
        steps.forEach(function (s, i) {
          s.classList.toggle('is-active', i === active);
          s.classList.toggle('is-done', i < active);
        });
      }
      function update() {
        ticking = false;
        const r = wrap.getBoundingClientRect();
        const vh = window.innerHeight || 800;
        const prog = (vh * 0.8 - r.top) / (r.height + vh * 0.3);
        paint(Math.max(0, Math.min(1, prog)));
      }
      if (reduce) { paint(1); return; }
      window.addEventListener('scroll', function () {
        if (!ticking) { ticking = true; requestAnimationFrame(update); }
      }, { passive: true });
      window.addEventListener('resize', update);
      update();
    })();

    /* ══════════════ ACTIVE NAV HIGHLIGHT ══════════════ */
    (function () {
      if (!('IntersectionObserver' in window)) return;
      const map = {};
      const list = Array.prototype.slice.call(document.querySelectorAll('.nav__links a[href^="#"]'));
      if (!list.length) return;
      list.forEach(function (a) { map[a.getAttribute('href').slice(1)] = a; });
      let current = null;
      const io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          const a = map[e.target.id];
          if (!a || a === current) return;
          if (current) current.classList.remove('is-current');
          a.classList.add('is-current');
          current = a;
        });
      }, { rootMargin: '-45% 0px -50% 0px', threshold: 0 });
      Object.keys(map).forEach(function (id) {
        const s = document.getElementById(id);
        if (s) io.observe(s);
      });
    })();

    /* ══════════════ MAGNETIC BUTTON ══════════════ */
    (function () {
      if (reduce || !window.matchMedia) return;
      if (!matchMedia('(pointer:fine)').matches) return;
      document.querySelectorAll('[data-magnetic]').forEach(function (el) {
        let raf = 0, mx = 0, my = 0;
        function apply() {
          raf = 0;
          el.style.transform = 'translate(' + mx.toFixed(1) + 'px,' + my.toFixed(1) + 'px)';
        }
        el.addEventListener('pointermove', function (e) {
          const r = el.getBoundingClientRect();
          mx = ((e.clientX - (r.left + r.width / 2)) / r.width) * 14;
          my = ((e.clientY - (r.top + r.height / 2)) / r.height) * 10;
          if (!raf) raf = requestAnimationFrame(apply);
        });
        el.addEventListener('pointerleave', function () { mx = 0; my = 0; el.style.transform = ''; });
        el.addEventListener('pointercancel', function () { el.style.transform = ''; });
      });
    })();

    /* ══════════════ DEMO CLIP CONTROLLER ══════════════ */
    function initDemoClip() {
      // Intentionally idle: CV demonstration uses verified high-impact engineering visual frames
    }

    /* ══════════════ ARCHITECTURE DIAGRAMS ══════════════ */
    function initArchitecture(archList) {
      if (!Array.isArray(archList)) return;
      archList.forEach(function (arch) {
        const target = document.getElementById(arch.case_id === 'case-cv' ? 'arch-cv' : 'arch-ml');
        if (!target || !arch.nodes || !arch.nodes.length) return;

        target.style.display = 'block';
        target.innerHTML = '';

        const head = document.createElement('div');
        head.className = 'arch-box__head';
        head.innerHTML = '<span class="mono">' + arch.title + '</span>';
        target.appendChild(head);

        const flow = document.createElement('div');
        flow.className = 'arch-flow';

        arch.nodes.forEach(function (n, idx) {
          const node = document.createElement('div');
          node.className = 'arch-node';

          const label = document.createElement('span');
          label.className = 'arch-node__label mono';
          label.textContent = n.label;
          node.appendChild(label);

          if (n.note) {
            const note = document.createElement('span');
            note.className = 'arch-node__note mono';
            note.textContent = n.note;
            node.appendChild(note);
          }
          flow.appendChild(node);

          if (idx < arch.nodes.length - 1) {
            const arr = document.createElement('span');
            arr.className = 'arch-flow__arrow mono';
            arr.setAttribute('aria-hidden', 'true');
            arr.textContent = '→';
            flow.appendChild(arr);
          }
        });
        target.appendChild(flow);
      });
    }

    /* ══════════════ EVIDENCE GALLERIES ══════════════ */
    function initEvidenceGalleries(evidenceList, confirmedFactIds) {
      if (!Array.isArray(evidenceList)) return;
      const cvContainer = document.getElementById('gallery-cv');
      const mlContainer = document.getElementById('gallery-ml');

      const cvGrid = cvContainer ? cvContainer.querySelector('.evidence-gallery__grid') : null;
      const mlGrid = mlContainer ? mlContainer.querySelector('.evidence-gallery__grid') : null;

      evidenceList.forEach(function (item) {
        if (!confirmedFactIds.has(item.fact_id)) return;
        if (!item.src || item.src.indexOf("shots/") === 0 || item.src.indexOf("assets/") !== 0) return;
        const VALID_SOURCES = ['real_data', 'public_dataset', 'illustrative'];
        if (!item.source || VALID_SOURCES.indexOf(item.source) === -1) return;
        const grid = item.case_id === 'case-cv' ? cvGrid : mlGrid;
        const container = item.case_id === 'case-cv' ? cvContainer : mlContainer;
        if (!grid || !container) return;

        // Verify image loads
        const testImg = new Image();
        testImg.onload = function () {
          container.style.display = 'block';

          const card = document.createElement('div');
          card.className = 'gallery-card';
          card.setAttribute('data-zoom', item.src);
          card.setAttribute('data-caption', item.title + ' · ' + item.caption);

          const thumb = document.createElement('div');
          thumb.className = 'gallery-card__thumb';
          const img = document.createElement('img');
          img.src = item.src;
          img.alt = item.alt || item.title;
          img.loading = 'lazy';
          thumb.appendChild(img);
          card.appendChild(thumb);

          const body = document.createElement('div');
          body.className = 'gallery-card__body';
          const title = document.createElement('h4');
          title.className = 'gallery-card__title mono';
          title.textContent = item.title;
          body.appendChild(title);

          const cap = document.createElement('p');
          cap.className = 'gallery-card__caption';
          cap.textContent = item.caption;
          body.appendChild(cap);

          card.appendChild(body);
          grid.appendChild(card);
          Lightbox.register();
        };
        testImg.src = item.src;
      });
    }

    /* ══════════════ SKILLS & PULSE ══════════════ */
    function initSkills(skillsList) {
      const container = document.getElementById('skillsContainer');
      if (!container || !Array.isArray(skillsList)) return;

      const groups = {};
      skillsList.forEach(function (s) {
        const g = s.group || 'Other';
        if (!groups[g]) groups[g] = [];
        groups[g].push(s);
      });

      container.innerHTML = '';
      Object.keys(groups).forEach(function (grpName) {
        const groupEl = document.createElement('div');
        groupEl.className = 'skillgroup';

        const title = document.createElement('h3');
        title.className = 'skillgroup__title mono';
        title.textContent = grpName;
        groupEl.appendChild(title);

        const ul = document.createElement('ul');
        ul.className = 'tags';

        groups[grpName].forEach(function (sk) {
          const li = document.createElement('li');
          li.className = 'tag-item';

          if (sk.evidence_ids && sk.evidence_ids.length > 0) {
            const a = document.createElement('a');
            a.className = 'skill-link mono';
            a.href = '#' + sk.evidence_ids[0];
            a.textContent = sk.name;
            a.setAttribute('data-target', sk.evidence_ids[0]);

            a.addEventListener('click', function (e) {
              e.preventDefault();
              const targetId = sk.evidence_ids[0];
              const targetEl = document.getElementById(targetId);
              if (targetEl) {
                targetEl.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
                targetEl.classList.remove('target-pulse');
                void targetEl.offsetWidth;
                targetEl.classList.add('target-pulse');
                setTimeout(function () { targetEl.classList.remove('target-pulse'); }, 1800);
              }
            });

            li.appendChild(a);
          } else {
            const sp = document.createElement('span');
            sp.className = 'skill-link skill-link--static mono';
            sp.textContent = sk.name;
            li.appendChild(sp);
          }

          ul.appendChild(li);
        });

        groupEl.appendChild(ul);
        container.appendChild(groupEl);
      });
    }

    /* ══════════════ EXPERIENCE ══════════════ */
    function initExperience(experienceList) {
      const list = document.getElementById('experienceList');
      const pending = document.getElementById('experiencePending');
      if (!list) return;

      const isDev = new URLSearchParams(window.location.search).get('dev') === '1';
      const valid = (experienceList || []).filter(function (xp) {
        return xp && xp.role && xp.dates && Array.isArray(xp.bullets) && xp.bullets.length > 0;
      });

      if (valid.length > 0) {
        list.innerHTML = '';
        valid.forEach(function (xp, i) {
          const li = document.createElement('li');
          li.className = 'xp__row';

          const idx = document.createElement('span');
          idx.className = 'xp__idx mono';
          idx.textContent = ('0' + (i + 1)).slice(-2);
          li.appendChild(idx);

          const main = document.createElement('div');
          main.className = 'xp__main';
          const co = document.createElement('h3');
          co.className = 'xp__co';
          co.textContent = xp.company;
          main.appendChild(co);

          const role = document.createElement('p');
          role.className = 'xp__role mono';
          role.textContent = xp.role;
          main.appendChild(role);

          if (xp.bullets && xp.bullets.length) {
            const ul = document.createElement('ul');
            ul.className = 'xp__bullets';
            xp.bullets.forEach(function (b) {
              const bli = document.createElement('li');
              bli.textContent = b;
              ul.appendChild(bli);
            });
            main.appendChild(ul);
          }
          li.appendChild(main);

          const dates = document.createElement('span');
          dates.className = 'xp__dates mono';
          dates.textContent = xp.dates;
          li.appendChild(dates);

          list.appendChild(li);
        });
        list.style.display = 'block';
        if (pending) pending.style.display = 'none';
      } else {
        list.style.display = 'none';
        if (pending) pending.style.display = isDev ? 'block' : 'none';
      }
    }

    /* ══════════════ SELECTED & GITHUB PROJECTS ══════════════ */
    function initProjects(curatedProjects, allData) {
      const grid = document.getElementById('projectsGrid');
      const buildingBlock = document.getElementById('buildingNowBlock');
      const buildingRepo = document.getElementById('buildingNowRepo');
      const buildingDesc = document.getElementById('buildingNowDesc');
      const buildingLink = document.getElementById('buildingNowLink');

      const discoveryEl = document.getElementById('githubDiscovery');
      const featuredAreaEl = document.getElementById('githubFeaturedArea');
      const featuredGridEl = document.getElementById('githubFeaturedGrid');
      const allAreaEl = document.getElementById('githubAllArea');
      const totalBadgeEl = document.getElementById('githubTotalBadge');
      const searchInput = document.getElementById('ghSearchInput');
      const langSelect = document.getElementById('ghLangFilter');
      const topicSelect = document.getElementById('ghTopicFilter');
      const clearBtn = document.getElementById('ghClearFiltersBtn');
      const emptyEl = document.getElementById('githubFilterEmpty');
      const allGridEl = document.getElementById('githubAllGrid');
      const toggleWrapEl = document.getElementById('githubToggleWrap');
      const toggleBtn = document.getElementById('githubToggleBtn');

      if (!grid) return;

      const ghCfg = (allData && allData.github) || {};
      const user = ghCfg.user || 'shailendra6969';
      const portfolioTopic = (ghCfg.topic || 'portfolio').toLowerCase();
      const featuredTopic = (ghCfg.featured_topic || 'featured').toLowerCase();
      const featuredMax = typeof ghCfg.featured_max === 'number' ? ghCfg.featured_max : 6;
      const initialVisible = typeof ghCfg.initial_visible === 'number' ? ghCfg.initial_visible : 6;
      const cacheMinutes = typeof ghCfg.cache_minutes === 'number' ? ghCfg.cache_minutes : 15;
      const hideSet = new Set((ghCfg.hide || []).map(function (h) { return String(h).toLowerCase(); }));

      const CACHE_KEY = 'pa_github_projects_cache_v4';
      const CACHE_TTL = cacheMinutes * 60 * 1000;

      // Duplicate prevention set: manual case study URLs take precedence!
      const manualRepoUrls = new Set();
      (curatedProjects || []).forEach(function (p) {
        if (p.link) manualRepoUrls.add(p.link.trim().toLowerCase().replace(/\/+$/, ''));
      });
      if (allData && Array.isArray(allData.facts)) {
        allData.facts.forEach(function (f) {
          if (f.link && f.link.indexOf('github.com') !== -1) {
            manualRepoUrls.add(f.link.trim().toLowerCase().replace(/\/+$/, ''));
          }
        });
      }

      function formatRepoName(name) {
        if (!name) return '';
        return name.replace(/[-_]+/g, ' ');
      }

      function timeAgo(dateStr) {
        if (!dateStr) return '';
        const diffMs = Date.now() - new Date(dateStr).getTime();
        const mins = Math.floor(diffMs / 60000);
        if (mins < 60) return (mins || 1) + 'm ago';
        const hrs = Math.floor(mins / 60);
        if (hrs < 24) return hrs + 'h ago';
        const days = Math.floor(hrs / 24);
        if (days < 30) return days + 'd ago';
        const months = Math.floor(days / 30);
        if (months < 12) return months + 'mo ago';
        const years = Math.floor(months / 12);
        return years + 'y ago';
      }

      /* 1. Render Hand-Written Curated Projects */
      function renderCuratedProjects(projPushes) {
        grid.innerHTML = '';
        (curatedProjects || []).forEach(function (p) {
          const card = document.createElement('article');
          card.className = 'project-card';

          const top = document.createElement('div');
          top.className = 'project-card__top';

          const titleGrp = document.createElement('div');
          titleGrp.className = 'project-card__title-grp';
          const name = document.createElement('h3');
          name.className = 'project-card__name';
          name.textContent = p.name;
          titleGrp.appendChild(name);

          if (p.subtitle) {
            const sub = document.createElement('p');
            sub.className = 'project-card__sub mono';
            sub.textContent = p.subtitle;
            titleGrp.appendChild(sub);
          }
          top.appendChild(titleGrp);

          const statusPill = document.createElement('span');
          statusPill.className = 'project-card__status-tag mono';
          statusPill.textContent = p.status || 'Active';
          top.appendChild(statusPill);
          card.appendChild(top);

          if (p.tags && p.tags.length) {
            const tagsWrap = document.createElement('div');
            tagsWrap.className = 'project-card__tags';
            p.tags.forEach(function (tagText) {
              const tagEl = document.createElement('span');
              tagEl.className = 'tag mono';
              tagEl.textContent = tagText;
              tagsWrap.appendChild(tagEl);
            });
            card.appendChild(tagsWrap);
          }

          const hideDetailsFor = ['Precision Pharma Analytics', 'Grammar Scoring Engine'];
          const shouldHideDetails = p.hide_details || hideDetailsFor.indexOf(p.name) !== -1;

          if (!shouldHideDetails) {
            const details = document.createElement('div');
            details.className = 'project-card__details';

          if (p.problem) {
            const row = document.createElement('div');
            row.className = 'project-card__row';
            const lbl = document.createElement('span');
            lbl.className = 'project-card__label mono';
            lbl.textContent = 'Problem:';
            const val = document.createElement('p');
            val.className = 'project-card__val';
            val.textContent = p.problem;
            row.appendChild(lbl);
            row.appendChild(val);
            details.appendChild(row);
          }

          if (p.solution || p.approach) {
            const row = document.createElement('div');
            row.className = 'project-card__row';
            const lbl = document.createElement('span');
            lbl.className = 'project-card__label mono';
            lbl.textContent = 'Solution & Approach:';
            const val = document.createElement('p');
            val.className = 'project-card__val';
            val.textContent = (p.solution ? p.solution + ' ' : '') + (p.approach || '');
            row.appendChild(lbl);
            row.appendChild(val);
            details.appendChild(row);
          }

          if (p.result) {
            const row = document.createElement('div');
            row.className = 'project-card__row';
            const lbl = document.createElement('span');
            lbl.className = 'project-card__label mono';
            lbl.textContent = 'Key Result:';
            const val = document.createElement('p');
            val.className = 'project-card__val';
            val.textContent = p.result;
            row.appendChild(lbl);
            row.appendChild(val);
            details.appendChild(row);
          } else if (p.description && !p.problem) {
            const desc = document.createElement('p');
            desc.className = 'project-card__desc';
            desc.textContent = p.description;
            details.appendChild(desc);
          }

            card.appendChild(details);
          }

          const foot = document.createElement('div');
          foot.className = 'project-card__foot';

          if (p.link && p.link.trim()) {
            const repoLink = document.createElement('a');
            repoLink.className = 'btn btn--ghost btn--sm mono';
            repoLink.setAttribute('href', p.link);
            repoLink.setAttribute('target', '_blank');
            repoLink.setAttribute('rel', 'noopener noreferrer');
            repoLink.textContent = 'View Repository';
            const arr = document.createElement('span');
            arr.className = 'btn__arrow';
            arr.setAttribute('aria-hidden', 'true');
            arr.textContent = '↗';
            repoLink.appendChild(arr);
            foot.appendChild(repoLink);
          } else {
            const note = document.createElement('span');
            note.className = 'project-card__status-note mono';
            note.textContent = 'In development';
            foot.appendChild(note);
          }

          if (p.demo_link && p.demo_link.trim()) {
            const demoLink = document.createElement('a');
            demoLink.className = 'btn btn--accent btn--sm mono';
            demoLink.setAttribute('href', p.demo_link);
            demoLink.setAttribute('target', '_blank');
            demoLink.setAttribute('rel', 'noopener noreferrer');
            demoLink.textContent = 'Live Demo';
            const demoArr = document.createElement('span');
            demoArr.className = 'btn__arrow';
            demoArr.setAttribute('aria-hidden', 'true');
            demoArr.textContent = '↗';
            demoLink.appendChild(demoArr);
            foot.appendChild(demoLink);
          }

          const pushDate = (projPushes || {})[p.name.toLowerCase()];
          if (pushDate) {
            const timeEl = document.createElement('span');
            timeEl.className = 'project-card__time dim mono';
            timeEl.textContent = 'Pushed ' + timeAgo(pushDate);
            foot.appendChild(timeEl);
          }

          card.appendChild(foot);
          grid.appendChild(card);
        });
      }

      /* 2. Building Now Live Block */
      function renderBuildingNow(newestRepo) {
        if (!buildingBlock) return;
        if (newestRepo && newestRepo.name && newestRepo.html_url && newestRepo.pushed_at) {
          const newestDate = new Date(newestRepo.pushed_at);
          const diffDays = Math.floor((Date.now() - newestDate.getTime()) / (1000 * 60 * 60 * 24));
          if (diffDays < 90 && newestRepo.name !== "—") {
            if (buildingRepo) buildingRepo.textContent = newestRepo.name;
            if (buildingDesc) buildingDesc.textContent = newestRepo.description || 'Active development';
            if (buildingLink) buildingLink.setAttribute('href', newestRepo.html_url);
            buildingBlock.style.display = 'block';
            return;
          }
        }
        buildingBlock.style.display = 'none';
      }

      /* 3. Helper to create Auto-Listed Project Card */
      function createAutoProjectCard(repo, isFeatured) {
        const card = document.createElement('article');
        card.className = 'project-card github-card' + (isFeatured ? ' github-card--featured' : '');

        const top = document.createElement('div');
        top.className = 'project-card__top';

        const titleGrp = document.createElement('div');
        titleGrp.className = 'project-card__title-grp';
        const nameEl = document.createElement('h3');
        nameEl.className = 'project-card__name';
        nameEl.textContent = formatRepoName(repo.name);
        titleGrp.appendChild(nameEl);

        const badgeGrp = document.createElement('div');
        badgeGrp.className = 'github-card__badges';

        if (isFeatured) {
          const featBadge = document.createElement('span');
          featBadge.className = 'tag tag--accent mono';
          featBadge.textContent = '★ Featured';
          badgeGrp.appendChild(featBadge);
        }

        const autoBadge = document.createElement('span');
        autoBadge.className = 'project-card__status-tag mono github-card__auto-badge';
        autoBadge.textContent = 'Auto-listed from GitHub';
        badgeGrp.appendChild(autoBadge);

        top.appendChild(titleGrp);
        top.appendChild(badgeGrp);
        card.appendChild(top);

        if (repo.description) {
          const desc = document.createElement('p');
          desc.className = 'project-card__desc github-card__desc';
          desc.textContent = repo.description;
          card.appendChild(desc);
        }

        const metaWrap = document.createElement('div');
        metaWrap.className = 'github-card__meta';

        if (repo.language) {
          const langBadge = document.createElement('span');
          langBadge.className = 'tag mono github-card__lang';
          langBadge.textContent = repo.language;
          metaWrap.appendChild(langBadge);
        }

        const filteredTopics = (repo.topics || []).filter(function (t) {
          const lower = String(t).toLowerCase();
          return lower !== 'portfolio' && lower !== 'featured';
        });

        filteredTopics.forEach(function (topicText) {
          const topicEl = document.createElement('span');
          topicEl.className = 'tag mono';
          topicEl.textContent = topicText;
          metaWrap.appendChild(topicEl);
        });

        if (metaWrap.childNodes.length > 0) {
          card.appendChild(metaWrap);
        }

        const foot = document.createElement('div');
        foot.className = 'project-card__foot';

        if (repo.html_url && /^https?:\/\//i.test(repo.html_url)) {
          const repoLink = document.createElement('a');
          repoLink.className = 'btn btn--ghost btn--sm mono';
          repoLink.setAttribute('href', repo.html_url);
          repoLink.setAttribute('target', '_blank');
          repoLink.setAttribute('rel', 'noopener noreferrer');
          repoLink.textContent = 'View Repository';
          const arr = document.createElement('span');
          arr.className = 'btn__arrow';
          arr.setAttribute('aria-hidden', 'true');
          arr.textContent = '↗';
          repoLink.appendChild(arr);
          foot.appendChild(repoLink);
        }

        if (repo.homepage && typeof repo.homepage === 'string') {
          const hp = repo.homepage.trim();
          if (/^https?:\/\//i.test(hp)) {
            const demoLink = document.createElement('a');
            demoLink.className = 'btn btn--accent btn--sm mono';
            demoLink.setAttribute('href', hp);
            demoLink.setAttribute('target', '_blank');
            demoLink.setAttribute('rel', 'noopener noreferrer');
            demoLink.textContent = 'Live Demo';
            const demoArr = document.createElement('span');
            demoArr.className = 'btn__arrow';
            demoArr.setAttribute('aria-hidden', 'true');
            demoArr.textContent = '↗';
            demoLink.appendChild(demoArr);
            foot.appendChild(demoLink);
          }
        }

        if (repo.pushed_at) {
          const timeEl = document.createElement('span');
          timeEl.className = 'project-card__time dim mono';
          timeEl.textContent = 'Pushed ' + timeAgo(repo.pushed_at);
          foot.appendChild(timeEl);
        }

        card.appendChild(foot);
        return card;
      }

      /* 4. Render Dynamic GitHub Discovery UI */
      function renderGitHubDiscovery(validRepos) {
        if (!discoveryEl) return;

        if (!validRepos || validRepos.length === 0) {
          discoveryEl.style.display = 'none';
          if (featuredAreaEl) featuredAreaEl.style.display = 'none';
          return;
        }

        discoveryEl.style.display = 'flex';

        // Partition: Featured vs All
        const featuredRepos = [];
        const allRepos = [];

        validRepos.forEach(function (r) {
          const topics = (r.topics || []).map(function (t) { return String(t).toLowerCase(); });
          const isFeatured = topics.indexOf(featuredTopic) !== -1;
          if (isFeatured && featuredRepos.length < featuredMax) {
            featuredRepos.push(r);
          } else {
            allRepos.push(r);
          }
        });

        // A. Featured Area
        if (featuredAreaEl && featuredGridEl) {
          if (featuredRepos.length > 0) {
            featuredGridEl.innerHTML = '';
            featuredRepos.forEach(function (r) {
              featuredGridEl.appendChild(createAutoProjectCard(r, true));
            });
            featuredAreaEl.style.display = 'block';
          } else {
            featuredAreaEl.style.display = 'none';
          }
        }

        // B. All Projects Area
        if (!allAreaEl || !allGridEl) return;

        if (totalBadgeEl) {
          totalBadgeEl.textContent = 'All Projects — ' + allRepos.length;
        }

        // Populate language and topic dropdowns
        const langSet = new Set();
        const topicSet = new Set();

        allRepos.forEach(function (r) {
          if (r.language) langSet.add(r.language);
          (r.topics || []).forEach(function (t) {
            const tl = String(t).toLowerCase();
            if (tl !== 'portfolio' && tl !== 'featured') {
              topicSet.add(t);
            }
          });
        });

        if (langSelect) {
          const curLang = langSelect.value;
          langSelect.innerHTML = '<option value="">All languages</option>';
          Array.from(langSet).sort().forEach(function (lang) {
            const opt = document.createElement('option');
            opt.value = lang;
            opt.textContent = lang;
            if (lang === curLang) opt.selected = true;
            langSelect.appendChild(opt);
          });
        }

        if (topicSelect) {
          const curTopic = topicSelect.value;
          topicSelect.innerHTML = '<option value="">All topics</option>';
          Array.from(topicSet).sort().forEach(function (top) {
            const opt = document.createElement('option');
            opt.value = top;
            opt.textContent = top;
            if (top === curTopic) opt.selected = true;
            topicSelect.appendChild(opt);
          });
        }

        let isExpanded = false;

        function updateAllGrid() {
          const q = searchInput ? searchInput.value.trim().toLowerCase() : '';
          const selLang = langSelect ? langSelect.value.trim().toLowerCase() : '';
          const selTopic = topicSelect ? topicSelect.value.trim().toLowerCase() : '';

          const isFilterActive = q.length > 0 || selLang.length > 0 || selTopic.length > 0;
          if (clearBtn) clearBtn.style.display = isFilterActive ? 'inline-flex' : 'none';

          const filtered = allRepos.filter(function (r) {
            if (q) {
              const nameMatch = (r.name || '').toLowerCase().indexOf(q) !== -1;
              const descMatch = (r.description || '').toLowerCase().indexOf(q) !== -1;
              const topicMatch = (r.topics || []).some(function (t) { return String(t).toLowerCase().indexOf(q) !== -1; });
              if (!nameMatch && !descMatch && !topicMatch) return false;
            }
            if (selLang) {
              if ((r.language || '').toLowerCase() !== selLang) return false;
            }
            if (selTopic) {
              const hasTopic = (r.topics || []).some(function (t) { return String(t).toLowerCase() === selTopic; });
              if (!hasTopic) return false;
            }
            return true;
          });

          allGridEl.innerHTML = '';

          if (filtered.length === 0) {
            if (emptyEl) emptyEl.style.display = 'block';
            allGridEl.style.display = 'none';
            if (toggleWrapEl) toggleWrapEl.style.display = 'none';
            return;
          }

          if (emptyEl) emptyEl.style.display = 'none';
          allGridEl.style.display = 'grid';

          const visibleList = isExpanded ? filtered : filtered.slice(0, initialVisible);
          visibleList.forEach(function (r) {
            allGridEl.appendChild(createAutoProjectCard(r, false));
          });

          if (toggleWrapEl && toggleBtn) {
            if (filtered.length > initialVisible) {
              toggleWrapEl.style.display = 'flex';
              if (isExpanded) {
                toggleBtn.textContent = 'Show fewer';
              } else {
                toggleBtn.textContent = 'Show all (' + filtered.length + ')';
              }
            } else {
              toggleWrapEl.style.display = 'none';
            }
          }
        }

        if (searchInput) searchInput.oninput = updateAllGrid;
        if (langSelect) langSelect.onchange = updateAllGrid;
        if (topicSelect) topicSelect.onchange = updateAllGrid;
        if (clearBtn) {
          clearBtn.onclick = function () {
            if (searchInput) searchInput.value = '';
            if (langSelect) langSelect.value = '';
            if (topicSelect) topicSelect.value = '';
            updateAllGrid();
          };
        }
        if (toggleBtn) {
          toggleBtn.onclick = function () {
            isExpanded = !isExpanded;
            updateAllGrid();
          };
        }

        updateAllGrid();
      }

      /* 5. Filter helper */
      function isValidRepo(r) {
        if (!r || r.fork === true || r.archived === true) return false;
        if (!r.description || !r.description.trim()) return false;
        const nameLower = (r.name || '').toLowerCase();
        if (hideSet.has(nameLower)) return false;
        const topics = Array.isArray(r.topics) ? r.topics.map(function (t) { return String(t).toLowerCase(); }) : [];
        if (topics.indexOf(portfolioTopic) === -1) return false;
        const repoUrl = (r.html_url || '').trim().toLowerCase().replace(/\/+$/, '');
        if (manualRepoUrls.has(repoUrl)) return false; // Hand-written takes precedence!
        return true;
      }

      /* 6. Fetching with pagination up to 5 pages */
      async function fetchAllGitHubRepos(username) {
        const all = [];
        for (let page = 1; page <= 5; page++) {
          const url = 'https://api.github.com/users/' + encodeURIComponent(username) + '/repos?per_page=100&page=' + page + '&sort=pushed';
          const res = await fetch(url);
          if (!res.ok) throw new Error('HTTP ' + res.status);
          const list = await res.json();
          if (!Array.isArray(list)) throw new Error('Invalid response');
          all.push.apply(all, list);
          if (list.length < 100) break;
        }
        return all;
      }

      // Check cache first
      let cachedData = null;
      try {
        const raw = localStorage.getItem(CACHE_KEY);
        if (raw) {
          cachedData = JSON.parse(raw);
          if (cachedData && cachedData.timestamp && (Date.now() - cachedData.timestamp < CACHE_TTL)) {
            renderCuratedProjects(cachedData.pushes || {});
            renderBuildingNow(cachedData.newest || null);
            renderGitHubDiscovery(cachedData.repos || []);
            if (typeof assistant !== 'undefined' && assistant && assistant.setGitHubProjects) {
              assistant.setGitHubProjects(cachedData.repos || []);
            }
            return;
          }
        }
      } catch (e) {}

      // Initial render of curated projects with empty pushes while loading
      renderCuratedProjects((cachedData && cachedData.pushes) || {});
      renderBuildingNow((cachedData && cachedData.newest) || null);

      // Live fetch
      fetchAllGitHubRepos(user)
        .then(function (rawRepos) {
          const pushes = {};
          rawRepos.forEach(function (r) {
            if (r.name && r.pushed_at) pushes[r.name.toLowerCase()] = r.pushed_at;
          });

          const filteredRaw = rawRepos.filter(function (r) { return !r.fork && !r.archived; });
          const newest = filteredRaw.length ? filteredRaw[0] : null;

          const validRepos = rawRepos.filter(isValidRepo);
          validRepos.sort(function (a, b) {
            return new Date(b.pushed_at || 0) - new Date(a.pushed_at || 0);
          });

          try {
            localStorage.setItem(CACHE_KEY, JSON.stringify({
              timestamp: Date.now(),
              pushes: pushes,
              newest: newest ? { name: newest.name, description: newest.description, html_url: newest.html_url, pushed_at: newest.pushed_at } : null,
              repos: validRepos
            }));
          } catch (e) {}

          renderCuratedProjects(pushes);
          renderBuildingNow(newest);
          renderGitHubDiscovery(validRepos);
          if (typeof assistant !== 'undefined' && assistant && assistant.setGitHubProjects) {
            assistant.setGitHubProjects(validRepos);
          }
        })
        .catch(function () {
          // Graceful fallback to cached data if available, even if stale
          if (cachedData && Array.isArray(cachedData.repos)) {
            renderCuratedProjects(cachedData.pushes || {});
            renderBuildingNow(cachedData.newest || null);
            renderGitHubDiscovery(cachedData.repos || []);
            if (typeof assistant !== 'undefined' && assistant && assistant.setGitHubProjects) {
              assistant.setGitHubProjects(cachedData.repos || []);
            }
          } else {
            // Cleanly hide dynamic section without console error
            renderCuratedProjects({});
            renderBuildingNow(null);
            renderGitHubDiscovery([]);
            if (typeof assistant !== 'undefined' && assistant && assistant.setGitHubProjects) {
              assistant.setGitHubProjects([]);
            }
          }
        });
    }
    
    /* ══════════════ VISUALS & HONEST LABELLING ══════════════ */
    function initVisualsAndImages(visuals) {
      const v = visuals || {};
      const VALID_SOURCES = ['real_data', 'public_dataset', 'illustrative'];

      function getSourceText(srcObj) {
        if (!srcObj) return '';
        const s = srcObj.source;
        if (s === 'illustrative') return 'Illustrative example, not employer data';
        if (s === 'public_dataset') return 'Public dataset: ' + (srcObj.dataset_name || 'Open benchmark');
        if (s === 'real_data') return 'Employer manufacturing data';
        return '';
      }

      // A. Hero Scanner
      const heroScanner = document.getElementById('heroScanner');
      const scanSrcEl = document.getElementById('scanSource');
      const scanValid = v.scanner && VALID_SOURCES.indexOf(v.scanner.source) !== -1;
      if (heroScanner) {
        if (scanValid) {
          heroScanner.style.display = '';
          if (scanSrcEl) {
            scanSrcEl.textContent = getSourceText(v.scanner);
            scanSrcEl.style.display = 'inline-block';
          }
        } else {
          heroScanner.style.display = 'none';
        }
      }

      // B. Good vs Bad Frames
      const cvFramesHead = document.getElementById('cvFramesHead');
      const cvFramesWrap = document.getElementById('cvFramesWrap');
      const gbValid = v.good_bad_frames && VALID_SOURCES.indexOf(v.good_bad_frames.source) !== -1;
      if (cvFramesWrap) {
        if (gbValid) {
          cvFramesWrap.style.display = '';
          if (cvFramesHead) cvFramesHead.style.display = '';
          const srcText = getSourceText(v.good_bad_frames);
          cvFramesWrap.querySelectorAll('figure.frame figcaption').forEach(function (fc) {
            if (!fc.querySelector('.frame__source')) {
              const sp = document.createElement('span');
              sp.className = 'frame__source mono';
              sp.textContent = srcText;
              fc.appendChild(sp);
            }
          });
        } else {
          cvFramesWrap.style.display = 'none';
          if (cvFramesHead) cvFramesHead.style.display = 'none';
        }
      }

      // C. Model Diagnostics & Explainability
      const cvVisualHead = document.getElementById('cvVisualHead');
      const cvVisualShowcase = document.getElementById('cvVisualShowcase');
      const qmValid = v.qc_metrics && v.qc_metrics.source === "real_data" && Boolean(v.qc_metrics.dataset_name) && Boolean(v.qc_metrics.model_name);
      const gcValid = v.gradcam && v.gradcam.source === "real_data" && Boolean(v.gradcam.dataset_name) && Boolean(v.gradcam.model_name);
      if (cvVisualShowcase) {
        if (qmValid || gcValid) {
          cvVisualShowcase.style.display = '';
          if (cvVisualHead) cvVisualHead.style.display = '';
          const cards = cvVisualShowcase.querySelectorAll('.visual-card');
          if (cards[0]) {
            cards[0].style.display = qmValid ? '' : 'none';
            if (qmValid && !cards[0].querySelector('.visual-card__source')) {
              const p = document.createElement('p');
              p.className = 'visual-card__source mono';
              p.textContent = getSourceText(v.qc_metrics);
              cards[0].querySelector('.visual-card__body')?.appendChild(p);
            }
          }
          if (cards[1]) {
            cards[1].style.display = gcValid ? '' : 'none';
            if (gcValid && !cards[1].querySelector('.visual-card__source')) {
              const p = document.createElement('p');
              p.className = 'visual-card__source mono';
              p.textContent = getSourceText(v.gradcam);
              cards[1].querySelector('.visual-card__body')?.appendChild(p);
            }
          }
        } else {
          cvVisualShowcase.style.display = 'none';
          if (cvVisualHead) cvVisualHead.style.display = 'none';
        }
      }
    }

    /* ══════════════ OPEN GRAPH & SEO ══════════════ */
    function initMetaTags(site) {
      if (!site || !site.url || !site.url.trim()) return;
      const rawUrl = site.url.trim().replace(/\/$/, '');
      const ogUrl = document.getElementById('ogUrl');
      const ogImg = document.getElementById('ogImage');
      const twImg = document.getElementById('twitterImage');
      if (ogUrl) ogUrl.setAttribute('content', rawUrl + '/');
      if (ogImg) ogImg.setAttribute('content', rawUrl + '/assets/og.png');
      if (twImg) twImg.setAttribute('content', rawUrl + '/assets/og.png');
    }

    /* ══════════════ PROFILE & CONTACT ══════════════ */
    function initProfile(profile) {
      const p = profile || {};
      const isDev = new URLSearchParams(window.location.search).get('dev') === '1';

      // About strip
      const bioEl = document.getElementById('aboutBio');
      const availEl = document.getElementById('aboutAvail');
      const stripEl = document.getElementById('aboutStrip');

      if (p.bio && p.bio.trim()) {
        if (bioEl) bioEl.textContent = p.bio;
      } else if (bioEl) {
        bioEl.style.display = 'none';
      }

      if (p.availability && p.availability.trim()) {
        if (availEl) availEl.textContent = p.availability;
      } else if (availEl) {
        availEl.style.display = 'none';
      }

      if ((!p.bio || !p.bio.trim()) && (!p.availability || !p.availability.trim()) && stripEl) {
        stripEl.style.display = 'none';
      }

      // GitHub
      if (p.github) {
        const ghEl = document.getElementById('contactGithub');
        const ghVal = document.getElementById('contactGithubVal');
        if (ghEl) ghEl.href = p.github;
        if (ghVal) ghVal.textContent = p.github_label || p.github.replace(/^https?:\/\//, '');
      }

      // LinkedIn
      const liEl = document.getElementById('contactLinkedin');
      const liPending = document.getElementById('contactLinkedinPending');
      const liHero = document.getElementById('heroLinkedin');
      if (p.linkedin && p.linkedin.trim()) {
        if (liEl) {
          liEl.href = p.linkedin;
          const liVal = document.getElementById('contactLinkedinVal');
          if (liVal) liVal.textContent = p.linkedin_label || p.linkedin.replace(/^https?:\/\/(www\.)?linkedin\.com\//, '');
          liEl.style.display = 'flex';
        }
        if (liPending) liPending.style.display = 'none';
        if (liHero) {
          liHero.href = p.linkedin;
          liHero.removeAttribute('data-dev-only');
          liHero.classList.remove('is-pending');
          liHero.innerHTML = 'LinkedIn<span class="btn__arrow" aria-hidden="true">↗</span>';
          liHero.style.display = 'inline-flex';
        }
      } else {
        if (liEl) liEl.style.display = 'none';
        if (liPending) liPending.style.display = isDev ? 'flex' : 'none';
        if (liHero && !isDev) liHero.style.display = 'none';
      }

      // Email
      const emailEl = document.getElementById('contactEmail');
      const emailPending = document.getElementById('contactEmailPending');
      if (p.email && p.email.trim()) {
        const emailLink = document.getElementById('contactEmailLink');
        const copyBtn = document.getElementById('copyEmailBtn');
        if (emailLink) {
          emailLink.href = 'mailto:' + encodeURIComponent(p.email) + '?subject=Inquiry%20via%20Portfolio';
          emailLink.textContent = p.email;
        }
        if (copyBtn) {
          copyBtn.onclick = function (e) {
            e.preventDefault();
            navigator.clipboard.writeText(p.email).then(function () {
              copyBtn.textContent = 'copied!';
              setTimeout(function () { copyBtn.textContent = 'copy'; }, 2000);
            }).catch(function () {});
          };
        }
        if (emailEl) emailEl.style.display = 'flex';
        if (emailPending) emailPending.style.display = 'none';
      } else {
        if (emailEl) emailEl.style.display = 'none';
        if (emailPending) emailPending.style.display = isDev ? 'flex' : 'none';
      }

      // Resume
      const resEl = document.getElementById('contactResume');
      const resPending = document.getElementById('contactResumePending');
      if (p.resume && p.resume.trim()) {
        if (resEl) {
          resEl.href = p.resume;
          resEl.style.display = 'flex';
        }
        if (resPending) resPending.style.display = 'none';
      } else {
        if (resEl) resEl.style.display = 'none';
        if (resPending) resPending.style.display = isDev ? 'flex' : 'none';
      }

      // Recruiter At-A-Glance Card bindings
      const recResumeBtn = document.getElementById('recruiterResumeBtn');
      const recLinkedinBtn = document.getElementById('recruiterLinkedinBtn');
      const recCopyEmailBtn = document.getElementById('recruiterCopyEmailBtn');

      if (recResumeBtn && p.resume) {
        recResumeBtn.href = p.resume;
      }
      if (recLinkedinBtn && p.linkedin) {
        recLinkedinBtn.href = p.linkedin;
      }
      if (recCopyEmailBtn && p.email) {
        recCopyEmailBtn.textContent = 'Copy Email: ' + p.email;
        recCopyEmailBtn.onclick = function (e) {
          e.preventDefault();
          navigator.clipboard.writeText(p.email).then(function () {
            recCopyEmailBtn.textContent = 'copied to clipboard!';
            setTimeout(function () {
              recCopyEmailBtn.textContent = 'Copy Email: ' + p.email;
            }, 2000);
          }).catch(function () {});
        };
      }
    }

    /* ══════════════ ACHIEVEMENTS ══════════════ */
    function initAchievements(achievements) {
      const el = document.getElementById('heroAchievements');
      if (!el) return;
      if (!Array.isArray(achievements) || !achievements.length) {
        el.style.display = 'none';
        return;
      }
      el.innerHTML = '';
      achievements.forEach(function (ach) {
        const pill = document.createElement('div');
        pill.className = 'achievement-pill mono';
        pill.setAttribute('role', 'listitem');

        const dot = document.createElement('span');
        dot.className = 'achievement-dot';
        dot.setAttribute('aria-hidden', 'true');
        pill.appendChild(dot);

        const txt = document.createElement('span');
        txt.textContent = ach.text;
        pill.appendChild(txt);

        el.appendChild(pill);
      });
      el.style.display = 'flex';
    }

    /* ══════════════ ASK MY AI CHAT LOGIC ══════════════ */
    const log = $('#chatLog');
    const form = $('#chatForm');
    const input = $('#chatInput');
    const chips = $('#chips');
    const factCount = $('#factCount');
    const traceList = $('#traceList');
    const traceState = $('#traceState');
    const sendBtn = form ? form.querySelector('button[type=submit]') : null;

    if (!log || !form || !input) return;

    let assistant = null;
    let busy = false;
    let autoplayed = false;

    function emptyState() {
      const p = document.createElement('p');
      p.className = 'chat__empty';
      const s = document.createElement('span');
      s.textContent = '> ready.';
      p.appendChild(s);
      p.appendChild(document.createTextNode(
        '\nRetrieval over verified evidence.\nAsk anything — answers cite verified sources.'
      ));
      log.appendChild(p);
    }

    function el(tag, cls, text) {
      const n = document.createElement(tag);
      if (cls) n.className = cls;
      if (text != null) n.textContent = text;
      return n;
    }

    function addUser(text) {
      const m = el('div', 'msg msg--user');
      m.appendChild(el('span', 'msg__who mono', 'you'));
      m.appendChild(el('div', 'msg__body', text));
      log.appendChild(m);
      scrollLog();
    }

    function addAnswer(ans) {
      const refused = ans.facts.length === 0;
      const m = el('div', 'msg ' + (refused ? 'msg--refusal' : 'msg--ai'));
      m.appendChild(el('span', 'msg__who mono', refused ? 'no evidence' : 'ai'));

      const body = el('div', 'msg__body');
      if (refused) {
        body.appendChild(el('span', 'flag mono', 'NO_EVIDENCE'));
        body.appendChild(document.createTextNode(ans.text));
      } else {
        body.appendChild(document.createTextNode(ans.text));
        if (ans.claims && ans.claims.length) {
          const ul = el('div', 'msg__list');
          ans.claims.forEach(function (c) { ul.appendChild(el('div', 'msg__item', c)); });
          body.appendChild(ul);
        }
      }
      m.appendChild(body);

      if (ans.facts.length) {
        const src = el('div', 'msg__sources');
        src.appendChild(el('span', 'msg__sources-label', 'evidence · ' + ans.facts.length));
        ans.facts.forEach(function (f) {
          const a = document.createElement('a');
          a.className = 'src';
          a.href = f.link || '#';
          if (/^https?:/.test(f.link || '')) { a.target = '_blank'; a.rel = 'noopener noreferrer'; }
          a.appendChild(el('span', 'src__id', f.id));
          a.appendChild(el('span', 'src__type', f.evidence_type));
          a.title = f.claim;
          src.appendChild(a);
        });
        m.appendChild(src);
      }

      log.appendChild(m);
      scrollLog();
    }

    function scrollLog() { log.scrollTop = log.scrollHeight; }

    function traceRow(step) { return traceList ? traceList.querySelector('[data-step=' + step + ']') : null; }
    function setTrace(state, activeStep, detail, label) {
      if (!traceList) return;
      if (traceState) traceState.textContent = state;
      traceList.querySelectorAll('.trace__row').forEach(function (r) {
        const isTarget = r.getAttribute('data-step') === activeStep;
        if (isTarget) {
          r.classList.add(label === 'refused' ? 'is-refused' : 'is-active');
          r.classList.remove('is-done');
          const s = r.querySelector('.trace__state');
          if (s) s.textContent = label || 'running';
          const d = r.querySelector('.trace__detail');
          if (d && detail) d.textContent = detail;
        }
      });
    }
    function finishTrace(step, detail, label) {
      const r = traceRow(step);
      if (!r) return;
      r.classList.remove('is-active');
      r.classList.add(label === 'refused' ? 'is-refused' : 'is-done');
      const s = r.querySelector('.trace__state');
      if (s) s.textContent = label || 'done';
      const d = r.querySelector('.trace__detail');
      if (d && detail) d.textContent = detail;
    }
    function resetTrace() {
      if (!traceList) return;
      traceList.querySelectorAll('.trace__row').forEach(function (r) {
        r.classList.remove('is-active', 'is-done', 'is-refused');
        const s = r.querySelector('.trace__state'); if (s) s.textContent = 'idle';
        const d = r.querySelector('.trace__detail'); if (d) d.textContent = '—';
      });
      if (traceState) traceState.textContent = 'running';
    }

    async function run(question, opts) {
      if (busy || !assistant) return;
      const stealFocus = !opts || opts.focus !== false;
      busy = true;
      if (sendBtn) sendBtn.disabled = true;
      if (input) input.disabled = true;

      const empty = log.querySelector('.chat__empty');
      if (empty) empty.remove();

      addUser(question);
      resetTrace();

      /* 01 Question */
      setTrace('running', 'question', question.length + ' chars', 'running');
      await wait(180);
      finishTrace('question', question.length + ' chars', 'done');

      /* 02 Retrieve */
      setTrace('running', 'retrieve', 'scoring candidate facts…', 'running');
      await wait(PA.CFG.RETRIEVE_MS);
      const result = assistant.ask(question);
      finishTrace('retrieve',
        result.trace.candidates + ' evaluated \u2192 ' + result.trace.kept + ' kept \u00b7 top ' +
        result.trace.top.toFixed(2), 'done');

      /* 03 Ground */
      setTrace('running', 'ground', result.refused ? 'checking verification…' : 'resolving sources…', 'running');
      await wait(PA.CFG.GROUND_MS);
      if (result.refused) {
        finishTrace('ground', result.trace.reason, 'refused');
      } else {
        finishTrace('ground', result.facts.length + ' verified sources', 'done');
      }

      /* 04 Answer */
      setTrace('running', 'answer', result.refused ? 'withholding' : 'composing…', 'running');
      const ans = await PA.generate(question, result);
      await wait(PA.CFG.ANSWER_MS);

      const words = (ans.text + ' ' + (ans.claims || []).join(' ')).trim().split(/\s+/).filter(Boolean).length;
      if (ans.facts.length === 0) {
        finishTrace('answer', 'refused \u00b7 0 sources', 'refused');
        if (traceState) traceState.textContent = 'refused';
      } else {
        finishTrace('answer', words + ' words \u00b7 ' + ans.facts.length + ' source' +
          (ans.facts.length === 1 ? '' : 's'), 'done');
        if (traceState) traceState.textContent = 'grounded';
      }

      addAnswer(ans);

      busy = false;
      if (sendBtn) sendBtn.disabled = false;
      if (input) {
        input.disabled = false;
        if (stealFocus && document.activeElement !== input) {
          try { input.focus({ preventScroll: true }); } catch (e) { input.focus(); }
        }
      }
    }

    paletteAsk = function (q) { run(q, { focus: false }); };

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      const q = input.value.trim();
      if (!q) return;
      input.value = '';
      run(q);
    });

    if (chips) {
      chips.addEventListener('click', function (e) {
        const b = e.target.closest('.chip');
        if (!b) return;
        run(b.textContent.trim());
      });
    }

    /* ══════════════ BOOT & DATA FETCH ══════════════ */
    fetch('data/facts.json', { cache: 'no-cache' })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (data) {
        const confirmedFacts = (data.facts || []).filter(function (f) {
          return f && f.status === 'confirmed' && f.claim;
        });
        const confirmedSet = new Set(confirmedFacts.map(function (f) { return f.id; }));

        assistant = PA.createAssistant(data.facts || [], data.refusal_message, data.projects || []);
        if (factCount) factCount.textContent = String(confirmedFacts.length);
        emptyState();

        // 0. Visuals, honest labeling & SEO
        initVisualsAndImages(data.visuals);
        initMetaTags(data.site);

        // 1. Achievements
        initAchievements(data.achievements);

        // 2. Profile & About strip
        initProfile(data.profile);

        // 3. Scanner label
        if (data.scanner && data.scanner.label) {
          const scCap = document.getElementById('scanCaption');
          if (scCap) scCap.textContent = data.scanner.label;
        }

        // 4. Architecture Diagrams
        initArchitecture(data.architecture);

        // 5. Demo Clip
        initDemoClip();

        // 6. Churn Business Note & Corrected Badge
        const churnNoteEl = document.getElementById('churnMetricCallout');
        if (churnNoteEl) {
          if (data.churn_business_note) {
            churnNoteEl.textContent = data.churn_business_note;
            churnNoteEl.style.display = 'block';
          } else {
            churnNoteEl.style.display = 'none';
          }
        }
        const churnBadgeEl = document.getElementById('churnTilesBadge');
        if (churnBadgeEl) {
          const hasCorrected = confirmedSet.has('churn-result-corrected');
          churnBadgeEl.textContent = hasCorrected ? 'corrected output' : 'pre-fix notebook output';
        }

        // 7. Evidence Galleries
        initEvidenceGalleries(data.evidence, confirmedSet);

        // 8. Experience
        initExperience(data.experience);

        // 9. Selected Projects
        initProjects(data.projects, data);

        // 10. Skills
        initSkills(data.skills);

        // Register all data-zoom triggers with Lightbox
        Lightbox.register();

        // Autoplay initial sample question
        if ('IntersectionObserver' in window) {
          const section = document.getElementById('ask-my-ai');
          const aio = new IntersectionObserver(function (entries) {
            if (autoplayed) return;
            if (entries[0].isIntersecting) {
              autoplayed = true;
              aio.disconnect();
              setTimeout(function () {
                run(PA.CFG.AUTOPLAY_QUESTION, { focus: false });
              }, reduce ? 0 : 550);
            }
          }, { threshold: 0, rootMargin: '0px 0px -20% 0px' });
          if (section) aio.observe(section);
        }
      })
      .catch(function (err) {
        if (factCount) factCount.textContent = '!';
        const p = el('p', 'chat__empty');
        p.textContent = '> data/facts.json could not be loaded (' + (err && err.message) + ')';
        log.appendChild(p);
        if (traceState) traceState.textContent = 'error';
      });
  })();
}
