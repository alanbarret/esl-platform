/**
 * ESL Sign Language Plugin
 * - Button shows BELOW selected text instantly on selectionchange
 * - Avatar video only
 */
(function () {
  'use strict';

  // Use configured API URL, or fall back to same-origin (nginx proxies /api/ to ESL backend)
  const API   = (typeof eslConfig !== 'undefined' && eslConfig.apiUrl) ? eslConfig.apiUrl : window.location.origin;
  const AJAX  = (typeof eslConfig !== 'undefined') ? eslConfig.ajaxUrl : '';
  const NONCE = (typeof eslConfig !== 'undefined') ? eslConfig.nonce   : '';

  let selectedText = '';
  let currentData  = null;
  let triggerBtn, overlay, video, loading, errorEl, glossEl;

  // ── Build UI ──────────────────────────────────────────────────────────────
  function buildUI() {
    triggerBtn = document.createElement('button');
    triggerBtn.id = 'esl-trigger-btn';
    triggerBtn.innerHTML = '🤟 Sign Language';
    document.body.appendChild(triggerBtn);

    const div = document.createElement('div');
    div.id = 'esl-modal-overlay';
    div.innerHTML = `
      <div id="esl-modal">
        <div id="esl-modal-header">
          <div class="esl-logo">🤟</div>
          <div class="esl-header-text">
            <div class="esl-title">Emirates Sign Language</div>
            <div class="esl-subtitle" id="esl-subtitle">Generating…</div>
          </div>
          <button id="esl-modal-close" aria-label="Close">✕</button>
        </div>
        <div id="esl-selected-text"></div>
        <div id="esl-gloss"></div>
        <div id="esl-video-wrap">
          <div id="esl-loading"><div class="esl-spinner"></div><span>Generating sign language video…</span></div>
          <div id="esl-error"></div>
          <video id="esl-video" autoplay loop muted playsinline webkit-playsinline></video>
        </div>
      </div>`;
    document.body.appendChild(div);

    overlay = div;
    video   = document.getElementById('esl-video');
    loading = document.getElementById('esl-loading');
    errorEl = document.getElementById('esl-error');
    glossEl = document.getElementById('esl-gloss');
  }

  // ── Position button BELOW selection ──────────────────────────────────────
  function showBtn(rect) {
    if (!rect || rect.width === 0) return;
    const scrollY = window.scrollY || document.documentElement.scrollTop;
    const btnW = 160;
    // Centre horizontally over selection, place BELOW
    let left = rect.left + (rect.width / 2) - (btnW / 2);
    let top  = rect.bottom + scrollY + 10;          // 10px below selection
    // Clamp horizontally
    left = Math.max(8, Math.min(left, window.innerWidth - btnW - 8));
    triggerBtn.style.left    = left + 'px';
    triggerBtn.style.top     = top  + 'px';
    triggerBtn.style.display = 'flex';
  }

  function hideBtn() { triggerBtn.style.display = 'none'; }

  // ── Modal helpers ─────────────────────────────────────────────────────────
  function openModal() {
    // Deselect text before opening modal
    if (window.getSelection) window.getSelection().removeAllRanges();
    overlay.classList.add('esl-open');
    document.body.style.overflow = 'hidden';
    video.muted = true;
  }
  function closeModal() {
    overlay.classList.remove('esl-open');
    document.body.style.overflow = '';
    video.pause(); video.src = ''; video.style.display = 'none';
    loading.style.display = 'flex'; errorEl.style.display = 'none';
    glossEl.innerHTML = ''; currentData = null;
  }
  function showLoading() {
    loading.style.display = 'flex'; video.style.display = 'none'; errorEl.style.display = 'none';
  }
  function showErr(msg) {
    loading.style.display = 'none'; video.style.display = 'none';
    errorEl.textContent = '⚠️ ' + msg; errorEl.style.display = 'block';
  }
  function playVideo(url) {
    loading.style.display = 'none'; errorEl.style.display = 'none';
    video.src = url; video.style.display = 'block';
    var p = video.play(); if (p) p.catch(function(){});
  }

  // ── Translate ─────────────────────────────────────────────────────────────
  async function translate(text) {
    showLoading(); glossEl.innerHTML = '';
    document.getElementById('esl-subtitle').textContent = 'Generating…';
    currentData = null;

    var data = null;

    // 1. Direct API
    try {
      var r = await fetch(API + '/api/v1/translate', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({text: text}),
      });
      if (!r.ok) throw new Error('API ' + r.status);
      var body = await r.json();
      data = { tokens: body.gloss_tokens||[], avatar_url: body.avatar_video_url||null, video_url: body.video_url||null, base: API };
    } catch(e) {}

    // 2. WP AJAX fallback
    if (!data) {
      try {
        var fd = new FormData();
        fd.append('action','esl_translate'); fd.append('nonce',NONCE); fd.append('text',text);
        var r2 = await fetch(AJAX, {method:'POST', body:fd});
        var b2 = await r2.json();
        if (!b2.success) throw new Error(b2.data||'Failed');
        var d = b2.data;
        data = { tokens: d.tokens||[], avatar_url: d.avatar_video_url||null, video_url: d.video_url||null, base: d.api_base||API };
      } catch(e2) { showErr(e2.message||'Could not reach ESL Platform'); return; }
    }

    currentData = data;

    // Show gloss
    if (data.tokens.length) {
      glossEl.innerHTML = data.tokens.map(function(t){ return '<span class="esl-token">'+t+'</span>'; }).join('');
      document.getElementById('esl-subtitle').textContent = data.tokens.length + ' sign' + (data.tokens.length!==1?'s':'');
    }

    // Play avatar (prefer avatar, fallback to skeleton)
    var url = data.avatar_url ? (data.base + data.avatar_url) : (data.video_url ? (data.base + data.video_url) : null);
    if (url) playVideo(url);
    else showErr('No video available.');
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', function() {
    buildUI();

    // ── Key fix: use selectionchange for instant button show ──────────────
    var selTimer;
    document.addEventListener('selectionchange', function() {
      clearTimeout(selTimer);
      selTimer = setTimeout(function() {
        if (overlay.classList.contains('esl-open')) return;
        var sel = window.getSelection();
        var text = sel ? sel.toString().trim() : '';
        if (text.length < 3 || sel.rangeCount === 0) { hideBtn(); return; }
        selectedText = text;
        var rect = sel.getRangeAt(0).getBoundingClientRect();
        showBtn(rect);
      }, 150);
    });

    // Also on mouseup/touchend for extra reliability
    function onEnd() {
      if (overlay.classList.contains('esl-open')) return;
      setTimeout(function() {
        var sel = window.getSelection();
        var text = sel ? sel.toString().trim() : '';
        if (text.length < 3 || !sel || sel.rangeCount === 0) { hideBtn(); return; }
        selectedText = text;
        showBtn(sel.getRangeAt(0).getBoundingClientRect());
      }, 200);
    }
    document.addEventListener('mouseup',  onEnd);
    document.addEventListener('touchend', onEnd);

    // Hide button when clicking elsewhere
    document.addEventListener('mousedown', function(e) {
      if (e.target !== triggerBtn && !overlay.contains(e.target)) hideBtn();
    });
    document.addEventListener('touchstart', function(e) {
      if (e.target !== triggerBtn && !overlay.contains(e.target)) hideBtn();
    }, {passive:true});

    // Trigger
    triggerBtn.addEventListener('click', function() {
      hideBtn();
      document.getElementById('esl-selected-text').textContent = '\u201c' + selectedText + '\u201d';
      openModal();
      translate(selectedText);
    });

    // Close
    document.getElementById('esl-modal-close').addEventListener('click', closeModal);
    overlay.addEventListener('click', function(e){ if(e.target===overlay) closeModal(); });
    document.addEventListener('keydown', function(e){ if(e.key==='Escape') closeModal(); });
  });
})();
