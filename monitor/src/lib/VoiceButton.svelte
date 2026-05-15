<script>
  // Floating voice button — mic → Whisper STT → LLM → TTS → play
  // States: idle | recording | processing | playing | error

  let voiceState = $state('idle')   // idle | recording | processing | playing | error
  let lastText   = $state('')       // last transcription
  let lastReply  = $state('')       // last LLM reply
  let errMsg     = $state('')

  let mediaRecorder = null
  let chunks        = []
  let micStream     = null

  // ── WAV encoder ──────────────────────────────────────────────────────────
  function writeStr(view, offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i))
  }

  function audioBufferToWav(buf) {
    const sr        = buf.sampleRate
    const samples   = buf.getChannelData(0)       // mono
    const dataBytes = samples.length * 2
    const ab        = new ArrayBuffer(44 + dataBytes)
    const v         = new DataView(ab)
    writeStr(v, 0,  'RIFF');  v.setUint32(4,  36 + dataBytes, true)
    writeStr(v, 8,  'WAVE');  writeStr(v, 12, 'fmt ')
    v.setUint32(16, 16,  true);  v.setUint16(20, 1,    true)  // PCM
    v.setUint16(22, 1,   true);  v.setUint32(24, sr,   true)  // mono, sampleRate
    v.setUint32(28, sr * 2, true); v.setUint16(32, 2,  true)  // byteRate, blockAlign
    v.setUint16(34, 16,  true);  writeStr(v, 36, 'data')
    v.setUint32(40, dataBytes, true)
    let off = 44
    for (let i = 0; i < samples.length; i++, off += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]))
      v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
    }
    return new Blob([ab], { type: 'audio/wav' })
  }

  // ── Recording ─────────────────────────────────────────────────────────────
  async function startRecording() {
    errMsg = ''
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch (e) {
      errMsg = 'Mic access denied'
      voiceState = 'error'
      return
    }
    chunks = []
    mediaRecorder = new MediaRecorder(micStream)
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data) }
    mediaRecorder.onstop = handleStop
    mediaRecorder.start()
    voiceState = 'recording'
  }

  function stopRecording() {
    if (mediaRecorder?.state !== 'inactive') mediaRecorder?.stop()
    micStream?.getTracks().forEach(t => t.stop())
  }

  // ── Pipeline: decode → STT → chat → TTS → play ───────────────────────────
  async function handleStop() {
    voiceState = 'processing'
    try {
      // Decode browser audio → 16kHz WAV
      const blob = new Blob(chunks, { type: mediaRecorder.mimeType || 'audio/webm' })
      const ab   = await blob.arrayBuffer()
      const ctx  = new AudioContext({ sampleRate: 16000 })
      const audioBuf = await ctx.decodeAudioData(ab)
      await ctx.close()
      const wavBlob = audioBufferToWav(audioBuf)

      // STT
      const fd = new FormData()
      fd.append('file', wavBlob, 'recording.wav')
      const sttR = await fetch('/v1/audio/transcriptions', { method: 'POST', body: fd })
      if (!sttR.ok) throw new Error(`STT ${sttR.status}`)
      const sttJ  = await sttR.json()
      lastText = sttJ.text?.trim() || ''
      if (!lastText) { voiceState = 'idle'; return }

      // LLM (non-streaming for simplicity)
      const chatR = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'Auto',
          messages: [
            { role: 'system', content:
              'You are a helpful voice assistant. Respond concisely — your reply will be read aloud. ' +
              'Avoid markdown, bullet lists, and code blocks unless explicitly asked. ' +
              'Keep answers under 3 sentences when possible.' },
            { role: 'user', content: lastText }
          ],
          thinking: false
        })
      })
      if (!chatR.ok) throw new Error(`Chat ${chatR.status}`)
      const chatJ = await chatR.json()
      lastReply = chatJ.choices?.[0]?.message?.content?.trim() || ''
      if (!lastReply) { voiceState = 'idle'; return }

      // TTS
      const ttsR = await fetch('/v1/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'tts', input: lastReply, voice: 'af_kore' })
      })
      if (!ttsR.ok) throw new Error(`TTS ${ttsR.status}`)
      const ttsBlob = await ttsR.blob()
      const ttsUrl  = URL.createObjectURL(ttsBlob)
      const audio   = new Audio(ttsUrl)
      voiceState = 'playing'
      audio.onended = () => { URL.revokeObjectURL(ttsUrl); voiceState = 'idle' }
      audio.onerror = () => { URL.revokeObjectURL(ttsUrl); voiceState = 'idle' }
      audio.play()
    } catch (e) {
      errMsg = e.message
      voiceState = 'error'
    }
  }

  // ── Button click handler ──────────────────────────────────────────────────
  function handleClick() {
    if      (voiceState === 'idle'  || voiceState === 'error') startRecording()
    else if (voiceState === 'recording') stopRecording()
    // processing / playing: ignore
  }

  // ── Derived helpers ───────────────────────────────────────────────────────
  const icon = $derived({
    idle:       '🎤',
    recording:  '⏹',
    processing: '⏳',
    playing:    '🔊',
    error:      '⚠',
  }[voiceState] ?? '🎤')

  const label = $derived({
    idle:       'Click to speak',
    recording:  'Click to stop',
    processing: 'Processing…',
    playing:    'Playing…',
    error:      errMsg || 'Error — click to retry',
  }[voiceState] ?? '')
</script>

<!-- Floating button + bubble -->
<div class="voice-root">
  {#if lastText || lastReply}
    <div class="bubble">
      {#if lastText}
        <p class="q"><span class="label">You</span> {lastText}</p>
      {/if}
      {#if lastReply}
        <p class="a"><span class="label">Agent</span> {lastReply}</p>
      {/if}
    </div>
  {/if}

  <button
    class="fab {voiceState}"
    onclick={handleClick}
    title={label}
    disabled={voiceState === 'processing' || voiceState === 'playing'}
  >
    {icon}
  </button>
  <span class="fab-label">{label}</span>
</div>

<style>
  .voice-root {
    position: fixed;
    bottom: 1.5rem;
    right: 1.5rem;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: .5rem;
    z-index: 100;
  }

  .bubble {
    background: #1e2230;
    border: 1px solid #ffffff12;
    border-radius: .75rem;
    padding: .6rem .8rem;
    max-width: 22rem;
    font-size: .75rem;
    line-height: 1.45;
    display: flex;
    flex-direction: column;
    gap: .35rem;
    box-shadow: 0 4px 24px #0006;
  }
  .q, .a { margin: 0; color: #c8ccd8; }
  .label {
    font-weight: 700;
    font-size: .68rem;
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-right: .35rem;
  }
  .q .label { color: #4e9af1; }
  .a .label { color: #4ef1a0; }

  .fab {
    width: 3.25rem;
    height: 3.25rem;
    border-radius: 50%;
    border: none;
    font-size: 1.4rem;
    cursor: pointer;
    background: #1e2230;
    border: 1.5px solid #ffffff18;
    box-shadow: 0 4px 18px #0008;
    transition: background .15s, transform .1s;
    display: flex; align-items: center; justify-content: center;
  }
  .fab:hover:not(:disabled) { transform: scale(1.06); }
  .fab:disabled { cursor: default; }

  .fab.idle     { background: #1e2230; }
  .fab.recording {
    background: #3a1515;
    border-color: #f1544e88;
    animation: ring 1.2s ease-in-out infinite;
  }
  .fab.processing { background: #1a2035; }
  .fab.playing  { background: #152a1e; border-color: #4ef1a088; }
  .fab.error    { background: #2a1515; border-color: #f1544e; }

  .fab-label {
    font-size: .68rem;
    opacity: .45;
    text-align: right;
    max-width: 10rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  @keyframes ring {
    0%,100% { box-shadow: 0 0 0 0    #f1544e55, 0 4px 18px #0008; }
    50%      { box-shadow: 0 0 0 10px #f1544e00, 0 4px 18px #0008; }
  }
</style>
