import { useState, useEffect, useRef } from 'react'
import { getHealth, getDogState, walkTo, speak, getTtsBlob, transcribe, simulateWalkVideo, simulateDirectionWalkVideo, simulateBallWalkVideo } from './api'
import './App.css'

const GRID_MIN = -10
const GRID_MAX = 10
const GRID_SIZE = 320
const BALL_POS = { x: 1.0, y: 0.0 } // default red ball position in the scene

function formatCoord(n) {
  const num = Number(n)
  if (Number.isNaN(num)) return '?'
  return num % 1 === 0 ? String(num) : num.toFixed(2)
}

/** Digit words for speaking decimals one-by-one (e.g. "2.25" → "two point two five") */
const DIGIT_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
const INT_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

/** Format coordinate for TTS: uses words and "point" for decimals so 3.5 says "three point five" not "three fifty". */
function formatCoordForSpeech(n) {
  const num = Number(n)
  if (Number.isNaN(num)) return '?'
  const abs = Math.abs(num)
  const intPart = Math.floor(abs)
  const prefix = num < 0 ? 'negative ' : ''
  const intWord = INT_WORDS[intPart] ?? String(intPart)
  const frac = abs - intPart
  if (frac < 1e-9) return prefix + intWord
  const decStr = abs.toFixed(2).replace(/\.?0+$/, '').split('.')[1] ?? ''
  if (!decStr) return prefix + intWord
  const decWords = [...decStr].map((d) => DIGIT_WORDS[Number(d)] ?? d).join(' ')
  return `${prefix}${intWord} point ${decWords}`
}

const NUMBER_WORDS = {
  zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20,
}

/** Replace spoken number words with digits so "three, one" -> "3, 1", "negative two" -> "-2". */
function wordsToNumbers(str) {
  if (!str || typeof str !== 'string') return str
  let s = str.toLowerCase().trim()
  for (const [word, num] of Object.entries(NUMBER_WORDS)) {
    const re = new RegExp(`(?:negative|minus)\\s*\\b${word}\\b`, 'gi')
    s = s.replace(re, ` -${num} `)
  }
  for (const [word, num] of Object.entries(NUMBER_WORDS)) {
    const re = new RegExp(`\\b${word}\\b`, 'gi')
    s = s.replace(re, ` ${num} `)
  }
  return s
}

/** Parse one coordinate from normalized text (e.g. " -2.25 " → -2.25). Returns number or NaN. */
function parseOneCoord(str) {
  if (!str || typeof str !== 'string') return Number.NaN
  const n = parseFloat(str.trim())
  return Number.isNaN(n) ? Number.NaN : n
}

/** Parse transcript for two coordinates. Supports "x = 3.5, y = -1.5" (or "x equals ... y equals ..."), comma-separated pairs, or two numbers.
 *  Decimals via "N point M" or "N point M M" (e.g. "2 point 2 5" → 2.25). Returns { x, y } or null. */
function parseCoordsFromTranscript(text) {
  if (!text || typeof text !== 'string') return null
  let normalized = wordsToNumbers(text)
  // Interpret "N point M" or "N point M M ..." as one decimal (e.g. "3 point 5" → 3.5, "2 point 2 5" → 2.25)
  normalized = normalized.replace(/(-?\d+(?:\.\d+)?)\s+point\s+((?:\d+\s*)+)/gi, (_, a, b) => `${a}.${b.replace(/\s/g, '')}`)
  // "x = 3.5, y = -1.5" or "x equals 3.5 y equals negative 1.5"
  normalized = normalized.replace(/\bequals\b/gi, '=')
  const xEq = /\bx\s*=\s*(-?\d+\.?\d*)/i.exec(normalized)
  const yEq = /\by\s*=\s*(-?\d+\.?\d*)/i.exec(normalized)
  if (xEq && yEq) {
    const x = parseFloat(xEq[1])
    const y = parseFloat(yEq[1])
    if (!Number.isNaN(x) && !Number.isNaN(y)) return { x, y }
    return null
  }
  if (normalized.includes(',')) {
    const parts = normalized.split(',').map((s) => s.trim())
    if (parts.length >= 2) {
      const x = parseOneCoord(parts[0])
      const y = parseOneCoord(parts[1])
      if (!Number.isNaN(x) && !Number.isNaN(y)) return { x, y }
    }
    return null
  }
  const numbers = []
  const tokenRegex = /(?:negative|minus)\s*(\d+\.?\d*)|(-?\d+\.?\d*)/g
  let m
  while ((m = tokenRegex.exec(normalized)) !== null) {
    const val = m[1] !== undefined ? -parseFloat(m[1]) : parseFloat(m[2])
    if (!Number.isNaN(val)) numbers.push(val)
  }
  if (numbers.length >= 2) return { x: numbers[0], y: numbers[1] }
  return null
}

function App() {
  const [x, setX] = useState(0)
  const [y, setY] = useState(0)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState(null)
  const [apiStatus, setApiStatus] = useState(null)
  const [dogState, setDogState] = useState({ x: 0, y: 0, status: 'idle' })
  const [log, setLog] = useState([])
  const [ttsError, setTtsError] = useState(null)
  const [recording, setRecording] = useState(false)
  const [lastVideoUrl, setLastVideoUrl] = useState(null)
  const [videoReady, setVideoReady] = useState(false)
  const [speakingCoords, setSpeakingCoords] = useState(false)
  const [lastTranscript, setLastTranscript] = useState(null)
  const [directionWalking, setDirectionWalking] = useState(null) // 'forward' | 'backward' | null
  const [movingToBall, setMovingToBall] = useState(false)
  const logEndRef = useRef(null)
  const lastVideoUrlRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])

  useEffect(() => {
    getHealth()
      .then((d) => setApiStatus(d.status))
      .catch(() => setApiStatus('error'))
    getDogState()
      .then((s) => setDogState({ x: s.x, y: s.y, status: s.status }))
      .catch(() => {})
  }, [])

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [log])

  useEffect(() => {
    return () => {
      if (lastVideoUrlRef.current) URL.revokeObjectURL(lastVideoUrlRef.current)
    }
  }, [])

  useEffect(() => {
    setVideoReady(false)
  }, [lastVideoUrl])

  const runSpeakCommand = async () => {
    if (speakingCoords) return
    setSpeakingCoords(true)
    setError(null)
    setLastTranscript(null)
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop()
      } catch (_) {}
      mediaRecorderRef.current = null
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      chunksRef.current = []
      recorder.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data)
      const stopped = new Promise((resolve) => { recorder.onstop = resolve })
      recorder.start(100)
      mediaRecorderRef.current = recorder
      await new Promise((r) => setTimeout(r, 4000))
      recorder.stop()
      stream.getTracks().forEach((t) => t.stop())
      await stopped
      await new Promise((r) => setTimeout(r, 100))
      const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
      mediaRecorderRef.current = null
      if (blob.size < 100) {
        setError('No audio captured. Try again.')
        return
      }
      const { text } = await transcribe(blob)
      setLastTranscript(text || '')
      if (!text) {
        setError('No speech detected. Try again.')
        return
      }
      // --- Check for direction commands (with optional trailing coordinates) ---
      const lower = text.toLowerCase()
      const isForward = /(?:walk|go|move)\s*(?:forward|forwards|ahead|straight)/i.test(lower)
      const isBackward = /(?:walk|go|move)\s*(?:backward|backwards|back)/i.test(lower)
      if (isForward || isBackward) {
        // If the direction phrase is followed by coordinates
        // (e.g. "move forward to 2.5, 1.5"), treat it as a walk-to-coordinates command.
        const dirToCoordsMatch = /(?:walk|go|move)\s*(?:forward|forwards|ahead|straight|backward|backwards|back)\s+(?:to\s+)?/i.exec(text)
        if (dirToCoordsMatch) {
          const remainder = text.slice(dirToCoordsMatch.index + dirToCoordsMatch[0].length)
          const dirCoords = parseCoordsFromTranscript(remainder)
          if (dirCoords) {
            const clamp = (n) => Math.max(GRID_MIN, Math.min(GRID_MAX, n))
            const targetX = clamp(dirCoords.x)
            const targetY = clamp(dirCoords.y)
            setX(targetX)
            setY(targetY)
            setSpeakingCoords(false)
            if (!sending && !recording) {
              setLog((prev) => [...prev, { type: 'user', text: `Walk to (${formatCoord(targetX)}, ${formatCoord(targetY)})` }])
              try {
                const result = await walkTo(targetX, targetY)
                setDogState({ x: result.x, y: result.y, status: result.status })
                const arrivalText = `Arrived at (${formatCoord(result.x)}, ${formatCoord(result.y)})`
                const spokenText = `Arrived at ${formatCoordForSpeech(result.x)}, ${formatCoordForSpeech(result.y)}`
                setLog((prev) => [...prev, { type: 'dog', text: arrivalText, spokenText }])
                setTtsError(null)
                speak(spokenText).catch((err) => setTtsError(typeof err?.message === 'string' ? err.message : String(err) || 'Speech failed'))
                setRecording(true)
                simulateWalkVideo(result.x, result.y)
                  .then((blob) => {
                    if (lastVideoUrlRef.current) URL.revokeObjectURL(lastVideoUrlRef.current)
                    const url = URL.createObjectURL(blob)
                    lastVideoUrlRef.current = url
                    setLastVideoUrl(url)
                  })
                  .catch((err) => setError(`Simulation video failed: ${err.message}`))
                  .finally(() => setRecording(false))
              } catch (err) {
                setError(err.message)
              }
            }
            return
          }
        }
        // Pure direction command (no coordinates) → direction walk
        const dir = isForward ? 'forward' : 'backward'
        setSpeakingCoords(false)
        handleDirectionWalk(dir)
        return
      }
      // --- Check for "move to ball" commands ---
      const isBall = /(?:move|go|walk|run|chase|fetch|get)\s*(?:to\s*)?(?:the\s*)?ball/i.test(lower)
        || /\bfetch\b/i.test(lower)
      if (isBall) {
        setSpeakingCoords(false)
        handleMoveToBall()
        return
      }
      // --- Fall back to coordinate parsing & walk ---
      const coords = parseCoordsFromTranscript(text)
      if (coords) {
        const clamp = (n) => Math.max(GRID_MIN, Math.min(GRID_MAX, n))
        const targetX = clamp(coords.x)
        const targetY = clamp(coords.y)
        setX(targetX)
        setY(targetY)
        if (!sending && !recording) {
          setLog((prev) => [...prev, { type: 'user', text: `Walk to (${formatCoord(targetX)}, ${formatCoord(targetY)})` }])
          try {
            const result = await walkTo(targetX, targetY)
            setDogState({ x: result.x, y: result.y, status: result.status })
            const arrivalText = `Arrived at (${formatCoord(result.x)}, ${formatCoord(result.y)})`
            const spokenText = `Arrived at ${formatCoordForSpeech(result.x)}, ${formatCoordForSpeech(result.y)}`
            setLog((prev) => [...prev, { type: 'dog', text: arrivalText, spokenText }])
            setTtsError(null)
            speak(spokenText).catch((err) => setTtsError(typeof err?.message === 'string' ? err.message : String(err) || 'Speech failed'))
            setRecording(true)
            simulateWalkVideo(result.x, result.y)
              .then((blob) => {
                if (lastVideoUrlRef.current) URL.revokeObjectURL(lastVideoUrlRef.current)
                const url = URL.createObjectURL(blob)
                lastVideoUrlRef.current = url
                setLastVideoUrl(url)
              })
              .catch((err) => setError(`Simulation video failed: ${err.message}`))
              .finally(() => setRecording(false))
          } catch (err) {
            setError(err.message)
          }
        }
      } else {
        setError(`Couldn't find a command in: "${text.slice(0, 80)}${text.length > 80 ? '…' : ''}"`)
      }
    } catch (err) {
      // Friendly message for microphone permission errors
      if (err.name === 'NotAllowedError' || /permission/i.test(err.message)) {
        setError('Microphone permission denied. Click the lock/camera icon in your browser address bar, allow microphone access for this site, then reload the page.')
      } else if (err.name === 'NotFoundError') {
        setError('No microphone found. Please connect a microphone and try again.')
      } else {
        setError(err.message || 'Speak command failed')
      }
    } finally {
      setSpeakingCoords(false)
    }
  }

  const handleWalk = async (e) => {
    e.preventDefault()
    if (sending) return
    const targetX = Number(x)
    const targetY = Number(y)
    if (Number.isNaN(targetX) || Number.isNaN(targetY)) {
      setError('Enter valid numbers for X and Y')
      return
    }
    setSending(true)
    setError(null)
    const startX = dogState.x
    const startY = dogState.y
    setLog((prev) => [...prev, { type: 'user', text: `Walk to (${formatCoord(targetX)}, ${formatCoord(targetY)})` }])
    try {
      const result = await walkTo(targetX, targetY)
      setDogState({ x: result.x, y: result.y, status: result.status })
      const arrivalText = `Arrived at (${formatCoord(result.x)}, ${formatCoord(result.y)})`
      const spokenText = `Arrived at ${formatCoordForSpeech(result.x)}, ${formatCoordForSpeech(result.y)}`
      setLog((prev) => [...prev, { type: 'dog', text: arrivalText, spokenText }])
      setTtsError(null)
      speak(spokenText).catch((err) => setTtsError(typeof err?.message === 'string' ? err.message : String(err) || 'Speech failed'))
      setRecording(true)
      simulateWalkVideo(result.x, result.y)
        .then((blob) => {
          if (lastVideoUrlRef.current) URL.revokeObjectURL(lastVideoUrlRef.current)
          const url = URL.createObjectURL(blob)
          lastVideoUrlRef.current = url
          setLastVideoUrl(url)
        })
        .catch((err) => setError(`Simulation video failed: ${err.message}`))
        .finally(() => setRecording(false))
    } catch (err) {
      setError(err.message)
      setLog((prev) => [...prev, { type: 'dog', text: `Error: ${err.message}`, error: true }])
    } finally {
      setSending(false)
    }
  }

  const handleGridClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    const nx = GRID_MIN + (px / rect.width) * (GRID_MAX - GRID_MIN)
    const ny = GRID_MAX - (py / rect.height) * (GRID_MAX - GRID_MIN)
    setX(Math.round(nx * 100) / 100)
    setY(Math.round(ny * 100) / 100)
  }

  /** Walk the dog forward or backward using vector geometry. */
  const handleDirectionWalk = async (direction) => {
    if (directionWalking || sending || recording) return
    setDirectionWalking(direction)
    setError(null)
    const label = direction === 'forward' ? 'Walk forward' : 'Walk backward'
    setLog((prev) => [...prev, { type: 'user', text: label }])
    try {
      const spokenText = `Walking ${direction}`
      setTtsError(null)
      speak(spokenText).catch((err) =>
        setTtsError(typeof err?.message === 'string' ? err.message : String(err) || 'Speech failed')
      )
      const { blob, finalX, finalY } = await simulateDirectionWalkVideo(direction)

      // Update coordinates and dog position on the graph
      setX(Math.round(finalX * 100) / 100)
      setY(Math.round(finalY * 100) / 100)
      setDogState({ x: finalX, y: finalY, status: 'idle' })

      if (lastVideoUrlRef.current) URL.revokeObjectURL(lastVideoUrlRef.current)
      const url = URL.createObjectURL(blob)
      lastVideoUrlRef.current = url
      setLastVideoUrl(url)

      const arrivalText = `Arrived at (${formatCoord(finalX)}, ${formatCoord(finalY)}) after walking ${direction}`
      const arrivalSpoken = `Arrived at ${formatCoordForSpeech(finalX)}, ${formatCoordForSpeech(finalY)}`
      setLog((prev) => [...prev, { type: 'dog', text: arrivalText, spokenText: arrivalSpoken }])
      speak(arrivalSpoken).catch(() => {})
    } catch (err) {
      setError(err.message)
      setLog((prev) => [...prev, { type: 'dog', text: `Error: ${err.message}`, error: true }])
    } finally {
      setDirectionWalking(null)
    }
  }

  /** Move the dog to the red ball in the scene. */
  const handleMoveToBall = async () => {
    if (movingToBall || directionWalking || sending || recording) return
    setMovingToBall(true)
    setError(null)
    setLog((prev) => [...prev, { type: 'user', text: `Move to ball at (${formatCoord(BALL_POS.x)}, ${formatCoord(BALL_POS.y)})` }])
    try {
      const spokenText = 'Moving to the ball'
      setTtsError(null)
      speak(spokenText).catch((err) =>
        setTtsError(typeof err?.message === 'string' ? err.message : String(err) || 'Speech failed')
      )
      const { blob, finalX, finalY } = await simulateBallWalkVideo({ ballX: BALL_POS.x, ballY: BALL_POS.y })

      // Update coordinates and dog position on the graph
      setX(Math.round(finalX * 100) / 100)
      setY(Math.round(finalY * 100) / 100)
      setDogState({ x: finalX, y: finalY, status: 'idle' })

      if (lastVideoUrlRef.current) URL.revokeObjectURL(lastVideoUrlRef.current)
      const url = URL.createObjectURL(blob)
      lastVideoUrlRef.current = url
      setLastVideoUrl(url)

      const arrivalText = `Arrived at ball — (${formatCoord(finalX)}, ${formatCoord(finalY)})`
      const arrivalSpoken = `Arrived at the ball at ${formatCoordForSpeech(finalX)}, ${formatCoordForSpeech(finalY)}`
      setLog((prev) => [...prev, { type: 'dog', text: arrivalText, spokenText: arrivalSpoken }])
      speak(arrivalSpoken).catch(() => {})
    } catch (err) {
      setError(err.message)
      setLog((prev) => [...prev, { type: 'dog', text: `Error: ${err.message}`, error: true }])
    } finally {
      setMovingToBall(false)
    }
  }

  const toPx = (val, axis) => {
    const range = GRID_MAX - GRID_MIN
    if (axis === 'x') return ((val - GRID_MIN) / range) * 100
    return ((GRID_MAX - val) / range) * 100
  }

  const anyBusy = sending || recording || !!directionWalking || movingToBall || speakingCoords

  return (
    <div className="app">
      <header className="header">
        <h1>🐕 KineticK9</h1>
        <p className="tagline">Walk the dog to coordinates</p>
        {apiStatus && (
          <span className={`status status-${apiStatus}`}>
            Sim {apiStatus === 'ok' ? 'connected' : 'disconnected'}
          </span>
        )}
      </header>

      <main className="main">
        <section className="walk-view" aria-label="Walk area">
          <div
            className="grid"
            style={{ width: GRID_SIZE, height: GRID_SIZE }}
            onClick={handleGridClick}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && handleGridClick(e)}
            aria-label="Click to set target coordinates"
          >
            <div
              className="ball-marker"
              style={{
                left: `${toPx(BALL_POS.x, 'x')}%`,
                top: `${toPx(BALL_POS.y, 'y')}%`,
              }}
              title={`Ball (${BALL_POS.x}, ${BALL_POS.y})`}
            />
            <div
              className="dog-marker"
              style={{
                left: `${toPx(dogState.x, 'x')}%`,
                top: `${toPx(dogState.y, 'y')}%`,
              }}
            />
          </div>
          <p className="position-readout">
            Dog at <strong>({formatCoord(dogState.x)}, {formatCoord(dogState.y)})</strong>
          </p>
        </section>

        <section className="coords-section">
          <form className="coords-form" onSubmit={handleWalk}>
            <label className="coords-label">
              <span>X</span>
              <input
                type="number"
                value={x}
                onChange={(e) => setX(e.target.value)}
                step="any"
                min={GRID_MIN}
                max={GRID_MAX}
                className="input input-num"
              />
            </label>
            <label className="coords-label">
              <span>Y</span>
              <input
                type="number"
                value={y}
                onChange={(e) => setY(e.target.value)}
                step="any"
                min={GRID_MIN}
                max={GRID_MAX}
                className="input input-num"
              />
            </label>
            <button type="submit" className="btn btn-primary" disabled={anyBusy}>
              {recording ? 'Simulating…' : sending ? '…' : 'Walk'}
            </button>
          </form>
          <p className="coords-hint">Click the grid to set X, Y or type values. Range {GRID_MIN} to {GRID_MAX}.</p>
          <div className="speak-commands-row">
            <button type="button" className="btn btn-primary speak-command-btn" onClick={runSpeakCommand} disabled={anyBusy}>
              {speakingCoords ? 'Listening… (4 sec)' : '🎤 Speak command'}
            </button>
            <span className="coords-hint speak-command-hint">Say coordinates, &quot;walk forward&quot;, &quot;move forward to 3, 5&quot;, or &quot;move to ball&quot;</span>
          </div>

          <div className="direction-walk-section">
            <h3 className="direction-walk-title">Direction walk</h3>
            <div className="direction-walk-row">
              <button
                type="button"
                className="btn btn-direction btn-forward"
                onClick={() => handleDirectionWalk('forward')}
                disabled={anyBusy}
              >
                {directionWalking === 'forward' ? 'Walking…' : '⬆ Walk Forward'}
              </button>
              <button
                type="button"
                className="btn btn-direction btn-backward"
                onClick={() => handleDirectionWalk('backward')}
                disabled={anyBusy}
              >
                {directionWalking === 'backward' ? 'Walking…' : '⬇ Walk Backward'}
              </button>
            </div>
          </div>

          <div className="ball-walk-section">
            <h3 className="direction-walk-title">Ball chase</h3>
            <button
              type="button"
              className="btn btn-ball"
              onClick={handleMoveToBall}
              disabled={anyBusy}
            >
              {movingToBall ? 'Chasing ball…' : '🏐 Move to Ball'}
            </button>
            <p className="coords-hint" style={{ marginTop: '0.35rem' }}>
              Ball is at ({formatCoord(BALL_POS.x)}, {formatCoord(BALL_POS.y)}) — shown as 🔴 on the grid
            </p>
          </div>

          {lastTranscript != null && lastTranscript !== '' && (
            <p className="coords-hint">Heard: &ldquo;{lastTranscript.slice(0, 100)}{lastTranscript.length > 100 ? '…' : ''}&rdquo;</p>
          )}
          {error && (
            <div className="banner error">{error}</div>
          )}
          {ttsError && (
            <div className="banner error">Voice: {ttsError}</div>
          )}
        </section>

        <section className="featured-video-section" aria-label="Walk recording">
          <h2 className="log-title">
            {(recording || directionWalking || movingToBall) ? 'Running simulation…' : lastVideoUrl ? 'Quadruped simulation' : 'Simulation video'}
          </h2>
          {(recording || directionWalking || movingToBall) ? (
            <div className="video-generating">
              <div className="video-generating-spinner" aria-hidden />
              <p className="video-generating-text">Running quadruped simulation…</p>
              <p className="video-generating-sub">This may take 30-60s on first run (JAX compilation)</p>
            </div>
          ) : lastVideoUrl ? (
            <div className="own-video-wrap">
              {!videoReady && (
                <div className="video-loading-overlay" aria-busy="true">
                  <div className="video-loading-spinner" aria-hidden />
                  <span>Loading video…</span>
                </div>
              )}
              <video
                key={lastVideoUrl}
                src={lastVideoUrl}
                controls
                className="own-video"
                onCanPlay={() => setVideoReady(true)}
              />
            </div>
          ) : (
            <div className="featured-video-placeholder">
              <p className="muted">Walk the dog to see the quadruped simulation here.</p>
            </div>
          )}
        </section>

        <section className="log-section" aria-label="Walk log">
          <h2 className="log-title">Log</h2>
          <div className="log">
            {log.length === 0 ? (
              <p className="muted">Walk commands and arrivals will appear here.</p>
            ) : (
              log.map((entry, i) => (
                <div key={i} className={`log-entry log-entry-${entry.type}`}>
                  <span className="log-role">{entry.type === 'user' ? 'You' : 'Dog'}</span>
                  <span className={`log-text ${entry.error ? 'error' : ''}`}>
                    {entry.text}
                  </span>
                  {entry.type === 'dog' && !entry.error && (
                    <button type="button" className="btn-speak" onClick={() => { setTtsError(null); speak(entry.spokenText || entry.text).catch((err) => setTtsError(typeof err?.message === 'string' ? err.message : String(err) || 'Speech failed')) }} title="Speak">
                      🔊
                    </button>
                  )}
                </div>
              ))
            )}
            <div ref={logEndRef} />
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
