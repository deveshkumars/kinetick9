/**
 * Record a placeholder walk animation (grid + dot) to a video blob.
 * If options.ttsBlob is provided, the dog's spoken "Arrived at..." is mixed in after the dog lands.
 */
const GRID_MIN = -10
const GRID_MAX = 10

function toPct(val, axis) {
  const range = GRID_MAX - GRID_MIN
  if (axis === 'x') return ((val - GRID_MIN) / range) * 100
  return ((GRID_MAX - val) / range) * 100
}

function runRecording(fromX, fromY, toX, toY, width, height, durationMs, audioStream, ttsDurationMs) {
  const totalMs = durationMs + (ttsDurationMs || 0)
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      reject(new Error('Canvas not supported'))
      return
    }

    const startTime = performance.now()
    const bg = '#161a20'
    const gridColor = '#2a3038'
    const dogColor = '#58a6ff'
    const borderColor = '#2a3038'

    function drawFrame(px, py) {
      ctx.fillStyle = bg
      ctx.fillRect(0, 0, width, height)
      ctx.strokeStyle = gridColor
      ctx.lineWidth = 1
      const step = width / 10
      for (let i = 0; i <= 10; i++) {
        ctx.beginPath()
        ctx.moveTo(i * step, 0)
        ctx.lineTo(i * step, height)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(0, i * step)
        ctx.lineTo(width, i * step)
        ctx.stroke()
      }
      ctx.strokeStyle = borderColor
      ctx.strokeRect(0, 0, width, height)

      const cx = (toPct(px, 'x') / 100) * width
      const cy = (toPct(py, 'y') / 100) * height
      const r = 8
      ctx.fillStyle = dogColor
      ctx.beginPath()
      ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = bg
      ctx.lineWidth = 2
      ctx.stroke()
    }

    let stream
    let recorder
    let chunks = []

    try {
      const videoStream = canvas.captureStream(30)
      if (audioStream && audioStream.getAudioTracks().length > 0) {
        stream = new MediaStream()
        videoStream.getVideoTracks().forEach((t) => stream.addTrack(t))
        audioStream.getAudioTracks().forEach((t) => stream.addTrack(t))
      } else {
        stream = videoStream
      }
      recorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9', videoBitsPerSecond: 2500000 })
      recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data)
      recorder.onstop = () => resolve(new Blob(chunks, { type: 'video/webm' }))
      recorder.onerror = () => reject(new Error('Recording failed'))
      recorder.start(100)
    } catch (err) {
      reject(err)
      return
    }

    function tick(now) {
      const elapsed = now - startTime
      const t = Math.min(elapsed / durationMs, 1)
      const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2
      const px = fromX + (toX - fromX) * ease
      const py = fromY + (toY - fromY) * ease
      drawFrame(px, py)
      if (elapsed < totalMs) {
        requestAnimationFrame(tick)
      } else {
        recorder.stop()
        stream.getTracks().forEach((track) => track.stop())
      }
    }

    requestAnimationFrame(tick)
  })
}

export function recordWalkVideo(fromX, fromY, toX, toY, width = 320, height = 320, durationMs = 2000, options = {}) {
  const { ttsBlob } = options
  if (!ttsBlob) {
    return runRecording(fromX, fromY, toX, toY, width, height, durationMs, null, 0)
  }

  return (async () => {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)()
    const arrayBuffer = await ttsBlob.arrayBuffer()
    const decoded = await audioCtx.decodeAudioData(arrayBuffer)
    const dest = audioCtx.createMediaStreamDestination()
    const source = audioCtx.createBufferSource()
    source.buffer = decoded
    source.connect(dest)
    source.start(durationMs / 1000)
    const ttsDurationMs = decoded.duration * 1000
    try {
      return await runRecording(fromX, fromY, toX, toY, width, height, durationMs, dest.stream, ttsDurationMs)
    } finally {
      audioCtx.close()
    }
  })()
}
