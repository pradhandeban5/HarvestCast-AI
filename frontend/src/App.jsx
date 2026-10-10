import { useEffect, useRef, useState } from 'react'
import {
  Activity, ArrowDownRight, ArrowUpRight, Boxes, Camera, ChevronDown, CircleHelp,
  CircleStop, Cloud, Command, Gauge, Leaf, Radio, RefreshCw, ScanLine, Settings2,
  ShieldCheck, Snowflake, Thermometer, Wind,
} from 'lucide-react'
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/v1/ws/telemetry'

const seedBatches = [
  { id: 1, batch_id: 'HC-7A30D8F1', produce: 'Avocados', chamber_id: 'COLD-02', vigor: 0.73, water_ratio: 0.61, lambda_days: 8.4, k_shape: 1.48, expected_rsl_days: 7.55, dispatch_action: 'DOMESTIC_COLD_TRANSIT', measured_at: new Date().toISOString() },
  { id: 2, batch_id: 'HC-B40EF629', produce: 'Tomatoes', chamber_id: 'COLD-01', vigor: 0.81, water_ratio: 0.54, lambda_days: 11.2, k_shape: 2.1, expected_rsl_days: 9.9, dispatch_action: 'EXPORT_GRADE_LONG_HOLD', measured_at: new Date(Date.now() - 5 * 60000).toISOString() },
  { id: 3, batch_id: 'HC-31D2A001', produce: 'Spinach', chamber_id: 'COLD-03', vigor: 0.38, water_ratio: 0.82, lambda_days: 3.6, k_shape: 1.25, expected_rsl_days: 3.35, dispatch_action: 'URGENT_LOCAL_DISPATCH', measured_at: new Date(Date.now() - 11 * 60000).toISOString() },
]

const fallbackChambers = [
  { chamber_id: 'COLD-01', temperature_c: 5.8, humidity_pct: 91.2, fan_pct: 54, setpoint_c: 8.5, target_humidity_pct: 85, target_fan_pct: 45 },
  { chamber_id: 'COLD-02', temperature_c: 6.3, humidity_pct: 89.6, fan_pct: 61, setpoint_c: 6, target_humidity_pct: 90, target_fan_pct: 60 },
  { chamber_id: 'COLD-03', temperature_c: 4.7, humidity_pct: 92.4, fan_pct: 84, setpoint_c: 3.5, target_humidity_pct: 92, target_fan_pct: 90 },
]

const survivalCurve = (batch) => Array.from({ length: 17 }, (_, index) => {
  const day = index * 0.75
  return { day, survival: Math.exp(-((day / batch.lambda_days) ** batch.k_shape)) * 100 }
})

const formatAge = (date) => {
  if (!date) return 'Awaiting first scan'
  const minutes = Math.max(0, Math.floor((Date.now() - new Date(date).getTime()) / 60000))
  return minutes < 1 ? 'Just now' : `${minutes}m ago`
}

const actionMeta = (action) => {
  if (action === 'URGENT_LOCAL_DISPATCH') return { label: 'Urgent dispatch', tone: 'urgent' }
  if (action === 'EXPORT_GRADE_LONG_HOLD') return { label: 'Export hold', tone: 'hold' }
  return { label: 'Cold transit', tone: 'transit' }
}

function App() {
  const [batches, setBatches] = useState(seedBatches)
  const [chambers, setChambers] = useState(fallbackChambers)
  const [selectedChamber, setSelectedChamber] = useState('COLD-02')
  const [connection, setConnection] = useState('connecting')
  const [lastSeen, setLastSeen] = useState(new Date().toISOString())
  const [optimizing, setOptimizing] = useState(false)
  const [cameraActive, setCameraActive] = useState(false)
  const [cameraBusy, setCameraBusy] = useState(false)
  const [scanBusy, setScanBusy] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const [produceType, setProduceType] = useState('Tomato')
  const [modelStatus, setModelStatus] = useState({ status: 'loading', checkpoint_available: false, classes: [] })
  const scanCanvas = useRef(null)
  const videoRef = useRef(null)
  const cameraStream = useRef(null)
  const captureCanvas = useRef(null)
  const latest = batches[0]
  const selected = chambers.find((chamber) => chamber.chamber_id === selectedChamber) || chambers[0]
  const selectedBatch = batches.find((batch) => batch.chamber_id === selectedChamber) || latest
  const chambersOnTarget = chambers.filter((chamber) => Math.abs(chamber.temperature_c - chamber.setpoint_c) <= 2).length

  useEffect(() => {
    let socket
    let heartbeat
    let reconnectTimer
    let cancelled = false
    const connect = () => {
      if (cancelled) return
      socket = new WebSocket(WS_URL)
      socket.onopen = () => {
        setConnection('live')
        heartbeat = window.setInterval(() => {
          if (socket.readyState === WebSocket.OPEN) socket.send('ping')
        }, 20000)
      }
      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          if (message.type === 'scan' && message.batch) {
            setBatches((current) => [message.batch, ...current.filter((item) => item.batch_id !== message.batch.batch_id)].slice(0, 50))
            setChambers((current) => mergeChamber(current, message.chamber))
            setLastSeen(new Date().toISOString())
          }
          if (message.type === 'chamber' && message.chamber) setChambers((current) => mergeChamber(current, message.chamber))
        } catch {
          setConnection('degraded')
        }
      }
      socket.onclose = () => {
        window.clearInterval(heartbeat)
        setConnection('degraded')
        reconnectTimer = window.setTimeout(connect, 2500)
      }
      socket.onerror = () => socket.close()
    }
    const loadInitial = async () => {
      try {
        const [batchResponse, chamberResponse, modelResponse] = await Promise.all([
          fetch(`${API_URL}/api/v1/batches?limit=30`),
          fetch(`${API_URL}/api/v1/chambers`),
          fetch(`${API_URL}/api/v1/model/status`),
        ])
        if (batchResponse.ok) {
          const data = await batchResponse.json()
          if (data.length) setBatches(data)
        }
        if (chamberResponse.ok) {
          const data = await chamberResponse.json()
          if (data.length) setChambers(data)
        }
        if (modelResponse.ok) {
          const data = await modelResponse.json()
          setModelStatus(data)
          if (data.checkpoint_available && data.classes?.length) {
            const products = [...new Set(data.classes.filter((item) => item !== 'Expired').map((item) => item.split('(')[0]))]
            if (products.length) setProduceType((current) => products.includes(current) ? current : products[0])
          }
        }
      } catch {
        setConnection((current) => current === 'connecting' ? 'simulation' : current)
      }
    }
    loadInitial()
    connect()
    return () => {
      cancelled = true
      window.clearInterval(heartbeat)
      window.clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [])

  useEffect(() => {
    const canvas = scanCanvas.current
    const pixels = latest?.scan_preview
    if (!canvas || !pixels?.length || !pixels[0]?.length) return
    const height = pixels.length
    const width = pixels[0].length
    canvas.width = width
    canvas.height = height
    const context = canvas.getContext('2d')
    const image = context.createImageData(width, height)
    pixels.forEach((row, y) => row.forEach((color, x) => {
      const offset = (y * width + x) * 4
      image.data[offset] = color[0] ?? 0
      image.data[offset + 1] = color[1] ?? 0
      image.data[offset + 2] = color[2] ?? 0
      image.data[offset + 3] = 255
    }))
    context.putImageData(image, 0, 0)
  }, [latest?.scan_preview])

  useEffect(() => () => {
    cameraStream.current?.getTracks().forEach((track) => track.stop())
  }, [])

  async function connectCamera() {
    setCameraError('')
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera access is unavailable. Open the dashboard on localhost or HTTPS.')
      return
    }
    setCameraBusy(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640, max: 1920 }, height: { ideal: 480, max: 1080 } },
        audio: false,
      })
      cameraStream.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setCameraActive(true)
    } catch (error) {
      const denied = error?.name === 'NotAllowedError' || error?.name === 'PermissionDeniedError'
      setCameraError(denied ? 'Camera permission was denied. Allow camera access in browser site settings.' : 'Could not open a webcam. Check that it is connected and not in use by another app.')
    } finally {
      setCameraBusy(false)
    }
  }

  function disconnectCamera() {
    cameraStream.current?.getTracks().forEach((track) => track.stop())
    cameraStream.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraActive(false)
    setCameraError('')
  }

  async function captureWebcamScan() {
    const video = videoRef.current
    const canvas = captureCanvas.current
    if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return
    setScanBusy(true)
    setCameraError('')
    try {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
      const imageBase64 = canvas.toDataURL('image/jpeg', 0.82).split(',')[1]
      const response = await fetch(`${API_URL}/api/v1/scans/webcam`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imageBase64, produce: produceType, chamber_id: selectedChamber }),
      })
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail || `Scan failed (${response.status})`)
      }
      const batch = await response.json()
      setBatches((current) => [batch, ...current.filter((item) => item.batch_id !== batch.batch_id)].slice(0, 50))
      setLastSeen(new Date().toISOString())
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : 'Could not submit webcam scan.')
    } finally {
      setScanBusy(false)
    }
  }

  async function optimize() {
    if (!selected || !selectedBatch) return
    setOptimizing(true)
    try {
      const response = await fetch(`${API_URL}/api/v1/chambers/${selected.chamber_id}/optimize?expected_rsl_days=${selectedBatch.expected_rsl_days}`, { method: 'POST' })
      if (!response.ok) throw new Error('Controller request failed')
      const chamber = await response.json()
      setChambers((current) => mergeChamber(current, chamber))
    } catch {
      const target = selectedBatch.expected_rsl_days < 4 ? [3.5, 92, 90] : selectedBatch.expected_rsl_days < 8 ? [6, 90, 60] : [8.5, 85, 45]
      setChambers((current) => mergeChamber(current, { ...selected, setpoint_c: target[0], target_humidity_pct: target[1], target_fan_pct: target[2] }))
    } finally {
      setOptimizing(false)
    }
  }

  const curve = latest ? survivalCurve(latest) : []
  const isLive = connection === 'live'
  const trainedModel = modelStatus.status === 'trained_interval_weibull'
  const produceOptions = trainedModel
    ? [...new Set(modelStatus.classes.filter((item) => item !== 'Expired').map((item) => item.split('(')[0]))]
    : ['Tomato', 'Apple', 'Banana', 'Carrot', 'Avocados', 'Spinach', 'Strawberries', 'Bell peppers']

  return (
    <div className="app-shell">
      <aside className="rail">
        <a className="brand-mark" href="#overview" aria-label="HarvestCast home"><Leaf size={21} strokeWidth={2.2} /></a>
        <nav className="rail-nav" aria-label="Primary navigation">
          <a className="rail-link active" href="#overview" aria-label="Overview" title="Overview"><Command size={18} /></a>
          <a className="rail-link" href="#batches" aria-label="Produce batches" title="Batches"><Boxes size={18} /></a>
          <a className="rail-link" href="#chambers" aria-label="Cold chambers" title="Chambers"><Snowflake size={18} /></a>
          <a className="rail-link" href="#analytics" aria-label="Analytics" title="Analytics"><Activity size={18} /></a>
        </nav>
        <div className="rail-bottom"><button className="rail-link" aria-label="Help" title="Help"><CircleHelp size={18} /></button><button className="rail-link" aria-label="Settings" title="Settings"><Settings2 size={18} /></button><div className="avatar">OP</div></div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div className="topbar-title"><span className="eyebrow">FIELD OPERATIONS / SOUTH REGION</span><h1>Cold-chain control room</h1></div>
          <div className="topbar-right"><div className={`connection ${isLive ? 'is-live' : ''}`}><span className="connection-dot" />{isLive ? 'LIVE TELEMETRY' : connection === 'simulation' ? 'DEMO MODE' : 'RECONNECTING'}</div><span className="topbar-date">{new Intl.DateTimeFormat('en', { weekday: 'short', month: 'short', day: 'numeric' }).format(new Date())}</span><button className="icon-button" title="Refresh telemetry" onClick={() => window.location.reload()}><RefreshCw size={16} /></button></div>
        </header>

        <section className="metric-strip" aria-label="Operations summary">
          <div className="metric-cell"><div className="metric-label"><span>Tracked batches</span><Boxes size={15} /></div><div className="metric-value">{batches.length.toString().padStart(2, '0')}<span className="metric-unit"> monitored</span></div><div className="metric-foot"><span className="signal-positive"><Radio size={12} /> {isLive ? 'Live feed' : 'Cached view'}</span><span>across {chambers.length} chambers</span></div></div>
          <div className="metric-cell"><div className="metric-label"><span>Mean shelf life</span><Activity size={15} /></div><div className="metric-value">{(batches.reduce((sum, batch) => sum + batch.expected_rsl_days, 0) / Math.max(batches.length, 1)).toFixed(1)}<span className="metric-unit"> days</span></div><div className="metric-foot"><span className="signal-positive">RSL estimate</span><span>across tracked batches</span></div></div>
          <div className="metric-cell"><div className="metric-label"><span>Urgent dispatch</span><ArrowDownRight size={15} /></div><div className="metric-value">{batches.filter((batch) => batch.expected_rsl_days < 4).length.toString().padStart(2, '0')}<span className="metric-unit"> lots</span></div><div className="metric-foot"><span className="signal-warning">FEFO priority</span><span>review allocations</span></div></div>
          <div className="metric-cell"><div className="metric-label"><span>Chambers on target</span><Gauge size={15} /></div><div className="metric-value">{chambersOnTarget}<span className="metric-unit"> / {chambers.length} within 2°C</span></div><div className="metric-foot"><span className={chambersOnTarget === chambers.length ? 'signal-positive' : 'signal-warning'}>{chambersOnTarget === chambers.length ? 'All stable' : 'Review variance'}</span><span>measured vs. setpoint</span></div></div>
        </section>

        <section className="dashboard-grid">
          <article className="panel scan-panel" id="analytics">
            <div className="panel-heading"><div><span className="section-index">01 / OPTICAL INTELLIGENCE</span><h2>Latest multispectral scan</h2></div><div className="scan-controls"><span className={`proxy-badge ${trainedModel ? 'trained-model-badge' : ''}`} title={trainedModel ? 'A pretrained MobileNetV3 backbone was fine-tuned on this folder-labeled image dataset; RSL remains a range-based estimate.' : 'No fine-tuned checkpoint is loaded. The current inference is untrained demonstration output.'}>{trainedModel ? 'TRAINED RGB · RANGE RSL' : modelStatus.status === 'loading' ? 'CHECKING MODEL' : 'UNTRAINED · DEMO ONLY'}</span><label className="produce-select"><span>PRODUCE</span><select value={produceType} onChange={(event) => setProduceType(event.target.value)}>{produceOptions.map((product) => <option key={product}>{product}</option>)}</select></label><button className="camera-button" onClick={cameraActive ? disconnectCamera : connectCamera} disabled={cameraBusy} title={cameraActive ? 'Disconnect webcam' : 'Connect webcam'} aria-label={cameraActive ? 'Disconnect webcam' : 'Connect webcam'}>{cameraActive ? <CircleStop size={14} /> : <Camera size={14} />}<span>{cameraBusy ? 'Connecting' : cameraActive ? 'Stop' : 'Connect'}</span></button><button className="capture-button" onClick={captureWebcamScan} disabled={!cameraActive || scanBusy} title="Capture and infer webcam frame" aria-label="Capture webcam scan"><ScanLine size={14} /><span>{scanBusy ? 'Scanning' : 'Capture'}</span></button></div></div>
            <div className="scan-content">
              <div className={`scan-visual ${cameraActive ? 'camera-connected' : ''}`} role="img" aria-label={cameraActive ? 'Live webcam preview' : 'Latest scan composite'}><canvas ref={scanCanvas} className="scan-canvas" /><video ref={videoRef} className={`webcam-feed ${cameraActive ? 'active' : ''}`} autoPlay playsInline muted /><div className="scan-grid" /><div className="scan-sweep" /><div className="scan-crosshair crosshair-one" /><div className="scan-crosshair crosshair-two" /><span className="scan-corner top-left" /><span className="scan-corner top-right" /><span className="scan-corner bottom-left" /><span className="scan-corner bottom-right" /><div className="scan-image-caption"><span>{cameraActive ? 'LIVE RGB CAMERA' : latest?.source === 'webcam_rgb_trained' ? 'WEBCAM · TRAINED RGB' : latest?.source === 'webcam_rgb_proxy' ? 'RGB + PROXY NIR' : 'SIMULATED RGB / NIR'}</span><span>{cameraActive ? `${videoRef.current?.videoWidth || 0} × ${videoRef.current?.videoHeight || 0} PX` : '32 × 24 PX'}</span></div></div>
              <canvas ref={captureCanvas} className="capture-canvas" aria-hidden="true" />
              <div className="scan-readout"><div className="batch-kicker"><span className="live-pip" />INFERENCE COMPLETE</div><h3>{latest?.produce || 'Awaiting produce'}</h3><div className="batch-code">{latest?.batch_id || 'NO BATCH DETECTED'}</div><div className="scan-result"><strong>{latest?.expected_rsl_days?.toFixed(1) ?? '--'}</strong><span>days<br />expected RSL</span></div><div className="scan-features"><div><span>Vigor proxy</span><b>{latest?.vigor?.toFixed(2) ?? '--'}</b></div><div><span>Water ratio proxy</span><b>{latest?.water_ratio?.toFixed(2) ?? '--'}</b></div><div><span>Predicted interval</span><b>{latest?.predicted_range || 'Not trained'}</b></div><div><span>Top-class confidence</span><b>{latest?.confidence != null ? `${(latest.confidence * 100).toFixed(0)}%` : 'N/A'}</b></div></div><div className={`action-tag ${actionMeta(latest?.dispatch_action).tone}`}>{actionMeta(latest?.dispatch_action).label}</div></div>
            </div>
            {cameraError && <div className="camera-error" role="alert">{cameraError}</div>}
            <div className="spectral-legend"><span><i className="legend-dot red" /> RED BAND</span><span><i className="legend-dot green" /> GREEN BAND</span><span><i className="legend-dot nir" /> {latest?.source === 'webcam_rgb_trained' ? 'RGB TRAINED MODEL' : latest?.source === 'webcam_rgb_proxy' ? 'PROXY NIR' : 'NIR BAND'}</span><span className="spectral-time">{latest?.source?.startsWith('webcam') ? 'WEBCAM INPUT' : 'SIMULATED FEED'} · {formatAge(lastSeen)}</span></div>
          </article>

          <article className="panel survival-panel">
            <div className="panel-heading"><div><span className="section-index">02 / SURVIVAL MODEL</span><h2>Weibull shelf-life curve</h2></div><span className="model-chip">k {latest?.k_shape?.toFixed(2) ?? '--'} <span>·</span> λ {latest?.lambda_days?.toFixed(1) ?? '--'}d</span></div>
            <div className="chart-summary"><strong>{latest?.expected_rsl_days?.toFixed(1) ?? '--'} <small>days</small></strong><span>mean remaining life<br />Γ(1 + 1/k) × λ</span></div>
            <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><AreaChart data={curve} margin={{ top: 12, right: 8, left: -18, bottom: 0 }}><defs><linearGradient id="survivalFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#b9ec67" stopOpacity={0.24} /><stop offset="100%" stopColor="#b9ec67" stopOpacity={0} /></linearGradient></defs><CartesianGrid stroke="#29312b" vertical={false} /><XAxis dataKey="day" tickFormatter={(value) => `${value}d`} tickLine={false} axisLine={false} tick={{ fill: '#768178', fontSize: 10 }} /><YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tickFormatter={(value) => `${value}%`} tickLine={false} axisLine={false} tick={{ fill: '#768178', fontSize: 10 }} /><Tooltip contentStyle={{ background: '#171e19', border: '1px solid #38433a', borderRadius: 4, color: '#edf2e9', fontSize: 12 }} formatter={(value) => [`${Number(value).toFixed(1)}%`, 'Survival']} labelFormatter={(value) => `Day ${value}`} /><Area type="monotone" dataKey="survival" stroke="#b9ec67" strokeWidth={2} fill="url(#survivalFill)" activeDot={{ r: 4, fill: '#b9ec67', stroke: '#111613', strokeWidth: 2 }} /></AreaChart></ResponsiveContainer></div>
            <div className="chart-footer"><span><i className="legend-line" /> PROBABILITY OF MARKETABLE QUALITY</span><span>0—12 DAYS</span></div>
          </article>

          <article className="panel chamber-panel" id="chambers">
            <div className="panel-heading"><div><span className="section-index">03 / ENVIRONMENTAL CONTROL</span><h2>Storage microclimate</h2></div><span className="auto-badge"><span className="live-pip" /> MPC ACTIVE</span></div>
            <div className="chamber-toolbar"><label htmlFor="chamber-select">CHAMBER</label><div className="select-wrap"><select id="chamber-select" value={selectedChamber} onChange={(event) => setSelectedChamber(event.target.value)}>{chambers.map((chamber) => <option key={chamber.chamber_id} value={chamber.chamber_id}>{chamber.chamber_id}</option>)}</select><ChevronDown size={14} /></div><span className="chamber-ready"><ShieldCheck size={14} /> Stable</span></div>
            <div className="hvac-grid"><div className="hvac-item"><div className="hvac-icon temp"><Thermometer size={16} /></div><span className="hvac-label">AIR TEMPERATURE</span><strong>{selected?.temperature_c?.toFixed(1) ?? '--'}<small>°C</small></strong><div className="setpoint-line">TARGET <b>{selected?.setpoint_c?.toFixed(1) ?? '--'}°C</b></div></div><div className="hvac-item"><div className="hvac-icon humidity"><Cloud size={16} /></div><span className="hvac-label">RELATIVE HUMIDITY</span><strong>{selected?.humidity_pct?.toFixed(1) ?? '--'}<small>%</small></strong><div className="setpoint-line">TARGET <b>{selected?.target_humidity_pct?.toFixed(0) ?? '--'}%</b></div></div><div className="hvac-item"><div className="hvac-icon fan"><Wind size={16} /></div><span className="hvac-label">EVAPORATOR FAN</span><strong>{selected?.fan_pct?.toFixed(0) ?? '--'}<small>%</small></strong><div className="setpoint-line">TARGET <b>{selected?.target_fan_pct?.toFixed(0) ?? '--'}%</b></div></div></div>
            <div className="control-footer"><div><span className="control-pulse" />AUTONOMOUS SETPOINT OPTIMIZATION<span className="control-sub">Based on {selectedBatch?.expected_rsl_days?.toFixed(1) ?? '--'}d chamber RSL</span></div><button className="apply-button" onClick={optimize} disabled={optimizing || !selected}><Settings2 size={15} />{optimizing ? 'Applying' : 'Re-optimize'}</button></div>
          </article>

          <article className="panel batches-panel" id="batches">
            <div className="panel-heading"><div><span className="section-index">04 / FEFO QUEUE</span><h2>Produce in cold storage</h2></div><button className="text-button" onClick={() => document.getElementById('batches-table')?.scrollIntoView({ behavior: 'smooth', block: 'center' })}>VIEW ALL <ArrowUpRight size={13} /></button></div>
            <div className="table-scroll"><table id="batches-table"><thead><tr><th>PRODUCE / LOT</th><th>CHAMBER</th><th>EXPECTED RSL</th><th>DISPATCH PLAN</th></tr></thead><tbody>{batches.slice(0, 5).map((batch) => { const action = actionMeta(batch.dispatch_action); return <tr key={batch.batch_id}><td><div className="produce-cell"><span className={`produce-swatch ${batch.produce.toLowerCase().replaceAll(' ', '-')}`}><Leaf size={13} /></span><div><b>{batch.produce}</b><small>{batch.batch_id}</small></div></div></td><td className="mono-cell">{batch.chamber_id}</td><td><strong className={batch.expected_rsl_days < 4 ? 'rsl-urgent' : ''}>{batch.expected_rsl_days.toFixed(1)}d</strong></td><td><span className={`action-tag ${action.tone} compact`}>{action.label}</span></td></tr> })}</tbody></table></div>
            <div className="table-foot"><span><span className="live-pip" /> UPDATING IN REAL TIME</span><span>{batches.length} batches · Last scan {formatAge(batches[0]?.measured_at)}</span></div>
          </article>
        </section>

        <footer className="page-footer"><span>HARVESTCAST AI <span className="footer-separator">/</span> EDGE-TO-CLOUD AGRICULTURAL INTELLIGENCE</span><span><span className="footer-light" /> SDG 02 ZERO HUNGER <span className="footer-separator">·</span> SDG 12 RESPONSIBLE PRODUCTION</span></footer>
      </main>
    </div>
  )
}

function mergeChamber(chambers, updated) {
  if (!updated) return chambers
  const exists = chambers.some((chamber) => chamber.chamber_id === updated.chamber_id)
  return exists
    ? chambers.map((chamber) => chamber.chamber_id === updated.chamber_id ? { ...chamber, ...updated } : chamber)
    : [updated, ...chambers]
}

export default App
