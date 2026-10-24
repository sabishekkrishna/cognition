import React, { useState, useRef, useEffect } from 'react';
import {
  Video, VideoOff, Mic, Square, Play, Pause, Activity,
  AlertTriangle, CheckCircle2, ShieldCheck, Gauge, Settings2
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from 'recharts';
import { SystemOverallStatus, MetricWarning } from '../types';
import { WarningCard } from '../components/WarningCard';

interface MetricHistoryPoint {
  time: string;
  sharpness: number;
  brightness: number;
  noise: number;
  reliability: number;
}

export const LiveMonitorPage: React.FC = () => {
  // Camera state
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [samplingFps, setSamplingFps] = useState<number>(2); // 1, 2, or 5 FPS
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const samplingTimerRef = useRef<any>(null);

  // Microphone monitoring state
  const [isMicActive, setIsMicActive] = useState(false);
  const [audioLevel, setAudioLevel] = useState<number>(0);
  const micStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  // WebSocket state
  const wsRef = useRef<WebSocket | null>(null);
  const [wsConnected, setWsConnected] = useState(false);

  // Live Metrics & Rolling chart
  const [liveImageMetrics, setLiveImageMetrics] = useState<any>(null);
  const [liveAudioMetrics, setLiveAudioMetrics] = useState<any>(null);
  const [liveStatus, setLiveStatus] = useState<string>('normal');
  const [liveReliabilityScore, setLiveReliabilityScore] = useState<number>(100);
  const [liveWarnings, setLiveWarnings] = useState<MetricWarning[]>([]);
  const [history, setHistory] = useState<MetricHistoryPoint[]>([]);

  // Setup WebSocket connection
  useEffect(() => {
    const ws = new WebSocket('ws://127.0.0.1:8000/ws/monitor');
    wsRef.current = ws;

    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onerror = () => setWsConnected(false);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'image_update') {
          setLiveImageMetrics(data.metrics);
          setLiveReliabilityScore(data.reliability_score);
          setLiveStatus(data.status);
          setLiveWarnings(data.warnings || []);

          // Add to rolling history
          const nowStr = new Date().toLocaleTimeString().split(' ')[0];
          setHistory((prev) => {
            const next = [
              ...prev,
              {
                time: nowStr,
                sharpness: data.metrics.sharpness,
                brightness: data.metrics.brightness,
                noise: data.metrics.noise_estimate,
                reliability: data.reliability_score
              }
            ];
            return next.slice(-25); // keep last 25 points
          });
        } else if (data.type === 'audio_update') {
          setLiveAudioMetrics(data.metrics);
        }
      } catch (err) {
        console.error('WS parse error:', err);
      }
    };

    return () => {
      stopCamera();
      stopMic();
      ws.close();
    };
  }, []);

  // Camera handling
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 } }
      });
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsCameraActive(true);

      // Start frame sampler interval
      const intervalMs = Math.round(1000 / samplingFps);
      samplingTimerRef.current = setInterval(captureAndSendFrame, intervalMs);
    } catch (err) {
      alert('Camera access denied or device not found: ' + err);
    }
  };

  const stopCamera = () => {
    if (samplingTimerRef.current) {
      clearInterval(samplingTimerRef.current);
      samplingTimerRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  };

  const captureAndSendFrame = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
    if (wsRef.current.readyState !== WebSocket.OPEN) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    canvas.width = 320;
    canvas.height = 240;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, 320, 240);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.7);

    wsRef.current.send(JSON.stringify({
      type: 'video_frame',
      frame_base64: dataUrl,
      timestamp: Date.now()
    }));
  };

  // Adjust sampling interval dynamically
  const changeFps = (fps: number) => {
    setSamplingFps(fps);
    if (isCameraActive) {
      if (samplingTimerRef.current) clearInterval(samplingTimerRef.current);
      samplingTimerRef.current = setInterval(captureAndSendFrame, Math.round(1000 / fps));
    }
  };

  // Microphone monitoring handling
  const startMic = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStreamRef.current = stream;
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const checkAudio = () => {
        if (!micStreamRef.current) return;
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
        const avg = sum / dataArray.length;
        setAudioLevel(Math.min(100, Math.round((avg / 128) * 100)));
        requestAnimationFrame(checkAudio);
      };
      checkAudio();
      setIsMicActive(true);
    } catch (err) {
      alert('Microphone access denied: ' + err);
    }
  };

  const stopMic = () => {
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setIsMicActive(false);
    setAudioLevel(0);
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1f293d]">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2.5">
            Real-Time Live Perceptual Monitor
            <span className={`text-xs px-2.5 py-0.5 rounded-full font-mono flex items-center gap-1.5 ${
              wsConnected ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400' : 'bg-rose-500/10 border border-rose-500/30 text-rose-400'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${wsConnected ? 'bg-emerald-400' : 'bg-rose-400'}`} />
              {wsConnected ? 'WebSocket Stream Active' : 'Disconnected'}
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Sample live camera video and audio frames with exponential smoothing and hysteresis.
          </p>
        </div>

        {/* FPS Control */}
        <div className="flex items-center gap-2 bg-[#111827] px-3 py-1.5 rounded-xl border border-slate-800 text-xs">
          <Settings2 className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-slate-400">Sampling Rate:</span>
          {[1, 2, 5].map((fps) => (
            <button
              key={fps}
              onClick={() => changeFps(fps)}
              className={`px-2 py-0.5 rounded font-mono text-[11px] font-medium transition ${
                samplingFps === fps
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {fps} FPS
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Live Video Canvas & Mic Bar */}
        <div className="space-y-5">
          {/* Camera Viewport */}
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-4 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                <Video className="w-4 h-4 text-cyan-400" />
                Live Camera Input
              </span>
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                isCameraActive ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-500'
              }`}>
                {isCameraActive ? `STREAMING @ ${samplingFps} FPS` : 'OFFLINE'}
              </span>
            </div>

            {/* Video element */}
            <div className="relative aspect-video bg-[#0b0f19] rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
              <video
                ref={videoRef}
                playsInline
                muted
                className={`w-full h-full object-cover ${isCameraActive ? 'block' : 'hidden'}`}
              />
              <canvas ref={canvasRef} className="hidden" />

              {!isCameraActive && (
                <div className="text-center p-6">
                  <VideoOff className="w-10 h-10 text-slate-600 mx-auto mb-2" />
                  <p className="text-xs text-slate-400 font-medium">Camera is currently paused</p>
                  <p className="text-[11px] text-slate-600 mt-1">Click below to grant camera access and start monitoring.</p>
                </div>
              )}

              {isCameraActive && (
                <div className="absolute top-2 left-2 flex items-center gap-1.5 px-2 py-0.5 rounded bg-black/60 backdrop-blur-sm text-[10px] font-mono text-cyan-300 border border-cyan-500/30">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                  ANALYZING FRAMES
                </div>
              )}
            </div>

            {/* Camera Toggle Button */}
            {isCameraActive ? (
              <button
                onClick={stopCamera}
                className="w-full py-2.5 rounded-xl bg-rose-600/90 hover:bg-rose-500 text-white font-semibold text-xs tracking-wide flex items-center justify-center gap-2 transition"
              >
                <VideoOff className="w-4 h-4" /> Stop Camera Monitoring
              </button>
            ) : (
              <button
                onClick={startCamera}
                className="w-full py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs tracking-wide flex items-center justify-center gap-2 transition shadow-sm"
              >
                <Video className="w-4 h-4" /> Start Camera Monitoring
              </button>
            )}
          </div>

          {/* Microphone Live Monitor */}
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                <Mic className="w-4 h-4 text-violet-400" />
                Live Acoustic Level
              </span>
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                isMicActive ? 'bg-violet-500/20 text-violet-400' : 'bg-slate-800 text-slate-500'
              }`}>
                {isMicActive ? 'ACTIVE' : 'IDLE'}
              </span>
            </div>

            {/* VU Meter */}
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] font-mono text-slate-400">
                <span>Microphone Level:</span>
                <span className="text-white">{audioLevel}%</span>
              </div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-75 ${
                    audioLevel > 80 ? 'bg-rose-400' : audioLevel > 50 ? 'bg-amber-400' : 'bg-violet-400'
                  }`}
                  style={{ width: `${audioLevel}%` }}
                />
              </div>
            </div>

            {/* Mic Toggle Button */}
            {isMicActive ? (
              <button
                onClick={stopMic}
                className="w-full py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center justify-center gap-2 transition"
              >
                <Square className="w-3.5 h-3.5 fill-current" /> Stop Microphone
              </button>
            ) : (
              <button
                onClick={startMic}
                className="w-full py-2 rounded-lg bg-violet-600/20 hover:bg-violet-600/30 border border-violet-500/40 text-violet-300 text-xs font-medium flex items-center justify-center gap-2 transition"
              >
                <Mic className="w-3.5 h-3.5" /> Start Live Mic Monitoring
              </button>
            )}
          </div>
        </div>

        {/* Right Columns: Rolling Metrics Chart, Hysteresis, and Live Alarms */}
        <div className="lg:col-span-2 space-y-5">
          {/* Real-Time Rolling Quality Line Chart */}
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-[10px] font-mono uppercase text-slate-400">Time-Series Stream</span>
                <h3 className="text-xs font-semibold text-white uppercase tracking-wider">
                  Rolling Quality Metrics & Smoothed Reliability Index
                </h3>
              </div>
              <div className="flex items-center gap-3 text-[11px] font-mono">
                <span className="flex items-center gap-1.5 text-cyan-400">
                  <span className="w-2 h-2 rounded-full bg-cyan-400" /> Sharpness
                </span>
                <span className="flex items-center gap-1.5 text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" /> Reliability
                </span>
                <span className="flex items-center gap-1.5 text-amber-400">
                  <span className="w-2 h-2 rounded-full bg-amber-400" /> Brightness
                </span>
              </div>
            </div>

            {/* Recharts Line Chart */}
            <div className="h-64 w-full">
              {history.length > 1 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
                    <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 10 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 10 }} domain={[0, 'auto']} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0d1322', borderColor: '#1f293d', borderRadius: 8, fontSize: 11 }}
                    />
                    <Line type="monotone" dataKey="sharpness" stroke="#06b6d4" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="reliability" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="brightness" stroke="#f59e0b" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-center text-slate-600">
                  <Activity className="w-8 h-8 mb-2" />
                  <p className="text-xs">Awaiting live frame stream to populate rolling chart...</p>
                </div>
              )}
            </div>
          </div>

          {/* Sustained Deviation & Hysteresis Status Card */}
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div>
                <span className="text-[10px] font-mono uppercase text-slate-400">Hysteresis Filter</span>
                <h4 className="text-xs font-semibold text-white uppercase tracking-wider">
                  Sustained Deviation Detection Status
                </h4>
              </div>
              <span className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold uppercase ${
                liveStatus === 'warning'
                  ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                  : liveStatus === 'caution'
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                  : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
              }`}>
                STATE: {liveStatus}
              </span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              To eliminate false alarms caused by transient single-frame motion blur or lighting blinks, a rolling buffer of 5 frames with 3-consecutive-sample hysteresis is active. Warnings are triggered only upon sustained degradation.
            </p>

            {/* Live Warnings List */}
            <div className="space-y-2.5 pt-1">
              <h5 className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">
                Live Active Alarms ({liveWarnings.length})
              </h5>
              {liveWarnings.length > 0 ? (
                liveWarnings.map((w, idx) => (
                  <WarningCard key={idx} warning={w} />
                ))
              ) : (
                <div className="p-3.5 rounded-xl bg-emerald-950/20 border border-emerald-500/30 text-emerald-300 text-xs flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                  <span>No sustained deviations. Continuous perceptual feed is within baseline tolerances.</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
