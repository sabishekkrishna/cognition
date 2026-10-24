import React, { useState, useRef } from 'react';
import {
  Upload, Mic, Square, Play, Pause, RotateCcw,
  Sparkles, Sliders, CheckCircle2, AlertTriangle, Volume2
} from 'lucide-react';
import { AudioAnalysisResponse } from '../types';
import { api } from '../services/api';
import { ReliabilityGauge } from '../components/ReliabilityGauge';
import { WarningCard } from '../components/WarningCard';
import { WaveformVisualizer } from '../components/WaveformVisualizer';

// Converts any browser-recorded audio (WebM/Opus/etc.) to standard 16-bit PCM WAV
async function convertBlobToPcmWav(blob: Blob): Promise<Blob> {
  const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
  if (!AudioCtx) return blob;
  const audioContext = new AudioCtx();
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const numSamples = audioBuffer.length;

    // Downmix to mono float32
    const monoData = new Float32Array(numSamples);
    if (numChannels === 1) {
      monoData.set(audioBuffer.getChannelData(0));
    } else {
      const ch0 = audioBuffer.getChannelData(0);
      const ch1 = audioBuffer.getChannelData(1);
      for (let i = 0; i < numSamples; i++) {
        monoData[i] = (ch0[i] + ch1[i]) / 2.0;
      }
    }

    // Build 16-bit PCM RIFF WAV
    const wavBuffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(wavBuffer);

    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // 1 channel
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); // 16-bit
    writeString(36, 'data');
    view.setUint32(40, numSamples * 2, true);

    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
      const s = Math.max(-1, Math.min(1, monoData[i]));
      const val = s < 0 ? s * 0x8000 : s * 0x7FFF;
      view.setInt16(offset, val, true);
      offset += 2;
    }

    return new Blob([view], { type: 'audio/wav' });
  } catch (e) {
    console.warn('Fallback to original recording blob:', e);
    return blob;
  } finally {
    try {
      await audioContext.close();
    } catch (_) {}
  }
}

export const AudioLabPage: React.FC = () => {
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioFileName, setAudioFileName] = useState<string>('sample.wav');
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AudioAnalysisResponse | null>(null);

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Synthetic degradation controls
  const [volumeFactor, setVolumeFactor] = useState<number>(1.0);
  const [noiseType, setNoiseType] = useState<string>('none');
  const [noiseLevel, setNoiseLevel] = useState<number>(0.0);
  const [clippingGain, setClippingGain] = useState<number>(1.0);

  // Comparison state
  const [degradeComparison, setDegradeComparison] = useState<{
    applied: string;
    degradedAudioUrl: string;
    originalAudioUrl: string;
    degradedAnalysis: any;
    originalAnalysis: any;
  } | null>(null);

  const handleAudioUpload = async (file: Blob | File) => {
    setErrorMessage(null);

    // Validation checks
    if (file.size === 0) {
      setErrorMessage('The selected file is empty (0 bytes). Please upload a valid recording.');
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      setErrorMessage('Audio file exceeds maximum allowed size of 50 MB.');
      return;
    }

    setAudioBlob(file);
    const fileName = file instanceof File ? file.name : 'mic_recording.wav';
    setAudioFileName(fileName);

    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    setDegradeComparison(null);
    resetSliders();

    try {
      setLoading(true);
      const res = await api.analyzeAudio(file);
      setAnalysis(res);
    } catch (err: any) {
      setErrorMessage(err.message || 'Error analyzing audio recording');
    } finally {
      setLoading(false);
    }
  };

  const loadSampleAudio = async () => {
    try {
      setLoading(true);
      setErrorMessage(null);
      const blob = await api.getSampleAudioBlob();
      await handleAudioUpload(blob);
    } catch (err: any) {
      setErrorMessage('Error loading sample audio: ' + err.message);
      setLoading(false);
    }
  };

  const startRecording = async () => {
    setErrorMessage(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const rawBlob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType || 'audio/webm' });
        // Convert webm/opus to true 16-bit PCM WAV
        const wavBlob = await convertBlobToPcmWav(rawBlob);
        await handleAudioUpload(wavBlob);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      setErrorMessage('Microphone access denied or unavailable: ' + err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const applySyntheticDegradation = async () => {
    if (!audioBlob) return;
    try {
      setLoading(true);
      const res = await api.degradeAudio(audioBlob, {
        volume_factor: volumeFactor,
        noise_type: noiseType !== 'none' ? noiseType : undefined,
        noise_level: noiseLevel,
        clipping_gain: clippingGain
      });

      setDegradeComparison({
        applied: res.degradation_applied,
        degradedAudioUrl: res.degraded_audio_base64,
        originalAudioUrl: res.original_audio_base64,
        degradedAnalysis: res.degraded_analysis,
        originalAnalysis: res.original_analysis
      });

      setAnalysis(res.degraded_analysis);
    } catch (err: any) {
      alert('Degradation error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetSliders = () => {
    setVolumeFactor(1.0);
    setNoiseType('none');
    setNoiseLevel(0.0);
    setClippingGain(1.0);
  };

  const isDegraded = degradeComparison && degradeComparison.applied !== 'None (Clean Original)';

  return (
    <div className="space-y-6">
      {/* Error Banner */}
      {errorMessage && (
        <div className="p-3.5 rounded-xl bg-rose-950/50 border border-rose-700/60 flex items-center gap-2.5 text-xs text-rose-200 shadow-sm animate-pulse">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
          <span className="flex-1 font-mono">{errorMessage}</span>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-rose-400 hover:text-white font-bold px-2 py-0.5 rounded bg-rose-900/40 transition"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Title & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1f293d]">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2.5">
            Acoustic Signal & Audio Lab
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-violet-500/10 border border-violet-500/30 text-violet-400 font-mono">
              Signal Processing + Acoustic CNN
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Measure RMS loudness dBFS, peak amplitude, clipping distortion, SNR, and evaluate speech suitability before LLM processing.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={loadSampleAudio}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-violet-600/20 hover:bg-violet-600/30 border border-violet-500/40 text-xs text-violet-300 font-medium transition"
          >
            <Sparkles className="w-3.5 h-3.5" />
            Load Sample Audio
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Upload, Mic Recorder & Degradation Controls */}
        <div className="space-y-5">
          {/* Audio Input Options Card */}
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-4 shadow-sm space-y-4">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Audio Capture</h3>

            {/* Mic Record Button */}
            <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 text-center">
              {isRecording ? (
                <button
                  onClick={stopRecording}
                  className="w-full py-3 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs tracking-wide flex items-center justify-center gap-2 animate-pulse transition"
                >
                  <Square className="w-4 h-4 fill-current" />
                  Stop Recording ({audioChunksRef.current.length} chunks)
                </button>
              ) : (
                <button
                  onClick={startRecording}
                  className="w-full py-3 rounded-xl bg-violet-600 hover:bg-violet-500 text-white font-semibold text-xs tracking-wide flex items-center justify-center gap-2 transition"
                >
                  <Mic className="w-4 h-4" />
                  Start Microphone Recording
                </button>
              )}
              <p className="text-[10px] text-slate-500 mt-2 font-mono">PCM WAV in-browser encoding. Zero background recording.</p>
            </div>

            {/* Or File Upload */}
            <label className="border-2 border-dashed border-slate-700 hover:border-violet-500/60 rounded-xl p-4 flex flex-col items-center justify-center cursor-pointer transition bg-slate-900/30 group">
              <Upload className="w-5 h-5 text-slate-400 group-hover:text-violet-400 mb-1.5 transition" />
              <span className="text-xs font-medium text-slate-200">Upload Audio File (WAV, MP3, FLAC, OGG)</span>
              <span className="text-[10px] text-slate-400 mt-0.5">Loaded: {audioFileName}</span>
              <input
                type="file"
                accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleAudioUpload(e.target.files[0]);
                  }
                }}
              />
            </label>

            {/* Audio Waveform & Player */}
            {audioBlob && (
              <div className="space-y-3 pt-2">
                <span className="text-xs font-medium text-slate-300">Acoustic Waveform Canvas:</span>
                <WaveformVisualizer audioBlob={audioBlob} height={70} />

                <audio
                  controls
                  src={
                    degradeComparison
                      ? degradeComparison.degradedAudioUrl
                      : audioUrl || undefined
                  }
                  className="w-full h-8"
                />

                {isDegraded && (
                  <p className="text-[11px] font-mono text-amber-400 bg-amber-950/20 p-2 rounded border border-amber-900/40">
                    Acoustic Degradation: {degradeComparison.applied}
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Synthetic Audio Degradation Test Panel */}
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                <Sliders className="w-4 h-4 text-violet-400" />
                Acoustic Degradation Panel
              </h3>
              <button
                onClick={resetSliders}
                className="text-[11px] text-slate-400 hover:text-slate-200 flex items-center gap-1 font-mono"
              >
                <RotateCcw className="w-3 h-3" /> Reset
              </button>
            </div>

            <div className="space-y-3 text-xs">
              {/* Volume / Attenuation */}
              <div>
                <div className="flex justify-between text-slate-400 font-mono mb-1">
                  <span>Volume Scale:</span>
                  <span className="text-white">{volumeFactor.toFixed(2)}x</span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="2.5"
                  step="0.05"
                  value={volumeFactor}
                  onChange={(e) => setVolumeFactor(parseFloat(e.target.value))}
                  className="w-full accent-violet-400 bg-slate-800 h-1.5 rounded cursor-pointer"
                />
              </div>

              {/* Noise Type */}
              <div>
                <div className="flex justify-between text-slate-400 font-mono mb-1">
                  <span>Noise Injection Type:</span>
                  <span className="text-white capitalize">{noiseType}</span>
                </div>
                <div className="grid grid-cols-4 gap-1 font-mono text-[11px]">
                  {['none', 'white', 'pink', 'low_rumble'].map((t) => (
                    <button
                      key={t}
                      onClick={() => setNoiseType(t)}
                      className={`py-1 rounded border capitalize ${
                        noiseType === t
                          ? 'bg-violet-600/30 border-violet-400 text-white font-bold'
                          : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-800'
                      }`}
                    >
                      {t.replace('_', ' ')}
                    </button>
                  ))}
                </div>
              </div>

              {/* Noise Level */}
              {noiseType !== 'none' && (
                <div>
                  <div className="flex justify-between text-slate-400 font-mono mb-1">
                    <span>Noise Intensity:</span>
                    <span className="text-white">{Math.round(noiseLevel * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="1.0"
                    step="0.05"
                    value={noiseLevel}
                    onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
                    className="w-full accent-violet-400 bg-slate-800 h-1.5 rounded cursor-pointer"
                  />
                </div>
              )}

              {/* Clipping Saturation */}
              <div>
                <div className="flex justify-between text-slate-400 font-mono mb-1">
                  <span>Clipping Overdrive Gain:</span>
                  <span className="text-white">{clippingGain.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min="1.0"
                  max="6.0"
                  step="0.5"
                  value={clippingGain}
                  onChange={(e) => setClippingGain(parseFloat(e.target.value))}
                  className="w-full accent-violet-400 bg-slate-800 h-1.5 rounded cursor-pointer"
                />
              </div>

              <button
                disabled={!audioBlob || loading}
                onClick={applySyntheticDegradation}
                className="w-full py-2.5 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-semibold text-xs tracking-wide transition shadow-sm mt-2"
              >
                {loading ? 'Analyzing Audio...' : 'Apply Degradation & Re-Analyze'}
              </button>
            </div>
          </div>
        </div>

        {/* Right Columns: Analysis, Gauges, Warnings & Acoustic CNN */}
        <div className="lg:col-span-2 space-y-5">
          {analysis ? (
            <>
              {/* Gauges & Status Card */}
              <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-[#1f293d] gap-4">
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                      Speech Suitability
                    </span>
                    <h3 className="text-base font-bold text-white flex items-center gap-2">
                      Audio Reliability Status:
                      <span className={`px-2 py-0.5 rounded text-xs font-mono font-bold uppercase ${
                        analysis.status === 'warning'
                          ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                          : analysis.status === 'caution'
                          ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                          : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      }`}>
                        {analysis.status}
                      </span>
                    </h3>
                  </div>

                  {/* Dual Score Gauges */}
                  <div className="flex items-center gap-4">
                    <ReliabilityGauge score={analysis.reliability_score} label="Reliability" size={105} />
                    <ReliabilityGauge score={analysis.risk_score} label="Acoustic Risk" size={105} />
                  </div>
                </div>

                {/* Warnings */}
                <div className="mt-4">
                  <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2.5">
                    Acoustic Warnings ({analysis.warnings.length})
                  </h4>
                  {analysis.warnings.length > 0 ? (
                    <div className="space-y-2.5">
                      {analysis.warnings.map((w, idx) => (
                        <WarningCard key={idx} warning={w} />
                      ))}
                    </div>
                  ) : (
                    <div className="p-3.5 rounded-xl bg-emerald-950/20 border border-emerald-500/30 text-emerald-300 text-xs flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                      <span>Audio quality matches speech recognition baseline. Suitable for downstream transcription.</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Acoustic Metrics Grid */}
              <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
                <h3 className="text-xs font-semibold text-white uppercase tracking-wider">
                  Signal Measurements & Speech Activity
                </h3>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 font-mono">
                  {/* RMS Loudness */}
                  <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                    <span className="text-[10px] uppercase text-slate-400">RMS Loudness</span>
                    <div className="text-base font-bold text-white mt-0.5">
                      {analysis.metrics.rms_dbfs.toFixed(1)} dBFS
                    </div>
                    <span className="text-[10px] text-slate-400">
                      Peak: {analysis.metrics.peak_amplitude.toFixed(2)}
                    </span>
                  </div>

                  {/* Clipping */}
                  <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                    <span className="text-[10px] uppercase text-slate-400">Clipping %</span>
                    <div className={`text-base font-bold mt-0.5 ${analysis.metrics.clipping_pct > 1.0 ? 'text-rose-400' : 'text-white'}`}>
                      {analysis.metrics.clipping_pct.toFixed(2)}%
                    </div>
                    <span className="text-[10px] text-slate-400">
                      Expected: ≤ 0.5%
                    </span>
                  </div>

                  {/* SNR */}
                  <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                    <span className="text-[10px] uppercase text-slate-400">Estimated SNR</span>
                    <div className={`text-base font-bold mt-0.5 ${analysis.metrics.snr_db < 10 ? 'text-rose-400' : 'text-white'}`}>
                      {analysis.metrics.snr_db.toFixed(1)} dB
                    </div>
                    <span className="text-[10px] text-slate-400">
                      Baseline Dev: {analysis.baseline_deviations?.snr_db?.toFixed(1) ?? '0.0'}σ
                    </span>
                  </div>

                  {/* Noise Floor */}
                  <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                    <span className="text-[10px] uppercase text-slate-400">Background Noise</span>
                    <div className="text-base font-bold text-white mt-0.5">
                      {analysis.metrics.background_noise_db.toFixed(1)} dBFS
                    </div>
                    <span className="text-[10px] text-slate-400">
                      10th percentile energy
                    </span>
                  </div>

                  {/* VAD Speech Ratio */}
                  <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                    <span className="text-[10px] uppercase text-slate-400">Speech Activity (VAD)</span>
                    <div className="text-base font-bold text-white mt-0.5">
                      {(analysis.metrics.speech_ratio * 100).toFixed(0)}% Speech
                    </div>
                    <span className="text-[10px] text-slate-400">
                      {(analysis.metrics.silence_ratio * 100).toFixed(0)}% Pause/Silence
                    </span>
                  </div>

                  {/* Spectral Centroid */}
                  <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                    <span className="text-[10px] uppercase text-slate-400">Spectral Centroid</span>
                    <div className="text-base font-bold text-white mt-0.5">
                      {analysis.metrics.spectral_centroid_hz.toFixed(0)} Hz
                    </div>
                    <span className="text-[10px] text-slate-400">
                      Rolloff: {analysis.metrics.spectral_rolloff_hz.toFixed(0)} Hz
                    </span>
                  </div>
                </div>
              </div>

              {/* CNN Acoustic Condition Classification */}
              <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-[10px] font-mono uppercase text-slate-400">CNN Model</span>
                    <h3 className="text-xs font-semibold text-white uppercase tracking-wider">
                      Acoustic Condition Classifier ({analysis.classification.model_name})
                    </h3>
                  </div>
                  <span className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold uppercase ${
                    analysis.classification.primary_condition === 'Clean Speech'
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                  }`}>
                    {analysis.classification.primary_condition}
                  </span>
                </div>

                <div className="space-y-2">
                  {Object.entries(analysis.classification.condition_probabilities).map(([cond, prob]) => (
                    <div key={cond} className="space-y-1">
                      <div className="flex justify-between text-xs font-mono">
                        <span className="text-slate-300">{cond}</span>
                        <span className="text-white font-semibold">{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${cond === 'Clean Speech' ? 'bg-violet-400' : 'bg-slate-500'}`}
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="h-96 rounded-2xl bg-[#111827] border border-[#1f293d] p-8 flex flex-col items-center justify-center text-center">
              <Volume2 className="w-12 h-12 text-slate-600 mb-3" />
              <h3 className="text-sm font-semibold text-slate-200">No Audio Loaded</h3>
              <p className="text-xs text-slate-500 max-w-sm mt-1">
                Record a microphone segment or click "Load Sample Audio" above to analyze speech quality.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
