import React, { useState, useEffect } from 'react';
import {
  ShieldAlert, ShieldCheck, Eye, Mic, Bot, Activity,
  AlertTriangle, RefreshCw, ArrowUpRight, ArrowDownRight,
  ExternalLink, Layers
} from 'lucide-react';
import { SystemOverallStatus, EventItem } from '../types';
import { api } from '../services/api';
import { ReliabilityGauge } from '../components/ReliabilityGauge';
import { WarningCard } from '../components/WarningCard';

interface OverviewPageProps {
  status: SystemOverallStatus | null;
  isOnline?: boolean;
  lastUpdated?: Date;
  onNavigate: (tab: string) => void;
  onRefresh: () => void;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  status,
  isOnline = true,
  lastUpdated,
  onNavigate,
  onRefresh
}) => {
  const [events, setEvents] = useState<EventItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadEvents();
  }, []);

  const loadEvents = async () => {
    try {
      setLoading(true);
      const data = await api.listEvents('all', 'all', 8);
      setEvents(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const overallScore = isOnline ? (status?.overall_reliability_score ?? 100) : 0;
  const imageScore = isOnline ? (status?.image_reliability_score ?? 100) : 0;
  const audioScore = isOnline ? (status?.audio_reliability_score ?? 100) : 0;
  const timeFormatted = lastUpdated ? lastUpdated.toLocaleTimeString() : (status?.last_updated_formatted || 'Connecting...');

  return (
    <div className="space-y-6">
      {/* Offline Alert Banner */}
      {!isOnline && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 flex items-center gap-3 animate-pulse">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
          <div className="flex-1">
            <p className="text-xs font-semibold text-rose-200">Backend Monitoring Service Disconnected</p>
            <p className="text-[11px] text-rose-300/80 mt-0.5">
              Unable to reach local engine daemon at 127.0.0.1:8000. Active perception telemetry is paused. Showing last cached state; system will automatically reconnect.
            </p>
          </div>
          <button
            onClick={() => { onRefresh(); loadEvents(); }}
            className="px-3 py-1 rounded bg-rose-900/60 hover:bg-rose-800 text-xs font-mono text-rose-200 border border-rose-700/60 transition"
          >
            Retry Now
          </button>
        </div>
      )}

      {/* Top Banner & Quick Metrics */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1f293d]">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2.5">
            Reliability Overview Dashboard
            <span className={`text-xs px-2.5 py-0.5 rounded-full border font-mono font-medium ${
              isOnline ? 'bg-blue-500/10 border-blue-500/30 text-blue-400' : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
            }`}>
              {isOnline ? 'Real-Time Telemetry' : 'Disconnected'}
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Continuous on-device perception monitoring for input degradation and model reasoning unreliability.
            <span className="ml-2 text-slate-500 font-mono">Last update: {timeFormatted}</span>
          </p>
        </div>

        <button
          onClick={() => { onRefresh(); loadEvents(); }}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 font-medium transition"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh Status
        </button>
      </div>

      {/* 4 Primary Subsystem Gauge Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Overall System Card */}
        <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-4 flex flex-col items-center relative overflow-hidden shadow-sm">
          <div className="w-full flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-semibold uppercase tracking-wider text-[11px]">Overall Reliability</span>
            <Activity className="w-4 h-4 text-blue-400" />
          </div>
          <ReliabilityGauge score={overallScore} label="Global Health Index" size={130} showRisk />
          <div className="w-full mt-2 pt-2 border-t border-slate-800/80 flex justify-between text-[11px] text-slate-400 font-mono">
            <span>State:</span>
            <span className={`uppercase font-bold ${
              !isOnline ? 'text-rose-400' : status?.overall_state === 'warning' ? 'text-rose-400' : status?.overall_state === 'caution' ? 'text-amber-400' : 'text-emerald-400'
            }`}>
              {isOnline ? (status?.overall_state || 'NORMAL') : 'OFFLINE'}
            </span>
          </div>
        </div>

        {/* Visual Input Card */}
        <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-4 flex flex-col items-center relative overflow-hidden shadow-sm">
          <div className="w-full flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-semibold uppercase tracking-wider text-[11px]">Visual Perception</span>
            <Eye className="w-4 h-4 text-cyan-400" />
          </div>
          <ReliabilityGauge score={imageScore} label="Image Quality Score" size={130} />
          
          <div className="w-full my-1 px-1 text-[10px] text-slate-400 flex flex-col gap-0.5 border-t border-slate-800/50 pt-1.5 font-mono">
            <div className="flex justify-between truncate">
              <span className="text-slate-500">Baseline:</span>
              <span className="text-slate-300 truncate max-w-[120px]" title={status?.active_image_baseline?.name}>
                {status?.active_image_baseline?.name || 'Default Clear Room'}
              </span>
            </div>
            {status?.last_image_metrics && (
              <div className="flex justify-between text-slate-400 text-[9px]">
                <span>Bri: {status.last_image_metrics.brightness.toFixed(0)}</span>
                <span>Sha: {status.last_image_metrics.sharpness.toFixed(0)}</span>
                <span>Con: {status.last_image_metrics.contrast.toFixed(0)}</span>
              </div>
            )}
          </div>

          <div className="w-full mt-1 pt-1.5 border-t border-slate-800/80 flex justify-between items-center text-[11px] font-mono">
            <button
              onClick={() => onNavigate('image-lab')}
              className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1 text-[11px]"
            >
              Open Image Lab <ArrowUpRight className="w-3 h-3" />
            </button>
            <span className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-bold ${
              !isOnline ? 'bg-slate-800 text-slate-400' : status?.image_state === 'warning' ? 'bg-rose-500/20 text-rose-400' : 'bg-emerald-500/20 text-emerald-400'
            }`}>
              {isOnline ? (status?.image_state || 'NORMAL') : 'OFFLINE'}
            </span>
          </div>
        </div>

        {/* Acoustic Input Card */}
        <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-4 flex flex-col items-center relative overflow-hidden shadow-sm">
          <div className="w-full flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-semibold uppercase tracking-wider text-[11px]">Acoustic Analysis</span>
            <Mic className="w-4 h-4 text-violet-400" />
          </div>
          <ReliabilityGauge score={audioScore} label="Speech Suitability Score" size={130} />

          <div className="w-full my-1 px-1 text-[10px] text-slate-400 flex flex-col gap-0.5 border-t border-slate-800/50 pt-1.5 font-mono">
            <div className="flex justify-between truncate">
              <span className="text-slate-500">Baseline:</span>
              <span className="text-slate-300 truncate max-w-[120px]" title={status?.active_audio_baseline?.name}>
                {status?.active_audio_baseline?.name || 'Quiet Studio'}
              </span>
            </div>
            {status?.last_audio_metrics && (
              <div className="flex justify-between text-slate-400 text-[9px]">
                <span>RMS: {status.last_audio_metrics.rms_dbfs.toFixed(0)}dB</span>
                <span>SNR: {status.last_audio_metrics.snr_db.toFixed(0)}dB</span>
                <span>Clip: {status.last_audio_metrics.clipping_pct.toFixed(1)}%</span>
              </div>
            )}
          </div>

          <div className="w-full mt-1 pt-1.5 border-t border-slate-800/80 flex justify-between items-center text-[11px] font-mono">
            <button
              onClick={() => onNavigate('audio-lab')}
              className="text-violet-400 hover:text-violet-300 flex items-center gap-1 text-[11px]"
            >
              Open Audio Lab <ArrowUpRight className="w-3 h-3" />
            </button>
            <span className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-bold ${
              !isOnline ? 'bg-slate-800 text-slate-400' : status?.audio_state === 'warning' ? 'bg-rose-500/20 text-rose-400' : 'bg-emerald-500/20 text-emerald-400'
            }`}>
              {isOnline ? (status?.audio_state || 'NORMAL') : 'OFFLINE'}
            </span>
          </div>
        </div>

        {/* Local LLM Layer */}
        <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-4 flex flex-col justify-between shadow-sm">
          <div className="w-full flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-semibold uppercase tracking-wider text-[11px]">Foreground Local LLM</span>
            <Bot className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="my-auto py-2 text-center">
            <div className={`w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-2 ${
              isOnline ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400' : 'bg-slate-800 border border-slate-700 text-slate-400'
            }`}>
              <ShieldCheck className="w-6 h-6" />
            </div>
            <p className="text-xs font-semibold text-white">
              {status?.active_provider ? `Provider: ${status.active_provider.toUpperCase()}` : 'Coupled Guard'}
            </p>
            <p className="text-[11px] text-slate-400 mt-0.5">Input degradation fed to prompt guard</p>
          </div>
          <div className="w-full mt-2 pt-2 border-t border-slate-800/80 flex justify-between items-center text-[11px] font-mono">
            <button
              onClick={() => onNavigate('llm-console')}
              className="text-emerald-400 hover:text-emerald-300 flex items-center gap-1 text-[11px]"
            >
              Launch Console <ArrowUpRight className="w-3 h-3" />
            </button>
            <span className={status?.llm_state === 'warning' ? 'text-rose-400 font-bold' : 'text-slate-400'}>
              {isOnline ? (status?.llm_state?.toUpperCase() || 'READY') : 'OFFLINE'}
            </span>
          </div>
        </div>
      </div>

      {/* Active Warnings Panel */}
      <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider">
              Active Warnings & Degradation Alarms ({status?.active_warnings.length ?? 0})
            </h3>
          </div>
          <span className="text-[11px] text-slate-400">
            {status?.active_warnings.length === 0 ? 'No sustained anomalies detected' : 'Requires user intervention'}
          </span>
        </div>

        {status?.active_warnings && status.active_warnings.length > 0 ? (
          <div className="space-y-3">
            {status.active_warnings.map((w, i) => (
              <WarningCard key={i} warning={w} />
            ))}
          </div>
        ) : (
          <div className="p-8 rounded-xl bg-slate-900/30 border border-dashed border-slate-800 flex flex-col items-center justify-center text-center">
            <ShieldCheck className="w-8 h-8 text-emerald-400 mb-2" />
            <p className="text-xs font-medium text-slate-300">Clean Input Channels</p>
            <p className="text-[11px] text-slate-500 max-w-md mt-1">
              Visual and acoustic signals currently match normal calibrated baseline envelopes. Downstream AI responses have maximum reliability.
            </p>
          </div>
        )}
      </div>

      {/* Quick Launch & Recent Events Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Quick Diagnostic Launchers */}
        <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
          <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center gap-2">
            <Layers className="w-4 h-4 text-blue-400" />
            Quick Workflows
          </h3>

          <div className="space-y-2">
            <button
              onClick={() => onNavigate('image-lab')}
              className="w-full p-3 rounded-xl bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 text-left flex items-center justify-between transition group"
            >
              <div>
                <p className="text-xs font-medium text-white group-hover:text-blue-400 transition">Test Visual Degradations</p>
                <p className="text-[11px] text-slate-400">Inject synthetic blur, noise, or exposure tint</p>
              </div>
              <ArrowUpRight className="w-4 h-4 text-slate-500 group-hover:text-blue-400 transition" />
            </button>

            <button
              onClick={() => onNavigate('audio-lab')}
              className="w-full p-3 rounded-xl bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 text-left flex items-center justify-between transition group"
            >
              <div>
                <p className="text-xs font-medium text-white group-hover:text-violet-400 transition">Microphone & Noise Testing</p>
                <p className="text-[11px] text-slate-400">Check clipping, loudness dBFS, and speech VAD</p>
              </div>
              <ArrowUpRight className="w-4 h-4 text-slate-500 group-hover:text-violet-400 transition" />
            </button>

            <button
              onClick={() => onNavigate('live-monitor')}
              className="w-full p-3 rounded-xl bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 text-left flex items-center justify-between transition group"
            >
              <div>
                <p className="text-xs font-medium text-white group-hover:text-cyan-400 transition">Live Camera & Mic Stream</p>
                <p className="text-[11px] text-slate-400">Continuous rolling window & hysteresis</p>
              </div>
              <ArrowUpRight className="w-4 h-4 text-slate-500 group-hover:text-cyan-400 transition" />
            </button>

            <button
              onClick={() => onNavigate('llm-console')}
              className="w-full p-3 rounded-xl bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 text-left flex items-center justify-between transition group"
            >
              <div>
                <p className="text-xs font-medium text-white group-hover:text-emerald-400 transition">Interact with Local LLM</p>
                <p className="text-[11px] text-slate-400">Ollama, LM Studio, or Offline Mock</p>
              </div>
              <ArrowUpRight className="w-4 h-4 text-slate-500 group-hover:text-emerald-400 transition" />
            </button>
          </div>
        </div>

        {/* Live Recent Events Feed */}
        <div className="lg:col-span-2 bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-400" />
              Recent Reliability Events
            </h3>
            <button
              onClick={() => onNavigate('events')}
              className="text-xs text-blue-400 hover:text-blue-300 font-medium flex items-center gap-1"
            >
              Full History <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 font-mono text-[11px]">
                  <th className="pb-2.5 font-medium">TIMESTAMP</th>
                  <th className="pb-2.5 font-medium">SUBSYSTEM</th>
                  <th className="pb-2.5 font-medium">SEVERITY</th>
                  <th className="pb-2.5 font-medium">EVENT DETAILS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {events.length > 0 ? (
                  events.map((ev) => (
                    <tr key={ev.id} className="hover:bg-slate-800/30 transition">
                      <td className="py-2.5 text-slate-400 text-[11px] whitespace-nowrap">
                        {ev.time_formatted.split(' ')[1]}
                      </td>
                      <td className="py-2.5">
                        <span className="uppercase text-[10px] font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                          {ev.subsystem}
                        </span>
                      </td>
                      <td className="py-2.5">
                        <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded ${
                          ev.severity === 'warning'
                            ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                            : ev.severity === 'caution'
                            ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                            : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        }`}>
                          {ev.severity}
                        </span>
                      </td>
                      <td className="py-2.5 text-slate-200 text-xs font-sans">
                        {ev.title}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={4} className="py-8 text-center text-slate-500 font-sans">
                      No reliability events logged yet. Trigger an image or audio test to observe events.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
