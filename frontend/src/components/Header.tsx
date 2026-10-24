import React from 'react';
import { Shield, Cpu, Activity, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { SystemOverallStatus } from '../types';

interface HeaderProps {
  status: SystemOverallStatus | null;
  activeProviderName: string;
}

export const Header: React.FC<HeaderProps> = ({ status, activeProviderName }) => {
  const overallState = status?.overall_state || 'normal';
  const score = status?.overall_reliability_score ?? 100;

  const getStatusBadge = () => {
    switch (overallState) {
      case 'warning':
        return {
          bg: 'bg-rose-500/15 border-rose-500/40 text-rose-400',
          dot: 'bg-rose-400 animate-ping',
          icon: AlertTriangle,
          label: 'RELIABILITY WARNING'
        };
      case 'caution':
        return {
          bg: 'bg-amber-500/15 border-amber-500/40 text-amber-400',
          dot: 'bg-amber-400',
          icon: AlertTriangle,
          label: 'ELEVATED RISK / CAUTION'
        };
      case 'unavailable':
        return {
          bg: 'bg-slate-700/50 border-slate-600 text-slate-300',
          dot: 'bg-slate-400',
          icon: Activity,
          label: 'MONITORING DISCONNECTED'
        };
      default:
        return {
          bg: 'bg-emerald-500/15 border-emerald-500/40 text-emerald-400',
          dot: 'bg-emerald-400',
          icon: CheckCircle2,
          label: 'ALL SYSTEMS NORMAL'
        };
    }
  };

  const badge = getStatusBadge();
  const Icon = badge.icon;

  return (
    <header className="h-16 px-6 bg-[#0b0f19] border-b border-[#1f293d] flex items-center justify-between z-10 shrink-0">
      {/* System Status Pill */}
      <div className="flex items-center gap-4">
        <div className={`flex items-center gap-2.5 px-3.5 py-1.5 rounded-full border text-xs font-semibold tracking-wide ${badge.bg}`}>
          <span className="relative flex h-2 w-2">
            <span className={`relative inline-flex rounded-full h-2 w-2 ${badge.dot}`}></span>
          </span>
          <Icon className="w-3.5 h-3.5" />
          <span>{badge.label}</span>
          <span className="font-mono ml-1.5 opacity-80">({score.toFixed(0)}/100)</span>
        </div>

        {/* Subsystem Quick indicators */}
        <div className="hidden md:flex items-center gap-3 text-xs text-slate-400 font-mono">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500">IMG:</span>
            <span className={status?.image_state === 'warning' ? 'text-rose-400 font-bold' : status?.image_state === 'caution' ? 'text-amber-400' : 'text-emerald-400'}>
              {status?.image_reliability_score.toFixed(0)}%
            </span>
          </div>
          <span className="text-slate-700">|</span>
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500">AUD:</span>
            <span className={status?.audio_state === 'warning' ? 'text-rose-400 font-bold' : status?.audio_state === 'caution' ? 'text-amber-400' : 'text-emerald-400'}>
              {status?.audio_reliability_score.toFixed(0)}%
            </span>
          </div>
          <span className="text-slate-700">|</span>
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500">LLM:</span>
            <span className="text-cyan-400">
              {status?.llm_state.toUpperCase()}
            </span>
          </div>
        </div>
      </div>

      {/* Right controls: Local LLM Engine & Privacy Badge */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/60 border border-[#1f293d] text-xs text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-blue-400" />
          <span className="text-slate-400">LLM Provider:</span>
          <span className="font-mono font-medium text-slate-200">{activeProviderName}</span>
        </div>

        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-950/30 border border-emerald-800/40 text-xs text-emerald-400 font-medium">
          <Shield className="w-3.5 h-3.5" />
          <span>Local-Only / Zero Telemetry</span>
        </div>
      </div>
    </header>
  );
};
