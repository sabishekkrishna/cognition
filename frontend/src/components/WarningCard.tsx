import React from 'react';
import { AlertTriangle, AlertCircle, ArrowRightCircle, ShieldAlert } from 'lucide-react';
import { MetricWarning } from '../types';

interface WarningCardProps {
  warning: MetricWarning;
}

export const WarningCard: React.FC<WarningCardProps> = ({ warning }) => {
  const isSevere = warning.severity === 'warning';

  return (
    <div className={`p-4 rounded-xl border transition-all ${
      isSevere
        ? 'bg-rose-950/20 border-rose-500/30 text-rose-200'
        : 'bg-amber-950/20 border-amber-500/30 text-amber-200'
    }`}>
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2">
          {isSevere ? (
            <ShieldAlert className="w-5 h-5 text-rose-400 shrink-0" />
          ) : (
            <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
          )}
          <div>
            <h4 className="text-xs font-semibold tracking-wide uppercase text-white">
              {warning.subsystem.toUpperCase()} HAZARD: {warning.metric.replace('_', ' ').toUpperCase()}
            </h4>
            <div className="flex items-center gap-2 text-[11px] font-mono mt-0.5">
              <span className="text-slate-400">Measured:</span>
              <span className="font-semibold text-white">{warning.measured_value.toFixed(1)}</span>
              <span className="text-slate-500">|</span>
              <span className="text-slate-400">Expected:</span>
              <span className="text-slate-300">{warning.expected_range}</span>
            </div>
          </div>
        </div>

        <span className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase font-bold tracking-wider ${
          isSevere
            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
            : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
        }`}>
          {warning.severity}
        </span>
      </div>

      {/* Explanation */}
      <p className="text-xs text-slate-300 leading-relaxed mb-3 pl-7">
        {warning.explanation}
      </p>

      {/* Recommended Corrective Action */}
      <div className="flex items-start gap-2 bg-slate-900/60 p-2.5 rounded-lg border border-slate-800 text-xs ml-7">
        <ArrowRightCircle className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
        <div>
          <span className="text-[11px] font-semibold text-cyan-400 uppercase tracking-wide mr-1.5">
            Recommended Action:
          </span>
          <span className="text-slate-300">{warning.recommendation}</span>
        </div>
      </div>
    </div>
  );
};
