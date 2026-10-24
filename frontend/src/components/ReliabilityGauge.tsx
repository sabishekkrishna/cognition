import React from 'react';

interface ReliabilityGaugeProps {
  score: number;
  label: string;
  size?: number;
  showRisk?: boolean;
}

export const ReliabilityGauge: React.FC<ReliabilityGaugeProps> = ({
  score,
  label,
  size = 140,
  showRisk = false
}) => {
  const clampedScore = Math.max(0, Math.min(100, score));
  const riskScore = 100 - clampedScore;
  const strokeWidth = 10;
  const radius = (size - strokeWidth * 2) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (clampedScore / 100) * circumference;

  const getColor = (s: number) => {
    if (s >= 80) return '#10b981'; // emerald
    if (s >= 60) return '#f59e0b'; // amber
    return '#f43f5e'; // rose
  };

  const currentColor = getColor(clampedScore);

  return (
    <div className="flex flex-col items-center justify-center p-3">
      <div className="relative" style={{ width: size, height: size }}>
        <svg className="transform -rotate-90" width={size} height={size}>
          {/* Background track */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="#1f293d"
            strokeWidth={strokeWidth}
            fill="transparent"
          />
          {/* Animated progress arc */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={currentColor}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            fill="transparent"
            className="transition-all duration-700 ease-out"
          />
        </svg>

        {/* Center score readout */}
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          <span className="text-2xl font-bold font-mono tracking-tight text-white">
            {clampedScore.toFixed(0)}
          </span>
          <span className="text-[10px] uppercase font-mono tracking-wider text-slate-400">
            {showRisk ? `Risk: ${riskScore.toFixed(0)}` : '/ 100'}
          </span>
        </div>
      </div>
      <span className="text-xs font-medium text-slate-300 mt-2 text-center">{label}</span>
    </div>
  );
};
