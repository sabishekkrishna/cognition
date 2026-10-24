import React from 'react';
import {
  LayoutDashboard,
  Image as ImageIcon,
  Mic,
  Video,
  Bot,
  Sliders,
  Database,
  History,
  Settings,
  ShieldCheck
} from 'lucide-react';

interface NavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  overallState: string;
}

export const Navigation: React.FC<NavigationProps> = ({
  activeTab,
  setActiveTab,
  overallState
}) => {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'image-lab', label: 'Image Lab', icon: ImageIcon },
    { id: 'audio-lab', label: 'Audio Lab', icon: Mic },
    { id: 'live-monitor', label: 'Live Monitor', icon: Video },
    { id: 'llm-console', label: 'Local LLM Console', icon: Bot },
    { id: 'baselines', label: 'Baseline Manager', icon: Sliders },
    { id: 'datasets', label: 'Datasets & Models', icon: Database },
    { id: 'events', label: 'Event History', icon: History },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside className="w-64 bg-[#0d1322] border-r border-[#1f293d] flex flex-col justify-between shrink-0">
      <div>
        {/* Brand header */}
        <div className="h-16 px-5 flex items-center gap-3 border-b border-[#1f293d]">
          <div className="w-9 h-9 rounded-lg bg-blue-600/20 border border-blue-500/40 flex items-center justify-center text-blue-400">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <h1 className="font-semibold text-sm tracking-wide text-white">RELIABILITY</h1>
            <p className="text-[10px] uppercase font-mono tracking-wider text-slate-400">On-Device AI Monitor</p>
          </div>
        </div>

        {/* Navigation list */}
        <nav className="p-3 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30 font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 border border-transparent'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-blue-400' : 'text-slate-500'}`} />
                <span>{item.label}</span>
                {item.id === 'live-monitor' && (
                  <span className="ml-auto w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer System Status Card */}
      <div className="p-4 border-t border-[#1f293d] bg-slate-900/40">
        <div className="flex items-center justify-between text-xs mb-2">
          <span className="text-slate-400">System State:</span>
          <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-medium uppercase ${
            overallState === 'normal'
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
              : overallState === 'caution'
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
              : overallState === 'warning'
              ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
              : 'bg-slate-700 text-slate-300'
          }`}>
            {overallState}
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11px] text-slate-400">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400"></div>
          <span>Local Air-Gapped Mode</span>
        </div>
      </div>
    </aside>
  );
};
