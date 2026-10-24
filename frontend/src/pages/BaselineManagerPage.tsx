import React, { useState, useEffect } from 'react';
import {
  Sliders, Plus, CheckCircle2, Trash2, RefreshCw, Upload,
  Shield, Layers, ChevronRight, AlertTriangle
} from 'lucide-react';
import { BaselineInfo } from '../types';
import { api } from '../services/api';

export const BaselineManagerPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'image' | 'audio'>('image');
  const [baselines, setBaselines] = useState<BaselineInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedBaseline, setSelectedBaseline] = useState<BaselineInfo | null>(null);

  // New Baseline Modal
  const [isCreating, setIsCreating] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [files, setFiles] = useState<File[]>([]);

  useEffect(() => {
    loadBaselines();
  }, [activeTab]);

  const loadBaselines = async () => {
    try {
      setLoading(true);
      const data = await api.listBaselines(activeTab);
      setBaselines(data);
      if (data.length > 0) {
        const active = data.find((b) => b.is_active) || data[0];
        setSelectedBaseline(active);
      } else {
        setSelectedBaseline(null);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleActivate = async (id: string) => {
    try {
      await api.activateBaseline(id);
      await loadBaselines();
    } catch (err: any) {
      alert('Activation error: ' + err.message);
    }
  };

  const handleDelete = async (id: string) => {
    if (confirm('Are you sure you want to delete this baseline profile? This action cannot be undone.')) {
      try {
        await api.deleteBaseline(id);
        await loadBaselines();
      } catch (err: any) {
        alert('Delete error: ' + err.message);
      }
    }
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || files.length === 0) {
      alert('Please provide a profile name and at least one calibration file.');
      return;
    }

    try {
      setLoading(true);
      await api.createBaseline(name, activeTab, description, files);
      setIsCreating(false);
      setName('');
      setDescription('');
      setFiles([]);
      await loadBaselines();
    } catch (err: any) {
      alert('Calibration error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Title & Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1f293d]">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2.5">
            Baseline Calibration Manager
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 font-mono">
              Statistical Envelopes
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Establish clean-room baseline profiles, calibrate robust distributions (medians, MAD, percentiles), and prevent baseline replacement accidents.
          </p>
        </div>

        {/* Tab switcher: Image vs Audio */}
        <div className="flex items-center gap-2 bg-[#111827] p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setActiveTab('image')}
            className={`px-4 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition ${
              activeTab === 'image'
                ? 'bg-cyan-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Image Profiles
          </button>
          <button
            onClick={() => setActiveTab('audio')}
            className={`px-4 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition ${
              activeTab === 'audio'
                ? 'bg-violet-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Audio Profiles
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Baselines List & Create Button */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-white uppercase tracking-wider">
              {activeTab.toUpperCase()} PROFILES ({baselines.length})
            </span>
            <button
              onClick={() => setIsCreating(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white transition shadow-sm"
            >
              <Plus className="w-3.5 h-3.5" /> Calibrate New
            </button>
          </div>

          <div className="space-y-2">
            {baselines.map((b) => (
              <div
                key={b.id}
                onClick={() => setSelectedBaseline(b)}
                className={`p-4 rounded-xl border cursor-pointer transition ${
                  selectedBaseline?.id === b.id
                    ? 'bg-slate-800/80 border-blue-500/50 shadow-sm'
                    : 'bg-[#111827] border-[#1f293d] hover:bg-slate-800/40'
                }`}
              >
                <div className="flex items-start justify-between gap-2 mb-1.5">
                  <h4 className="text-xs font-semibold text-white">{b.name}</h4>
                  {b.is_active ? (
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                      ACTIVE
                    </span>
                  ) : (
                    <button
                      onClick={(e) => { e.stopPropagation(); handleActivate(b.id); }}
                      className="px-2 py-0.5 rounded text-[10px] font-mono text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 transition"
                    >
                      Set Active
                    </button>
                  )}
                </div>

                <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                  {b.description || 'No description provided.'}
                </p>

                <div className="flex items-center justify-between mt-3 pt-2 border-t border-slate-800/60 text-[10px] font-mono text-slate-400">
                  <span>Samples: {b.sample_count}</span>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDelete(b.id); }}
                      className="text-slate-500 hover:text-rose-400 transition"
                      title="Delete profile"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Columns: Selected Baseline Detailed Distributions */}
        <div className="lg:col-span-2">
          {selectedBaseline ? (
            <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-5">
              <div className="flex items-start justify-between pb-3 border-b border-[#1f293d]">
                <div>
                  <span className="text-[10px] font-mono uppercase text-slate-400">Inspecting Baseline Profile</span>
                  <h3 className="text-base font-bold text-white mt-0.5">{selectedBaseline.name}</h3>
                  <p className="text-xs text-slate-400 mt-1">{selectedBaseline.description}</p>
                </div>
                {selectedBaseline.is_active ? (
                  <span className="px-3 py-1 rounded-lg text-xs font-mono font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5" /> ACTIVE PROFILE
                  </span>
                ) : (
                  <button
                    onClick={() => handleActivate(selectedBaseline.id)}
                    className="px-3 py-1 rounded-lg text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white transition"
                  >
                    Activate Profile
                  </button>
                )}
              </div>

              {/* Stats Table */}
              <div>
                <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
                  Robust Statistical Envelopes (Median, MAD & Percentiles)
                </h4>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead>
                      <tr className="border-b border-slate-800 text-[11px] text-slate-400">
                        <th className="pb-2 font-medium">METRIC</th>
                        <th className="pb-2 font-medium">MEDIAN</th>
                        <th className="pb-2 font-medium">MAD</th>
                        <th className="pb-2 font-medium">5TH %ILE</th>
                        <th className="pb-2 font-medium">95TH %ILE</th>
                        <th className="pb-2 font-medium">RANGE [MIN, MAX]</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 text-slate-300">
                      {Object.entries(selectedBaseline.stats).map(([metricKey, metricStats]: any) => (
                        <tr key={metricKey} className="hover:bg-slate-800/30 transition">
                          <td className="py-2.5 font-sans font-semibold text-white capitalize">
                            {metricKey.replace('_', ' ')}
                          </td>
                          <td className="py-2.5 text-cyan-400 font-bold">{metricStats.median}</td>
                          <td className="py-2.5 text-slate-400">{metricStats.mad}</td>
                          <td className="py-2.5 text-slate-300">{metricStats.p5}</td>
                          <td className="py-2.5 text-slate-300">{metricStats.p95}</td>
                          <td className="py-2.5 text-slate-400">[{metricStats.min}, {metricStats.max}]</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-64 rounded-2xl bg-[#111827] border border-[#1f293d] p-6 flex items-center justify-center text-slate-500 text-xs">
              Select or calibrate a baseline profile.
            </div>
          )}
        </div>
      </div>

      {/* Modal: Calibrate New Profile */}
      {isCreating && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-4">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Sliders className="w-4 h-4 text-blue-400" />
              Calibrate New {activeTab.toUpperCase()} Baseline
            </h3>

            <form onSubmit={handleCreate} className="space-y-4 text-xs">
              <div>
                <label className="text-[11px] font-mono text-slate-400 block mb-1">Profile Name:</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Factory Floor Camera 01 (Morning Lighting)"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-white focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="text-[11px] font-mono text-slate-400 block mb-1">Description / Operating Environment:</label>
                <textarea
                  rows={2}
                  placeholder="Lighting conditions, lens characteristics, or room acoustics..."
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl p-2.5 text-white focus:outline-none focus:border-blue-500 resize-none font-sans"
                />
              </div>

              <div>
                <label className="text-[11px] font-mono text-slate-400 block mb-1">
                  Upload Calibration Samples ({files.length} selected):
                </label>
                <input
                  type="file"
                  multiple
                  accept={activeTab === 'image' ? 'image/*' : 'audio/*'}
                  onChange={(e) => {
                    if (e.target.files) setFiles(Array.from(e.target.files));
                  }}
                  className="w-full text-slate-400 file:mr-3 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-blue-600/20 file:text-blue-300 hover:file:bg-blue-600/30 cursor-pointer"
                />
                <p className="text-[10px] text-slate-500 mt-1">
                  Select 5-20 clean reference files to calculate robust feature distributions.
                </p>
              </div>

              <div className="flex justify-end gap-2 pt-3 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setIsCreating(false)}
                  className="px-4 py-2 rounded-xl bg-slate-800 text-slate-300 font-medium hover:bg-slate-700 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="px-4 py-2 rounded-xl bg-blue-600 text-white font-semibold hover:bg-blue-500 transition shadow-sm"
                >
                  {loading ? 'Calibrating...' : 'Calibrate Baseline'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
