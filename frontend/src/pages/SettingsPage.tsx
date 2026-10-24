import React, { useState, useEffect } from 'react';
import {
  Settings, Save, Shield, HardDrive, Cpu, AlertTriangle,
  RotateCcw, CheckCircle2, Lock
} from 'lucide-react';
import { api } from '../services/api';

interface SettingsPageProps {
  onSettingsSaved?: () => void;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({ onSettingsSaved }) => {
  const [settings, setSettings] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [saved, setSaved] = useState(false);
  const [testStatus, setTestStatus] = useState<string | null>(null);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setLoading(true);
      const data = await api.getSettings();
      setSettings(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (key: string, value: string) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
    setSaved(false);
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setLoading(true);
      await api.updateSettings(settings);
      setSaved(true);
      onSettingsSaved?.();
      setTimeout(() => setSaved(false), 3000);
    } catch (err: any) {
      alert('Save error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const testEndpoint = async (provider: string, url: string) => {
    try {
      setTestStatus(`Testing ${provider}...`);
      const res = await api.testLLMConnection({
        provider,
        base_url: url,
        allow_remote: settings.allow_remote_llm === 'true'
      });
      if (res.overall_status === 'connected') {
        setTestStatus(`✓ ${provider} connected! (${res.discovered_models.length} models)`);
      } else {
        const firstFail = res.steps.find((s) => s.status === 'fail');
        setTestStatus(`✕ ${provider} failed: ${firstFail?.message || 'Connection refused'}`);
      }
    } catch (err: any) {
      setTestStatus(`✕ ${provider} error: ${err.message}`);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1f293d]">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2.5">
            System Settings & Local Thresholds
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 border border-slate-700 text-slate-300 font-mono">
              Local Storage
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Configure alert sensitivities, hysteresis windows, local model server URLs, and data retention limits.
          </p>
        </div>

        {/* Local Only Guarantee Badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-emerald-950/30 border border-emerald-800/40 text-emerald-400 text-xs font-mono font-medium">
          <Lock className="w-3.5 h-3.5" />
          <span>Local Air-Gapped Mode</span>
        </div>
      </div>

      <form onSubmit={handleSave} className="space-y-6 text-xs">
        {/* Local LLM Server Addresses */}
        <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
            <Cpu className="w-4 h-4 text-emerald-400" />
            Local Model Endpoints
          </h3>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Specify local loopback addresses for Ollama and OpenAI-compatible inference servers (e.g. LM Studio, llama.cpp). External cloud URLs are strictly prohibited.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-[11px] font-mono text-slate-400">Ollama Local Daemon URL:</label>
                <button
                  type="button"
                  onClick={() => testEndpoint('ollama', settings.ollama_url || 'http://127.0.0.1:11434')}
                  className="text-[10px] text-blue-400 hover:text-blue-300 font-mono"
                >
                  Test Reachability
                </button>
              </div>
              <input
                type="text"
                value={settings.ollama_url || 'http://127.0.0.1:11434'}
                onChange={(e) => handleChange('ollama_url', e.target.value)}
                className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-white font-mono focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-[11px] font-mono text-slate-400">OpenAI-Compat / LM Studio URL:</label>
                <button
                  type="button"
                  onClick={() => testEndpoint('lm_studio', settings.openai_compat_url || 'http://127.0.0.1:1234/v1')}
                  className="text-[10px] text-blue-400 hover:text-blue-300 font-mono"
                >
                  Test Reachability
                </button>
              </div>
              <input
                type="text"
                value={settings.openai_compat_url || 'http://127.0.0.1:1234/v1'}
                onChange={(e) => handleChange('openai_compat_url', e.target.value)}
                className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-white font-mono focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-[11px] font-mono text-slate-400">Custom Local HTTP API URL:</label>
                <button
                  type="button"
                  onClick={() => testEndpoint('custom_http', settings.custom_llm_url || 'http://localhost:8080')}
                  className="text-[10px] text-blue-400 hover:text-blue-300 font-mono"
                >
                  Test Reachability
                </button>
              </div>
              <input
                type="text"
                value={settings.custom_llm_url || 'http://localhost:8080'}
                onChange={(e) => handleChange('custom_llm_url', e.target.value)}
                placeholder="http://localhost:8080"
                className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-white font-mono focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="text-[11px] font-mono text-slate-400 block mb-1">Custom API Base Path:</label>
              <input
                type="text"
                value={settings.custom_llm_path || ''}
                onChange={(e) => handleChange('custom_llm_path', e.target.value)}
                placeholder="e.g. /v1, /api, or empty"
                className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-white font-mono focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          {/* SSRF Protection Policy */}
          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1.5">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="allow_remote_llm"
                checked={settings.allow_remote_llm === 'true'}
                onChange={(e) => handleChange('allow_remote_llm', e.target.checked ? 'true' : 'false')}
                className="rounded bg-slate-800 border-slate-700 text-blue-500"
              />
              <label htmlFor="allow_remote_llm" className="text-xs font-semibold text-slate-200 cursor-pointer">
                Permit Remote / Non-Loopback LLM IP Addresses (Overrides SSRF Safeguard)
              </label>
            </div>
            <p className="text-[10px] text-slate-500 pl-5">
              By default, connections to non-loopback addresses (outside 127.0.0.1 and localhost) are rejected to prevent internal network scanning and SSRF.
            </p>
          </div>

          {testStatus && (
            <p className="text-xs font-mono p-2 rounded bg-slate-900 border border-slate-800 text-slate-300">
              {testStatus}
            </p>
          )}
        </div>

        {/* Sensitivity & Hysteresis Thresholds */}
        <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
            <Settings className="w-4 h-4 text-blue-400" />
            Monitoring & Hysteresis Thresholds
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="text-[11px] font-mono text-slate-400 block mb-1">Minimum Sharpness Threshold:</label>
              <input
                type="number"
                step="5.0"
                value={settings.sharpness_min_threshold || '60.0'}
                onChange={(e) => handleChange('sharpness_min_threshold', e.target.value)}
                className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-white font-mono focus:outline-none focus:border-blue-500"
              />
              <span className="text-[10px] text-slate-500 mt-1 block">Laplacian variance threshold for blur alarms.</span>
            </div>

            <div>
              <label className="text-[11px] font-mono text-slate-400 block mb-1">Minimum Speech SNR Threshold (dB):</label>
              <input
                type="number"
                step="1.0"
                value={settings.snr_min_threshold || '12.0'}
                onChange={(e) => handleChange('snr_min_threshold', e.target.value)}
                className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-white font-mono focus:outline-none focus:border-blue-500"
              />
              <span className="text-[10px] text-slate-500 mt-1 block">Signal-to-noise ratio required for reliable ASR.</span>
            </div>

            <div>
              <label className="text-[11px] font-mono text-slate-400 block mb-1">Hysteresis Rolling Window Size:</label>
              <input
                type="number"
                min="3"
                max="15"
                value={settings.hysteresis_window_size || '5'}
                onChange={(e) => handleChange('hysteresis_window_size', e.target.value)}
                className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-white font-mono focus:outline-none focus:border-blue-500"
              />
              <span className="text-[10px] text-slate-500 mt-1 block">Number of frames averaged in moving buffer.</span>
            </div>

            <div>
              <label className="text-[11px] font-mono text-slate-400 block mb-1">Sustained Anomaly Count (Hysteresis):</label>
              <input
                type="number"
                min="2"
                max="10"
                value={settings.hysteresis_threshold || '3'}
                onChange={(e) => handleChange('hysteresis_threshold', e.target.value)}
                className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-white font-mono focus:outline-none focus:border-blue-500"
              />
              <span className="text-[10px] text-slate-500 mt-1 block">Consecutive abnormal frames before alert escalation.</span>
            </div>
          </div>
        </div>

        {/* Data Retention & Local Paths */}
        <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-violet-400" />
            Storage & Data Retention
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="text-[11px] font-mono text-slate-400 block mb-1">Event Audit Retention (Days):</label>
              <input
                type="number"
                min="1"
                max="365"
                value={settings.data_retention_days || '30'}
                onChange={(e) => handleChange('data_retention_days', e.target.value)}
                className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-white font-mono focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="text-[11px] font-mono text-slate-400 block mb-1">Local Database Location:</label>
              <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 font-mono text-[11px] truncate">
                ./data/monitor.db (SQLite 3)
              </div>
            </div>
          </div>
        </div>

        {/* Submit */}
        <div className="flex items-center justify-between pt-2">
          {saved ? (
            <span className="text-emerald-400 flex items-center gap-1.5 text-xs font-medium">
              <CheckCircle2 className="w-4 h-4" /> Settings successfully saved to disk.
            </span>
          ) : (
            <span />
          )}

          <button
            type="submit"
            disabled={loading}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition shadow-sm"
          >
            <Save className="w-3.5 h-3.5" />
            Save Configuration
          </button>
        </div>
      </form>
    </div>
  );
};
