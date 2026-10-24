import React, { useState, useEffect } from 'react';
import {
  Bot, Send, RefreshCw, CheckCircle2, AlertTriangle, XCircle,
  ThumbsUp, ThumbsDown, HelpCircle, ShieldAlert, Cpu, Sparkles, MessageSquare, Wrench
} from 'lucide-react';
import { LLMProviderInfo, LLMResponseInfo, LLMDiagnosticReport } from '../types';
import { api } from '../services/api';

interface LLMConsolePageProps {
  currentInputRiskScore?: number;
  currentInputSubsystem?: string;
}

export const LLMConsolePage: React.FC<LLMConsolePageProps> = ({
  currentInputRiskScore = 0.0,
  currentInputSubsystem = 'none'
}) => {
  const [providers, setProviders] = useState<LLMProviderInfo[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>('mock');
  const [providerConnected, setProviderConnected] = useState<boolean | null>(null);
  const [providerMessage, setProviderMessage] = useState<string>('');
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');

  // Diagnostic & Endpoint Configuration
  const [hostInput, setHostInput] = useState<string>('127.0.0.1');
  const [portInput, setPortInput] = useState<string>('11434');
  const [allowRemote, setAllowRemote] = useState<boolean>(false);
  const [diagnosticRunning, setDiagnosticRunning] = useState<boolean>(false);
  const [diagnosticReport, setDiagnosticReport] = useState<LLMDiagnosticReport | null>(null);
  const [showDiagnosticPanel, setShowDiagnosticPanel] = useState<boolean>(false);

  // Generation state
  const [prompt, setPrompt] = useState<string>('Analyze the reliability of the current visual component input.');
  const [systemPrompt, setSystemPrompt] = useState<string>('You are an expert on-device AI assistant. Answer accurately, concisely, and state any visual or acoustic uncertainties explicitly.');
  const [loading, setLoading] = useState<boolean>(false);
  const [currentResponse, setCurrentResponse] = useState<LLMResponseInfo | null>(null);

  // Human feedback state
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<string | null>(null);
  const [feedbackNotes, setFeedbackNotes] = useState<string>('');

  useEffect(() => {
    loadProviders();
  }, []);

  const loadProviders = async () => {
    try {
      const res = await api.getLLMProviders();
      setProviders(res.providers || []);
      const active = res.active_provider || 'mock';
      setSelectedProvider(active);
      updateDefaultPortForProvider(active);
      await checkHealthAndModels(active);
    } catch (err) {
      console.error(err);
    }
  };

  const updateDefaultPortForProvider = (provId: string) => {
    if (provId === 'ollama') setPortInput('11434');
    else if (provId === 'lm_studio') setPortInput('1234');
    else if (provId === 'llama_cpp') setPortInput('8080');
    else setPortInput('0');
  };

  const checkHealthAndModels = async (provId: string) => {
    try {
      const health = await api.checkLLMHealth(provId);
      setProviderConnected(health.is_connected);
      setProviderMessage(health.message);

      if (health.is_connected) {
        const modelsRes = await api.getLLMModels(provId);
        const mList = modelsRes.models || [];
        setModels(mList);
        if (mList.length > 0) {
          setSelectedModel(mList[0]);
        }
      } else {
        setModels([]);
        setSelectedModel('');
      }
    } catch (err: any) {
      setProviderConnected(false);
      setProviderMessage(err.message || 'Connection failed');
    }
  };

  const handleProviderChange = async (newProv: string) => {
    setSelectedProvider(newProv);
    updateDefaultPortForProvider(newProv);
    setDiagnosticReport(null);
    await checkHealthAndModels(newProv);
  };

  const runDiagnosticTest = async (saveAsActive = false) => {
    try {
      setDiagnosticRunning(true);
      setShowDiagnosticPanel(true);
      const report = await api.testLLMConnection({
        provider: selectedProvider,
        host: hostInput,
        port: parseInt(portInput, 10) || (selectedProvider === 'ollama' ? 11434 : 1234),
        allow_remote: allowRemote,
        save_if_successful: saveAsActive
      });
      setDiagnosticReport(report);

      if (report.overall_status === 'connected' || report.overall_status === 'warning') {
        setProviderConnected(true);
        setProviderMessage(`Diagnostic passed. Discovered ${report.discovered_models.length} model(s).`);
        setModels(report.discovered_models);
        if (report.discovered_models.length > 0) {
          setSelectedModel(report.discovered_models[0]);
        }
        if (saveAsActive) {
          await loadProviders();
        }
      } else {
        setProviderConnected(false);
        setProviderMessage('Diagnostic test failed. Inspect step results below.');
      }
    } catch (err: any) {
      setProviderConnected(false);
      setProviderMessage(err.message || 'Diagnostic execution failed');
    } finally {
      setDiagnosticRunning(false);
    }
  };

  const handleSendPrompt = async () => {
    if (!prompt.trim() || !selectedModel) return;
    setLoading(true);
    setFeedbackSubmitted(null);
    setCurrentResponse(null);

    try {
      const res = await api.generateLLM({
        provider: selectedProvider,
        model: selectedModel,
        prompt: prompt,
        system_prompt: systemPrompt,
        current_input_risk_score: currentInputRiskScore,
        current_input_subsystem: currentInputSubsystem
      });
      setCurrentResponse(res);
    } catch (err: any) {
      alert(err.message || 'LLM execution error');
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async (rating: 'correct' | 'incorrect' | 'uncertain') => {
    if (!currentResponse?.interaction_id) return;
    try {
      await api.submitLLMFeedback(currentResponse.interaction_id, rating, feedbackNotes);
      setFeedbackSubmitted(rating);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="space-y-6">
      {/* Title & Connection Status */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1f293d]">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2.5">
            Local LLM Provider Console & Output Monitor
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-mono">
              Ollama + OpenAI-Compat
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Connect directly to local loopback LLM daemons, stream prompts, and monitor response risk coupled to upstream input quality.
          </p>
        </div>

        {/* Upstream Risk Badge */}
        {currentInputRiskScore > 0 && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs">
            <span className="text-slate-400 font-mono">Coupled Input Risk:</span>
            <span className={`font-mono font-bold ${currentInputRiskScore > 40 ? 'text-rose-400' : 'text-amber-400'}`}>
              {currentInputRiskScore.toFixed(0)}/100 ({currentInputSubsystem})
            </span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Provider Settings & Model Selector */}
        <div className="space-y-5">
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
              <Cpu className="w-4 h-4 text-emerald-400" />
              Provider Configuration
            </h3>

            {/* Provider Dropdown */}
            <div>
              <label className="text-[11px] font-mono text-slate-400 block mb-1">Select Local Provider:</label>
              <select
                value={selectedProvider}
                onChange={(e) => handleProviderChange(e.target.value)}
                className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-blue-500"
              >
                {providers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Custom Host & Port Configuration */}
            {selectedProvider !== 'mock' && (
              <div className="space-y-2 p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                <div className="grid grid-cols-3 gap-2">
                  <div className="col-span-2">
                    <label className="text-[10px] font-mono text-slate-400 block mb-1">Target Host:</label>
                    <input
                      type="text"
                      value={hostInput}
                      onChange={(e) => setHostInput(e.target.value)}
                      placeholder="127.0.0.1"
                      className="w-full bg-[#0d1322] border border-[#1f293d] rounded-lg px-2.5 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] font-mono text-slate-400 block mb-1">Port:</label>
                    <input
                      type="text"
                      value={portInput}
                      onChange={(e) => setPortInput(e.target.value)}
                      placeholder="11434"
                      className="w-full bg-[#0d1322] border border-[#1f293d] rounded-lg px-2.5 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-blue-500"
                    />
                  </div>
                </div>

                <div className="flex items-center gap-2 pt-1">
                  <input
                    type="checkbox"
                    id="allowRemote"
                    checked={allowRemote}
                    onChange={(e) => setAllowRemote(e.target.checked)}
                    className="rounded bg-slate-800 border-slate-700 text-blue-500 text-xs"
                  />
                  <label htmlFor="allowRemote" className="text-[10px] text-slate-400 cursor-pointer">
                    Permit non-loopback IP (Overrides SSRF protection)
                  </label>
                </div>

                <div className="flex items-center gap-2 pt-1.5">
                  <button
                    onClick={() => runDiagnosticTest(false)}
                    disabled={diagnosticRunning}
                    className="flex-1 py-1.5 rounded-lg bg-blue-600/30 hover:bg-blue-600/40 border border-blue-500/40 text-[11px] text-blue-200 font-medium flex items-center justify-center gap-1.5 transition"
                  >
                    <Wrench className={`w-3.5 h-3.5 ${diagnosticRunning ? 'animate-spin' : ''}`} />
                    {diagnosticRunning ? 'Testing 5 Steps...' : '5-Step Diagnostic Probe'}
                  </button>
                  <button
                    onClick={() => runDiagnosticTest(true)}
                    disabled={diagnosticRunning}
                    className="py-1.5 px-2.5 rounded-lg bg-emerald-600/30 hover:bg-emerald-600/40 border border-emerald-500/40 text-[11px] text-emerald-200 font-medium transition"
                    title="Validate and save as active provider"
                  >
                    Save & Activate
                  </button>
                </div>
              </div>
            )}

            {/* 5-Step Diagnostic Results Panel */}
            {diagnosticReport && (
              <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-white">Diagnostic Results</span>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${
                    diagnosticReport.overall_status === 'connected'
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      : diagnosticReport.overall_status === 'warning'
                      ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                      : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                  }`}>
                    {diagnosticReport.overall_status}
                  </span>
                </div>

                <div className="space-y-2 font-mono text-[10px]">
                  {diagnosticReport.steps.map((st) => (
                    <div key={st.step_number} className="p-2 rounded-lg bg-[#0b0f19] border border-slate-800/80 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="flex items-center gap-1.5 font-bold text-slate-200">
                          {st.status === 'pass' ? (
                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                          ) : st.status === 'warn' ? (
                            <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                          ) : (
                            <XCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                          )}
                          Step {st.step_number}: {st.name}
                        </span>
                        <span className="text-slate-500 text-[9px]">{st.latency_ms}ms</span>
                      </div>
                      <p className="text-slate-400 pl-5 text-[10px] font-sans">{st.message}</p>
                      {st.remedy && (
                        <p className="text-amber-300/90 pl-5 text-[10px] font-sans bg-amber-950/20 p-1 rounded mt-1">
                          Remedy: {st.remedy}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Connection Status Indicator */}
            <div className={`p-3 rounded-xl border text-xs flex items-start gap-2.5 ${
              providerConnected === true
                ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-300'
                : providerConnected === false
                ? 'bg-rose-950/20 border-rose-500/30 text-rose-300'
                : 'bg-slate-900 border-slate-800 text-slate-400'
            }`}>
              {providerConnected === true ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              ) : providerConnected === false ? (
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              ) : (
                <RefreshCw className="w-4 h-4 animate-spin shrink-0 mt-0.5" />
              )}
              <div className="space-y-0.5">
                <span className="font-semibold block">
                  {providerConnected === true ? 'Server Connected' : providerConnected === false ? 'Connection Error' : 'Testing Connection...'}
                </span>
                <p className="text-[11px] opacity-80">{providerMessage}</p>
              </div>
            </div>

            {/* Model Selector */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-[11px] font-mono text-slate-400">Available Model:</label>
                <button
                  onClick={() => checkHealthAndModels(selectedProvider)}
                  className="text-[10px] text-blue-400 hover:text-blue-300 font-mono flex items-center gap-1"
                >
                  <RefreshCw className="w-3 h-3" /> Refresh
                </button>
              </div>

              {models.length > 0 ? (
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-blue-500"
                >
                  {models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              ) : (
                <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800 text-xs text-slate-500 font-mono">
                  No local models discovered. Verify server is running locally.
                </div>
              )}
            </div>

            {/* System Prompt Customizer */}
            <div>
              <label className="text-[11px] font-mono text-slate-400 block mb-1">System Instruction:</label>
              <textarea
                rows={3}
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl p-2.5 text-xs text-slate-300 focus:outline-none focus:border-blue-500 resize-none font-mono"
              />
            </div>
          </div>
        </div>

        {/* Right Columns: Prompt Input, Response Stream & Response Reliability Inspector */}
        <div className="lg:col-span-2 space-y-5">
          {/* Chat Workspace */}
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
              <MessageSquare className="w-4 h-4 text-blue-400" />
              Prompt Console
            </h3>

            {/* Input area */}
            <div className="space-y-2">
              <textarea
                rows={3}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter prompt for local AI model..."
                className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl p-3 text-xs text-white focus:outline-none focus:border-blue-500 resize-none font-sans"
              />

              <div className="flex items-center justify-between">
                <span className="text-[11px] font-mono text-slate-500">
                  Model: {selectedModel || 'None selected'}
                </span>
                <button
                  disabled={loading || !selectedModel || !prompt.trim()}
                  onClick={handleSendPrompt}
                  className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-semibold text-xs tracking-wide flex items-center gap-2 transition shadow-sm"
                >
                  <Send className="w-3.5 h-3.5" />
                  {loading ? 'Generating...' : 'Send Prompt to Model'}
                </button>
              </div>
            </div>

            {/* Response Display */}
            {currentResponse && (
              <div className="space-y-4 pt-3 border-t border-[#1f293d]">
                <div className="flex items-center justify-between text-xs font-mono text-slate-400">
                  <span className="font-semibold text-white">Generated Response:</span>
                  <span>Latency: {currentResponse.duration_ms} ms</span>
                </div>

                <div className="bg-[#0b0f19] p-4 rounded-xl border border-slate-800 text-xs text-slate-200 leading-relaxed whitespace-pre-wrap font-sans">
                  {currentResponse.response}
                </div>

                {/* Response Reliability Inspector */}
                <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                      <ShieldAlert className="w-4 h-4 text-cyan-400" />
                      Response Reliability Inspector
                    </span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                      currentResponse.response_risk_score > 35
                        ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                        : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    }`}>
                      Output Risk: {currentResponse.response_risk_score.toFixed(0)}/100
                    </span>
                  </div>

                  {/* Warning Messages */}
                  {currentResponse.warnings.length > 0 ? (
                    <div className="space-y-1.5">
                      {currentResponse.warnings.map((w, idx) => (
                        <div key={idx} className="flex items-start gap-2 text-xs text-amber-300 bg-amber-950/20 p-2 rounded border border-amber-900/40">
                          <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                          <span>{w}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-emerald-400 flex items-center gap-1.5">
                      <CheckCircle2 className="w-4 h-4" />
                      No uncertainty flags or input hazards detected in this generation.
                    </p>
                  )}

                  {/* Human Feedback Controls */}
                  <div className="pt-2 border-t border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                    <span className="text-slate-400">Evaluate Output Quality:</span>
                    <div className="flex items-center gap-2 font-mono text-[11px]">
                      <button
                        onClick={() => submitFeedback('correct')}
                        className={`flex items-center gap-1 px-3 py-1 rounded-lg border transition ${
                          feedbackSubmitted === 'correct'
                            ? 'bg-emerald-600 text-white border-emerald-500'
                            : 'bg-slate-900 text-slate-300 border-slate-700 hover:bg-slate-800'
                        }`}
                      >
                        <ThumbsUp className="w-3 h-3" /> Correct
                      </button>
                      <button
                        onClick={() => submitFeedback('incorrect')}
                        className={`flex items-center gap-1 px-3 py-1 rounded-lg border transition ${
                          feedbackSubmitted === 'incorrect'
                            ? 'bg-rose-600 text-white border-rose-500'
                            : 'bg-slate-900 text-slate-300 border-slate-700 hover:bg-slate-800'
                        }`}
                      >
                        <ThumbsDown className="w-3 h-3" /> Incorrect
                      </button>
                      <button
                        onClick={() => submitFeedback('uncertain')}
                        className={`flex items-center gap-1 px-3 py-1 rounded-lg border transition ${
                          feedbackSubmitted === 'uncertain'
                            ? 'bg-amber-600 text-white border-amber-500'
                            : 'bg-slate-900 text-slate-300 border-slate-700 hover:bg-slate-800'
                        }`}
                      >
                        <HelpCircle className="w-3 h-3" /> Uncertain
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
