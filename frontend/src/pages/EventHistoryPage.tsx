import React, { useState, useEffect } from 'react';
import {
  History, Search, Filter, Download, Trash2, RefreshCw,
  AlertTriangle, ShieldCheck, Info, X, ExternalLink
} from 'lucide-react';
import { EventItem } from '../types';
import { api } from '../services/api';

export const EventHistoryPage: React.FC = () => {
  const [events, setEvents] = useState<EventItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [subsystemFilter, setSubsystemFilter] = useState('all');
  const [severityFilter, setSeverityFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedEvent, setSelectedEvent] = useState<EventItem | null>(null);

  useEffect(() => {
    loadEvents();
  }, [subsystemFilter, severityFilter]);

  const loadEvents = async () => {
    try {
      setLoading(true);
      const data = await api.listEvents(subsystemFilter, severityFilter, 100);
      setEvents(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    if (confirm('Clear all logged reliability events? This action cannot be undone.')) {
      try {
        await api.clearEvents();
        await loadEvents();
      } catch (err: any) {
        alert('Clear error: ' + err.message);
      }
    }
  };

  const exportJson = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(events, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `reliability_events_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const filteredEvents = events.filter((e) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      e.title.toLowerCase().includes(q) ||
      e.message.toLowerCase().includes(q) ||
      e.subsystem.toLowerCase().includes(q) ||
      (e.metric_name && e.metric_name.toLowerCase().includes(q))
    );
  });

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1f293d]">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2.5">
            Reliability Event History & Audit Log
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 font-mono">
              Audit Trail
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Search, filter, inspect and export all structured warnings, sustained deviations, and metric violations recorded locally.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={exportJson}
            disabled={events.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-xs text-slate-300 font-medium transition"
          >
            <Download className="w-3.5 h-3.5" /> Export JSON
          </button>
          <button
            onClick={handleClear}
            disabled={events.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-950/30 hover:bg-rose-900/50 border border-rose-800/40 disabled:opacity-50 text-xs text-rose-300 font-medium transition"
          >
            <Trash2 className="w-3.5 h-3.5" /> Clear History
          </button>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-4 shadow-sm flex flex-col sm:flex-row items-center gap-3">
        <div className="relative flex-1 w-full">
          <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search events by keyword, metric, or subsystem..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl pl-10 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          {/* Subsystem Filter */}
          <select
            value={subsystemFilter}
            onChange={(e) => setSubsystemFilter(e.target.value)}
            className="bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-xs text-slate-300 font-mono focus:outline-none"
          >
            <option value="all">Subsystem: All</option>
            <option value="image">Image</option>
            <option value="audio">Audio</option>
            <option value="llm">LLM</option>
            <option value="system">System</option>
          </select>

          {/* Severity Filter */}
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-xs text-slate-300 font-mono focus:outline-none"
          >
            <option value="all">Severity: All</option>
            <option value="warning">Warning</option>
            <option value="caution">Caution</option>
            <option value="normal">Normal</option>
          </select>

          <button
            onClick={loadEvents}
            className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Events Table */}
      <div className="bg-[#111827] border border-[#1f293d] rounded-2xl shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-[#1f293d] bg-slate-900/40 text-[11px] text-slate-400">
                <th className="py-3 px-4 font-medium">TIMESTAMP</th>
                <th className="py-3 px-4 font-medium">SUBSYSTEM</th>
                <th className="py-3 px-4 font-medium">SEVERITY</th>
                <th className="py-3 px-4 font-medium">EVENT TITLE</th>
                <th className="py-3 px-4 font-medium">MEASURED VALUE</th>
                <th className="py-3 px-4 font-medium">EXPECTED RANGE</th>
                <th className="py-3 px-4 font-medium text-right">ACTION</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {filteredEvents.length > 0 ? (
                filteredEvents.map((ev) => (
                  <tr
                    key={ev.id}
                    onClick={() => setSelectedEvent(ev)}
                    className="hover:bg-slate-800/40 transition cursor-pointer"
                  >
                    <td className="py-3 px-4 text-slate-400 text-[11px] whitespace-nowrap">
                      {ev.time_formatted}
                    </td>
                    <td className="py-3 px-4">
                      <span className="uppercase text-[10px] font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                        {ev.subsystem}
                      </span>
                    </td>
                    <td className="py-3 px-4">
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
                    <td className="py-3 px-4 font-sans font-medium text-white max-w-xs truncate">
                      {ev.title}
                    </td>
                    <td className="py-3 px-4 text-cyan-400 font-bold">
                      {ev.measured_value !== null && ev.measured_value !== undefined ? ev.measured_value.toFixed(1) : '-'}
                    </td>
                    <td className="py-3 px-4 text-slate-400">
                      {ev.expected_range || '-'}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={(e) => { e.stopPropagation(); setSelectedEvent(ev); }}
                        className="text-blue-400 hover:text-blue-300 text-[11px] underline"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="py-12 text-center text-slate-500 font-sans">
                    No matching events found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Event Details Inspection Modal */}
      {selectedEvent && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-4">
            <div className="flex items-start justify-between">
              <div>
                <span className="text-[10px] font-mono uppercase text-slate-400">Event Detail Inspector</span>
                <h3 className="text-base font-bold text-white mt-0.5">{selectedEvent.title}</h3>
                <span className="text-xs text-slate-400 font-mono">{selectedEvent.time_formatted}</span>
              </div>
              <button
                onClick={() => setSelectedEvent(null)}
                className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                <span className="text-[10px] font-mono text-slate-400 block mb-1">Plain-Language Explanation:</span>
                <p className="text-slate-200 leading-relaxed font-sans">{selectedEvent.message}</p>
              </div>

              {selectedEvent.recommendation && (
                <div className="p-3 rounded-xl bg-cyan-950/20 border border-cyan-800/40 text-cyan-200">
                  <span className="text-[10px] font-mono text-cyan-400 block mb-1 uppercase font-semibold">
                    Recommended Corrective Action:
                  </span>
                  <p className="leading-relaxed font-sans">{selectedEvent.recommendation}</p>
                </div>
              )}

              {/* Raw JSON Metadata */}
              <div>
                <span className="text-[10px] font-mono text-slate-400 block mb-1">Raw Event Metadata:</span>
                <pre className="p-3 rounded-xl bg-[#0b0f19] border border-slate-800 font-mono text-[11px] text-slate-300 overflow-x-auto max-h-40">
                  {JSON.stringify(selectedEvent.metadata, null, 2)}
                </pre>
              </div>
            </div>

            <div className="flex justify-end pt-2 border-t border-slate-800">
              <button
                onClick={() => setSelectedEvent(null)}
                className="px-4 py-2 rounded-xl bg-slate-800 text-white font-medium hover:bg-slate-700 transition text-xs"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
