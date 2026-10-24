import React, { useState, useEffect } from 'react';
import {
  Database, Cpu, Play, RotateCcw, CheckCircle2,
  AlertCircle, Sparkles, Layers, HardDrive, RefreshCw
} from 'lucide-react';
import { DatasetInfo, ModelVersionInfo } from '../types';
import { api } from '../services/api';

export const DatasetsModelsPage: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [modelVersions, setModelVersions] = useState<ModelVersionInfo[]>([]);
  const [loading, setLoading] = useState(false);

  // Fine-tuning runner state
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [epochs, setEpochs] = useState<number>(3);
  const [learningRate, setLearningRate] = useState<number>(0.001);
  const [trainingJob, setTrainingJob] = useState<any>(null);

  useEffect(() => {
    loadAll();
  }, []);

  const loadAll = async () => {
    try {
      setLoading(true);
      const ds = await api.listDatasets();
      setDatasets(ds);
      if (ds.length > 0) setSelectedDatasetId(ds[0].id);

      const mv = await api.listModelVersions();
      setModelVersions(mv);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleStartTraining = async () => {
    if (!selectedDatasetId) return;
    try {
      setTrainingJob({ status: 'training', progress_pct: 10, current_epoch: 0, total_epochs: epochs });
      const res = await api.runModelTraining(selectedDatasetId, epochs, learningRate);
      setTrainingJob({ status: 'completed', progress_pct: 100, ...res });
      await loadAll();
    } catch (err: any) {
      setTrainingJob({ status: 'failed', error: err.message });
      alert('Training error: ' + err.message);
    }
  };

  const handleActivateModel = async (versionId: string) => {
    try {
      await api.activateModelVersion(versionId);
      await loadAll();
    } catch (err: any) {
      alert('Error activating model version: ' + err.message);
    }
  };

  const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1f293d]">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2.5">
            Local Datasets & MobileNetV3 Model Fine-Tuning
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-mono">
              Transfer Learning Pipeline
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Manage local inspection datasets, split train/validation sets, fine-tune MobileNetV3-Small on local CPU, and revert to known-good base weights.
          </p>
        </div>

        <button
          onClick={loadAll}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 font-medium transition"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} /> Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Local Dataset Registry */}
        <div className="space-y-5">
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                <Database className="w-4 h-4 text-cyan-400" />
                Local Dataset Registry ({datasets.length})
              </h3>
            </div>

            <div className="space-y-3">
              {datasets.map((d) => (
                <div
                  key={d.id}
                  onClick={() => setSelectedDatasetId(d.id)}
                  className={`p-4 rounded-xl border cursor-pointer transition ${
                    selectedDatasetId === d.id
                      ? 'bg-slate-800/80 border-cyan-500/50 shadow-sm'
                      : 'bg-[#0d1322] border-slate-800 hover:bg-slate-800/40'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <h4 className="text-xs font-semibold text-white">{d.name}</h4>
                    <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/40 px-2 py-0.5 rounded border border-cyan-800/40">
                      {d.sample_count} Samples
                    </span>
                  </div>

                  <p className="text-[11px] text-slate-400 mb-3">{d.description}</p>

                  {/* Classes tags */}
                  <div className="flex flex-wrap gap-1.5 mb-3">
                    {d.classes.map((c) => (
                      <span key={c} className="text-[10px] font-mono bg-slate-900 px-2 py-0.5 rounded text-slate-300 border border-slate-800">
                        {c}
                      </span>
                    ))}
                  </div>

                  {/* Train / Val / Test visual split */}
                  <div className="space-y-1 pt-2 border-t border-slate-800 text-[10px] font-mono">
                    <div className="flex justify-between text-slate-400">
                      <span>Train: {d.train_count} (70%)</span>
                      <span>Val: {d.val_count} (15%)</span>
                      <span>Test: {d.test_count} (15%)</span>
                    </div>
                    <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden flex">
                      <div className="bg-cyan-400 h-full" style={{ width: '70%' }} />
                      <div className="bg-amber-400 h-full" style={{ width: '15%' }} />
                      <div className="bg-emerald-400 h-full" style={{ width: '15%' }} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Local Fine-Tuning Execution Panel */}
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
              <Cpu className="w-4 h-4 text-emerald-400" />
              Transfer Learning Job Setup
            </h3>

            <div className="space-y-3 text-xs">
              <div>
                <label className="text-[11px] font-mono text-slate-400 block mb-1">Target Dataset:</label>
                <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800 text-white font-mono text-[11px]">
                  {selectedDataset?.name || 'No dataset selected'}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-[11px] font-mono text-slate-400 block mb-1">Epochs:</label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
                    className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-white font-mono text-xs focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="text-[11px] font-mono text-slate-400 block mb-1">Learning Rate:</label>
                  <input
                    type="number"
                    step="0.0005"
                    min="0.0001"
                    max="0.01"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.001)}
                    className="w-full bg-[#0d1322] border border-[#1f293d] rounded-xl px-3 py-2 text-white font-mono text-xs focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>

              <button
                disabled={!selectedDatasetId || trainingJob?.status === 'training'}
                onClick={handleStartTraining}
                className="w-full py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-semibold text-xs tracking-wide transition shadow-sm flex items-center justify-center gap-2 mt-2"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                {trainingJob?.status === 'training' ? 'Training MobileNetV3 on CPU...' : 'Launch Local Fine-Tuning'}
              </button>

              {/* Training Progress / Result */}
              {trainingJob && (
                <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2 mt-3 font-mono text-xs">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Job Status:</span>
                    <span className={`uppercase font-bold ${
                      trainingJob.status === 'completed' ? 'text-emerald-400' : trainingJob.status === 'training' ? 'text-cyan-400 animate-pulse' : 'text-rose-400'
                    }`}>
                      {trainingJob.status}
                    </span>
                  </div>

                  {trainingJob.val_accuracy !== undefined && (
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">Validation Accuracy:</span>
                      <span className="font-bold text-emerald-400">{(trainingJob.val_accuracy * 100).toFixed(1)}%</span>
                    </div>
                  )}

                  {trainingJob.version_tag && (
                    <div className="flex justify-between items-center text-[11px]">
                      <span className="text-slate-400">Checkpoint:</span>
                      <span className="text-slate-300">{trainingJob.version_tag}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Model Versions & Revert Controls */}
        <div className="space-y-5">
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-[10px] font-mono uppercase text-slate-400">Model Versioning</span>
                <h3 className="text-xs font-semibold text-white uppercase tracking-wider">
                  Checkpoint History & Safe Rollback
                </h3>
              </div>
              <span className="text-[11px] font-mono text-slate-400">
                {modelVersions.length} Registered Checkpoints
              </span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              If local training results in degraded performance or catastrophic forgetting, you can immediately revert back to the known-good base weights with one click.
            </p>

            <div className="space-y-3 pt-2">
              {modelVersions.map((m) => (
                <div
                  key={m.id}
                  className={`p-4 rounded-xl border transition ${
                    m.is_active
                      ? 'bg-slate-800/80 border-emerald-500/50 shadow-sm'
                      : 'bg-[#0d1322] border-slate-800'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div>
                      <h4 className="text-xs font-semibold text-white">{m.model_name}</h4>
                      <span className="text-[10px] font-mono text-slate-400">Tag: {m.version}</span>
                    </div>

                    {m.is_active ? (
                      <span className="px-2.5 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3" /> ACTIVE MODEL
                      </span>
                    ) : (
                      <button
                        onClick={() => handleActivateModel(m.id)}
                        className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-white transition flex items-center gap-1.5"
                      >
                        {m.id === 'base-pretrained' ? <RotateCcw className="w-3 h-3" /> : null}
                        {m.id === 'base-pretrained' ? 'Revert to Base' : 'Activate Version'}
                      </button>
                    )}
                  </div>

                  {m.metrics && (
                    <div className="flex items-center gap-4 text-[11px] font-mono text-slate-400 pt-2 border-t border-slate-800/60">
                      {m.metrics.val_accuracy !== undefined && (
                        <span>Accuracy: {(m.metrics.val_accuracy * 100).toFixed(1)}%</span>
                      )}
                      {m.metrics.epochs && (
                        <span>Epochs: {m.metrics.epochs}</span>
                      )}
                      {m.metrics.source && (
                        <span className="capitalize">{m.metrics.source}</span>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
