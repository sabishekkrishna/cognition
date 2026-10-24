import React, { useState } from 'react';
import {
  Upload, Sliders, AlertTriangle, CheckCircle2, RotateCcw,
  Sparkles, Layers, Image as ImageIcon, Info, ArrowRight
} from 'lucide-react';
import { ImageAnalysisResponse } from '../types';
import { api } from '../services/api';
import { ReliabilityGauge } from '../components/ReliabilityGauge';
import { WarningCard } from '../components/WarningCard';

export const ImageLabPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<ImageAnalysisResponse | null>(null);

  // Synthetic degradation controls
  const [blurKernel, setBlurKernel] = useState<number>(0);
  const [brightnessFactor, setBrightnessFactor] = useState<number>(1.0);
  const [contrastFactor, setContrastFactor] = useState<number>(1.0);
  const [noiseSigma, setNoiseSigma] = useState<number>(0.0);
  const [colorTint, setColorTint] = useState<string>('none');
  const [tintIntensity, setTintIntensity] = useState<number>(0.3);

  // Comparison state
  const [degradeComparison, setDegradeComparison] = useState<{
    applied: string;
    degradedImg: string;
    originalImg: string;
    degradedAnalysis: any;
    originalAnalysis: any;
  } | null>(null);
  const [viewMode, setViewMode] = useState<'degraded' | 'original' | 'split'>('degraded');

  const handleFileUpload = async (file: File) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setDegradeComparison(null);
    resetSliders();

    try {
      setLoading(true);
      const res = await api.analyzeImage(file);
      setAnalysis(res);
    } catch (err: any) {
      alert(err.message || 'Error analyzing image');
    } finally {
      setLoading(false);
    }
  };

  const loadSampleImage = async () => {
    try {
      setLoading(true);
      const blob = await api.getSampleImageBlob();
      const file = new File([blob], 'sample_calibration_pattern.jpg', { type: 'image/jpeg' });
      await handleFileUpload(file);
    } catch (err: any) {
      alert('Error loading sample: ' + err.message);
      setLoading(false);
    }
  };

  const applySyntheticDegradation = async () => {
    if (!selectedFile) return;
    try {
      setLoading(true);
      const res = await api.degradeImage(selectedFile, {
        brightness_factor: brightnessFactor,
        gaussian_blur_kernel: blurKernel,
        gaussian_noise_sigma: noiseSigma,
        contrast_factor: contrastFactor,
        color_tint: colorTint !== 'none' ? colorTint : undefined,
        color_tint_intensity: tintIntensity
      });

      setDegradeComparison({
        applied: res.degradation_applied,
        degradedImg: res.degraded_image_base64,
        originalImg: res.original_image_base64,
        degradedAnalysis: res.degraded_analysis,
        originalAnalysis: res.original_analysis
      });

      // Update active analysis to degraded
      setAnalysis(res.degraded_analysis);
    } catch (err: any) {
      alert('Degradation error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetSliders = () => {
    setBlurKernel(0);
    setBrightnessFactor(1.0);
    setContrastFactor(1.0);
    setNoiseSigma(0.0);
    setColorTint('none');
    setTintIntensity(0.3);
  };

  const isDegraded = degradeComparison && degradeComparison.applied !== 'None (Clean Original)';

  return (
    <div className="space-y-6">
      {/* Title & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1f293d]">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2.5">
            Visual Perception & Image Lab
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-mono">
              OpenCV + MobileNetV3
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Evaluate image quality metrics, detect physical degradations, inspect MobileNetV3 predictions, and test synthetic degradation.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={loadSampleImage}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/40 text-xs text-blue-300 font-medium transition"
          >
            <Sparkles className="w-3.5 h-3.5" />
            Load Sample Pattern
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Image Preview & Synthetic Degradation Panel */}
        <div className="space-y-5">
          {/* Upload / Drop Area */}
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-4 shadow-sm">
            <label className="border-2 border-dashed border-slate-700 hover:border-blue-500/60 rounded-xl p-5 flex flex-col items-center justify-center cursor-pointer transition bg-slate-900/30 group">
              <Upload className="w-7 h-7 text-slate-400 group-hover:text-blue-400 mb-2 transition" />
              <span className="text-xs font-semibold text-slate-200">Click to upload or drag & drop</span>
              <span className="text-[11px] text-slate-500 mt-0.5">PNG, JPG, BMP, WebP</span>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileUpload(e.target.files[0]);
                  }
                }}
              />
            </label>

            {/* Image Preview & Toggle */}
            {previewUrl && (
              <div className="mt-4 space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-medium">Preview Canvas:</span>
                  {degradeComparison && (
                    <div className="flex bg-slate-900 rounded-lg p-0.5 border border-slate-800 text-[10px] font-mono">
                      <button
                        onClick={() => setViewMode('degraded')}
                        className={`px-2 py-0.5 rounded ${viewMode === 'degraded' ? 'bg-blue-600 text-white' : 'text-slate-400'}`}
                      >
                        Degraded
                      </button>
                      <button
                        onClick={() => setViewMode('original')}
                        className={`px-2 py-0.5 rounded ${viewMode === 'original' ? 'bg-blue-600 text-white' : 'text-slate-400'}`}
                      >
                        Clean Original
                      </button>
                    </div>
                  )}
                </div>

                <div className="relative aspect-video bg-[#0b0f19] rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
                  <img
                    src={
                      degradeComparison
                        ? viewMode === 'degraded'
                          ? degradeComparison.degradedImg
                          : degradeComparison.originalImg
                        : previewUrl
                    }
                    alt="Preview"
                    className="max-h-full max-w-full object-contain"
                  />

                  {isDegraded && (
                    <span className="absolute top-2 left-2 px-2 py-0.5 rounded bg-rose-500/90 text-white text-[10px] font-mono font-bold tracking-wider">
                      SYNTHETICALLY DEGRADED
                    </span>
                  )}
                </div>

                {isDegraded && (
                  <p className="text-[11px] font-mono text-amber-400 bg-amber-950/20 p-2 rounded border border-amber-900/40">
                    Active Effects: {degradeComparison.applied}
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Synthetic Degradation Controls */}
          <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                Degradation Test Panel
              </h3>
              <button
                onClick={resetSliders}
                className="text-[11px] text-slate-400 hover:text-slate-200 flex items-center gap-1 font-mono"
              >
                <RotateCcw className="w-3 h-3" /> Reset
              </button>
            </div>

            <div className="space-y-3.5 text-xs">
              {/* Blur */}
              <div>
                <div className="flex justify-between text-slate-400 font-mono mb-1">
                  <span>Gaussian Blur (Kernel):</span>
                  <span className="text-white">{blurKernel === 0 ? 'None' : `${blurKernel}x${blurKernel}`}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="35"
                  step="2"
                  value={blurKernel}
                  onChange={(e) => setBlurKernel(parseInt(e.target.value))}
                  className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded cursor-pointer"
                />
              </div>

              {/* Brightness Factor */}
              <div>
                <div className="flex justify-between text-slate-400 font-mono mb-1">
                  <span>Exposure / Brightness:</span>
                  <span className="text-white">{brightnessFactor.toFixed(2)}x</span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="2.5"
                  step="0.05"
                  value={brightnessFactor}
                  onChange={(e) => setBrightnessFactor(parseFloat(e.target.value))}
                  className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded cursor-pointer"
                />
              </div>

              {/* Contrast Factor */}
              <div>
                <div className="flex justify-between text-slate-400 font-mono mb-1">
                  <span>Contrast Ratio:</span>
                  <span className="text-white">{contrastFactor.toFixed(2)}x</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="2.0"
                  step="0.05"
                  value={contrastFactor}
                  onChange={(e) => setContrastFactor(parseFloat(e.target.value))}
                  className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded cursor-pointer"
                />
              </div>

              {/* Gaussian Noise */}
              <div>
                <div className="flex justify-between text-slate-400 font-mono mb-1">
                  <span>Sensor Noise (Sigma):</span>
                  <span className="text-white">{noiseSigma.toFixed(0)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="60"
                  step="5"
                  value={noiseSigma}
                  onChange={(e) => setNoiseSigma(parseFloat(e.target.value))}
                  className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded cursor-pointer"
                />
              </div>

              {/* Color Tint */}
              <div>
                <div className="flex justify-between text-slate-400 font-mono mb-1">
                  <span>Colour Tint Shift:</span>
                  <span className="text-white uppercase">{colorTint}</span>
                </div>
                <div className="grid grid-cols-5 gap-1 font-mono text-[11px]">
                  {['none', 'red', 'green', 'blue', 'yellow'].map((t) => (
                    <button
                      key={t}
                      onClick={() => setColorTint(t)}
                      className={`py-1 rounded border capitalize ${
                        colorTint === t
                          ? 'bg-blue-600/30 border-blue-400 text-white font-bold'
                          : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-800'
                      }`}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>

              <button
                disabled={!selectedFile || loading}
                onClick={applySyntheticDegradation}
                className="w-full py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-semibold text-xs tracking-wide transition shadow-sm mt-2"
              >
                {loading ? 'Evaluating Model & Metrics...' : 'Apply Degradation & Re-Analyze'}
              </button>
            </div>
          </div>
        </div>

        {/* Right Columns: Analysis, Gauges, Warnings & MobileNetV3 */}
        <div className="lg:col-span-2 space-y-5">
          {analysis ? (
            <>
              {/* Gauges & Status Card */}
              <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-[#1f293d] gap-4">
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                      Visual Assessment
                    </span>
                    <h3 className="text-base font-bold text-white flex items-center gap-2">
                      Visual Reliability Status:
                      <span className={`px-2 py-0.5 rounded text-xs font-mono font-bold uppercase ${
                        analysis.status === 'warning'
                          ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                          : analysis.status === 'caution'
                          ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                          : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      }`}>
                        {analysis.status}
                      </span>
                    </h3>
                  </div>

                  {/* Dual Score Gauges */}
                  <div className="flex items-center gap-4">
                    <ReliabilityGauge score={analysis.reliability_score} label="Reliability" size={105} />
                    <ReliabilityGauge score={analysis.risk_score} label="Perception Risk" size={105} />
                  </div>
                </div>

                {/* Warnings Section */}
                <div className="mt-4">
                  <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2.5">
                    Metric Warnings & Downstream Impact ({analysis.warnings.length})
                  </h4>
                  {analysis.warnings.length > 0 ? (
                    <div className="space-y-2.5">
                      {analysis.warnings.map((w, idx) => (
                        <WarningCard key={idx} warning={w} />
                      ))}
                    </div>
                  ) : (
                    <div className="p-3.5 rounded-xl bg-emerald-950/20 border border-emerald-500/30 text-emerald-300 text-xs flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                      <span>All OpenCV metrics match calibrated baseline bounds. Visual model features are reliable.</span>
                    </div>
                  )}
                </div>
              </div>

              {/* OpenCV Measurements Grid */}
              <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
                <h3 className="text-xs font-semibold text-white uppercase tracking-wider">
                  OpenCV Quantitative Image Metrics
                </h3>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {/* Sharpness */}
                  <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                    <span className="text-[10px] font-mono uppercase text-slate-400">Sharpness (Laplacian)</span>
                    <div className="text-base font-bold font-mono text-white mt-0.5">
                      {analysis.metrics.sharpness.toFixed(1)}
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">
                      Baseline Dev: {analysis.baseline_deviations?.sharpness?.toFixed(1) ?? '0.0'}σ
                    </span>
                  </div>

                  {/* Brightness */}
                  <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                    <span className="text-[10px] font-mono uppercase text-slate-400">Brightness (Luminance)</span>
                    <div className="text-base font-bold font-mono text-white mt-0.5">
                      {analysis.metrics.brightness.toFixed(1)} / 255
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">
                      Baseline Dev: {analysis.baseline_deviations?.brightness?.toFixed(1) ?? '0.0'}σ
                    </span>
                  </div>

                  {/* Contrast */}
                  <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                    <span className="text-[10px] font-mono uppercase text-slate-400">Contrast (StdDev)</span>
                    <div className="text-base font-bold font-mono text-white mt-0.5">
                      {analysis.metrics.contrast.toFixed(1)}
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">
                      Baseline Dev: {analysis.baseline_deviations?.contrast?.toFixed(1) ?? '0.0'}σ
                    </span>
                  </div>

                  {/* Noise */}
                  <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                    <span className="text-[10px] font-mono uppercase text-slate-400">Noise Sigma (Immerkaer)</span>
                    <div className="text-base font-bold font-mono text-white mt-0.5">
                      {analysis.metrics.noise_estimate.toFixed(2)}
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">
                      Expected: ≤ 12.0
                    </span>
                  </div>

                  {/* Colour Cast */}
                  <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                    <span className="text-[10px] font-mono uppercase text-slate-400">Colour Cast Delta</span>
                    <div className="text-base font-bold font-mono text-white mt-0.5">
                      {analysis.metrics.color_cast_delta.toFixed(1)}
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">
                      RGB: {analysis.metrics.color_mean_r.toFixed(0)}/{analysis.metrics.color_mean_g.toFixed(0)}/{analysis.metrics.color_mean_b.toFixed(0)}
                    </span>
                  </div>

                  {/* Clipping */}
                  <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                    <span className="text-[10px] font-mono uppercase text-slate-400">Exposure Clipping</span>
                    <div className="text-base font-bold font-mono text-white mt-0.5">
                      Black {analysis.metrics.underexposure_pct}% | Sat {analysis.metrics.overexposure_pct}%
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">
                      {analysis.metrics.width}x{analysis.metrics.height} ({analysis.metrics.resolution_mp} MP)
                    </span>
                  </div>
                </div>
              </div>

              {/* MobileNetV3 Classifier Card */}
              <div className="bg-[#111827] border border-[#1f293d] rounded-2xl p-5 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-[10px] font-mono uppercase text-slate-400">Vision Model</span>
                    <h3 className="text-xs font-semibold text-white uppercase tracking-wider">
                      MobileNetV3-Small Classification & Feature Entropy
                    </h3>
                  </div>
                  <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/30 px-2 py-0.5 rounded border border-cyan-800/40">
                    {analysis.predictions.model_name}
                  </span>
                </div>

                {/* Top-5 Predictions */}
                <div className="space-y-2">
                  {analysis.predictions.top_predictions.map((p, idx) => (
                    <div key={idx} className="space-y-1">
                      <div className="flex justify-between text-xs font-mono">
                        <span className="text-slate-300 capitalize">
                          {idx + 1}. {p.class_name}
                        </span>
                        <span className="text-white font-semibold">
                          {(p.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${idx === 0 ? 'bg-cyan-400' : 'bg-slate-600'}`}
                          style={{ width: `${p.probability * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Model Confidence Statistics */}
                <div className="grid grid-cols-3 gap-3 pt-3 border-t border-slate-800 text-xs font-mono">
                  <div className="bg-slate-900/40 p-2.5 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-400">Prediction Entropy</span>
                    <div className="text-sm font-bold text-white mt-0.5">{analysis.predictions.entropy.toFixed(2)}</div>
                  </div>
                  <div className="bg-slate-900/40 p-2.5 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-400">Top-2 Margin</span>
                    <div className="text-sm font-bold text-white mt-0.5">{(analysis.predictions.margin * 100).toFixed(1)}%</div>
                  </div>
                  <div className="bg-slate-900/40 p-2.5 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-400">Feature Embedding</span>
                    <div className="text-sm font-bold text-white mt-0.5">{analysis.predictions.embedding_dimension}-dim</div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="h-96 rounded-2xl bg-[#111827] border border-[#1f293d] p-8 flex flex-col items-center justify-center text-center">
              <ImageIcon className="w-12 h-12 text-slate-600 mb-3" />
              <h3 className="text-sm font-semibold text-slate-200">No Image Loaded</h3>
              <p className="text-xs text-slate-500 max-w-sm mt-1">
                Upload a camera frame or click "Load Sample Pattern" above to start visual reliability diagnostics.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
