export interface MetricWarning {
  subsystem: string;
  metric: string;
  severity: 'caution' | 'warning' | 'normal';
  measured_value: number;
  expected_range: string;
  explanation: string;
  recommendation: string;
}

export interface ImageQualityMetrics {
  brightness: number;
  sharpness: number;
  contrast: number;
  color_mean_r: number;
  color_mean_g: number;
  color_mean_b: number;
  color_cast_delta: number;
  noise_estimate: number;
  underexposure_pct: number;
  overexposure_pct: number;
  width: number;
  height: number;
  resolution_mp: number;
}

export interface PredictionItem {
  class_id: number;
  class_name: string;
  probability: number;
}

export interface ModelPredictions {
  model_name: string;
  top_predictions: PredictionItem[];
  top_class: string;
  top_probability: number;
  entropy: number;
  margin: number;
  embedding_dimension: number;
  embedding_sample?: number[];
}

export interface ImageAnalysisResponse {
  metrics: ImageQualityMetrics;
  predictions: ModelPredictions;
  baseline_id: string | null;
  baseline_deviations: Record<string, number>;
  reliability_score: number;
  risk_score: number;
  status: 'normal' | 'caution' | 'warning';
  warnings: MetricWarning[];
  is_synthetically_degraded?: boolean;
  degradation_type?: string | null;
}

export interface AudioQualityMetrics {
  rms_dbfs: number;
  peak_amplitude: number;
  clipping_pct: number;
  background_noise_db: number;
  snr_db: number;
  speech_ratio: number;
  silence_ratio: number;
  spectral_centroid_hz: number;
  spectral_rolloff_hz: number;
  duration_sec: number;
  sample_rate: number;
}

export interface AudioClassificationResult {
  model_name: string;
  primary_condition: string;
  condition_probabilities: Record<string, number>;
}

export interface AudioAnalysisResponse {
  metrics: AudioQualityMetrics;
  classification: AudioClassificationResult;
  baseline_id: string | null;
  baseline_deviations: Record<string, number>;
  reliability_score: number;
  risk_score: number;
  status: 'normal' | 'caution' | 'warning';
  warnings: MetricWarning[];
  is_synthetically_degraded?: boolean;
  degradation_type?: string | null;
}

export interface SystemOverallStatus {
  overall_state: 'normal' | 'caution' | 'warning' | 'unavailable';
  overall_reliability_score: number;
  image_state: string;
  image_reliability_score: number;
  audio_state: string;
  audio_reliability_score: number;
  llm_state: string;
  active_warnings: MetricWarning[];
  recent_events_count: number;
  timestamp: number;
  last_updated_formatted?: string;
  monitoring_active?: boolean;
  active_image_baseline?: {
    id: string;
    name: string;
    sample_count: number;
    is_active: boolean;
  } | null;
  active_audio_baseline?: {
    id: string;
    name: string;
    sample_count: number;
    is_active: boolean;
  } | null;
  active_model?: string;
  active_provider?: string;
  last_image_metrics?: ImageQualityMetrics;
  last_audio_metrics?: AudioQualityMetrics;
}

export interface DiagnosticStepResult {
  step_number: number;
  name: string;
  status: 'pass' | 'fail' | 'warn';
  latency_ms: number;
  message: string;
  remedy?: string;
}

export interface LLMDiagnosticReport {
  provider: string;
  target_url: string;
  host: string;
  port: number;
  overall_status: 'connected' | 'warning' | 'failed';
  steps: DiagnosticStepResult[];
  discovered_models: string[];
  sample_response?: string;
  saved_as_active: boolean;
}

export interface LLMProviderInfo {
  id: string;
  name: string;
  type: string;
  base_url?: string;
}

export interface LLMResponseInfo {
  provider: string;
  model: string;
  prompt: string;
  response: string;
  duration_ms: number;
  input_risk_score: number;
  response_risk_score: number;
  warnings: string[];
  is_uncertain: boolean;
  is_empty: boolean;
  interaction_id: number;
}

export interface BaselineInfo {
  id: string;
  name: string;
  type: 'image' | 'audio';
  description: string;
  is_active: boolean;
  sample_count: number;
  stats: Record<string, any>;
  created_at: number;
  updated_at: number;
}

export interface DatasetInfo {
  id: string;
  name: string;
  type: 'image' | 'audio';
  description: string;
  classes: string[];
  sample_count: number;
  train_count: number;
  val_count: number;
  test_count: number;
  created_at: number;
}

export interface ModelVersionInfo {
  id: string;
  model_name: string;
  version: string;
  is_active: boolean;
  checkpoint_path: string | null;
  metrics: Record<string, any>;
  created_at: number;
}

export interface EventItem {
  id: number;
  timestamp: number;
  time_formatted: string;
  subsystem: string;
  severity: string;
  title: string;
  message: string;
  metric_name?: string;
  measured_value?: number;
  expected_range?: string;
  recommendation?: string;
  metadata: Record<string, any>;
}
