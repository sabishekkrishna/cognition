import {
  ImageAnalysisResponse, AudioAnalysisResponse,
  SystemOverallStatus, LLMResponseInfo, BaselineInfo,
  DatasetInfo, ModelVersionInfo, EventItem
} from '../types';

const API_BASE = 'http://127.0.0.1:8000/api';

export const api = {
  // Health & Status
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Failed to fetch health');
    return res.json();
  },

  async getStatus(): Promise<SystemOverallStatus> {
    const res = await fetch(`${API_BASE}/status`);
    if (!res.ok) throw new Error('Failed to fetch status');
    return res.json();
  },

  // Image Lab
  async analyzeImage(file: File): Promise<ImageAnalysisResponse> {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${API_BASE}/image/analyze`, {
      method: 'POST',
      body: form
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to analyze image');
    }
    return res.json();
  },

  async degradeImage(file: File, params: {
    brightness_factor: number;
    gaussian_blur_kernel: number;
    gaussian_noise_sigma: number;
    contrast_factor: number;
    color_tint?: string;
    color_tint_intensity?: number;
  }) {
    const form = new FormData();
    form.append('file', file);
    form.append('brightness_factor', params.brightness_factor.toString());
    form.append('gaussian_blur_kernel', params.gaussian_blur_kernel.toString());
    form.append('gaussian_noise_sigma', params.gaussian_noise_sigma.toString());
    form.append('contrast_factor', params.contrast_factor.toString());
    if (params.color_tint) {
      form.append('color_tint', params.color_tint);
      form.append('color_tint_intensity', (params.color_tint_intensity ?? 0.3).toString());
    }

    const res = await fetch(`${API_BASE}/image/degrade`, {
      method: 'POST',
      body: form
    });
    if (!res.ok) throw new Error('Failed to apply degradation');
    return res.json();
  },

  async getSampleImageBlob(): Promise<Blob> {
    const res = await fetch(`${API_BASE}/image/sample`);
    if (!res.ok) throw new Error('Failed to fetch sample image');
    return res.blob();
  },

  // Audio Lab
  async analyzeAudio(file: Blob | File): Promise<AudioAnalysisResponse> {
    const form = new FormData();
    form.append('file', file, 'audio.wav');
    const res = await fetch(`${API_BASE}/audio/analyze`, {
      method: 'POST',
      body: form
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to analyze audio');
    }
    return res.json();
  },

  async degradeAudio(file: Blob | File, params: {
    volume_factor: number;
    noise_type?: string;
    noise_level: number;
    clipping_gain: number;
  }) {
    const form = new FormData();
    form.append('file', file, 'audio.wav');
    form.append('volume_factor', params.volume_factor.toString());
    if (params.noise_type) {
      form.append('noise_type', params.noise_type);
      form.append('noise_level', params.noise_level.toString());
    }
    form.append('clipping_gain', params.clipping_gain.toString());

    const res = await fetch(`${API_BASE}/audio/degrade`, {
      method: 'POST',
      body: form
    });
    if (!res.ok) throw new Error('Failed to apply audio degradation');
    return res.json();
  },

  async getSampleAudioBlob(): Promise<Blob> {
    const res = await fetch(`${API_BASE}/audio/sample`);
    if (!res.ok) throw new Error('Failed to fetch sample audio');
    return res.blob();
  },

  // Local LLM
  async getLLMProviders() {
    const res = await fetch(`${API_BASE}/llm/providers`);
    return res.json();
  },

  async checkLLMHealth(provider?: string) {
    const url = provider ? `${API_BASE}/llm/health?provider=${provider}` : `${API_BASE}/llm/health`;
    const res = await fetch(url);
    return res.json();
  },

  async getLLMModels(provider?: string) {
    const url = provider ? `${API_BASE}/llm/models?provider=${provider}` : `${API_BASE}/llm/models`;
    const res = await fetch(url);
    return res.json();
  },

  async generateLLM(req: {
    provider: string;
    model: string;
    prompt: string;
    system_prompt?: string;
    temperature?: number;
    max_tokens?: number;
    current_input_risk_score?: number;
    current_input_subsystem?: string;
  }): Promise<LLMResponseInfo> {
    const res = await fetch(`${API_BASE}/llm/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to generate LLM response');
    }
    return res.json();
  },

  async submitLLMFeedback(interactionId: number, rating: string, notes?: string) {
    const res = await fetch(`${API_BASE}/llm/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        interaction_id: interactionId,
        user_rating: rating,
        user_notes: notes
      })
    });
    return res.json();
  },

  // Baselines
  async listBaselines(type?: string): Promise<BaselineInfo[]> {
    const url = type ? `${API_BASE}/baselines?type=${type}` : `${API_BASE}/baselines`;
    const res = await fetch(url);
    return res.json();
  },

  async getActiveBaseline(type: 'image' | 'audio') {
    const res = await fetch(`${API_BASE}/baselines/active?type=${type}`);
    return res.json();
  },

  async activateBaseline(id: string) {
    const res = await fetch(`${API_BASE}/baselines/activate/${id}`, { method: 'POST' });
    return res.json();
  },

  async deleteBaseline(id: string) {
    const res = await fetch(`${API_BASE}/baselines/${id}?confirm=true`, { method: 'DELETE' });
    return res.json();
  },

  async createBaseline(name: string, type: 'image' | 'audio', description: string, files: File[]) {
    const form = new FormData();
    form.append('name', name);
    form.append('description', description);
    for (const f of files) {
      form.append('files', f);
    }
    const endpoint = type === 'image' ? `${API_BASE}/baselines/create/image` : `${API_BASE}/baselines/create/audio`;
    const res = await fetch(endpoint, { method: 'POST', body: form });
    if (!res.ok) throw new Error('Failed to create baseline profile');
    return res.json();
  },

  // Datasets & Training
  async listDatasets(): Promise<DatasetInfo[]> {
    const res = await fetch(`${API_BASE}/datasets`);
    return res.json();
  },

  async createDemoDataset() {
    const res = await fetch(`${API_BASE}/datasets/demo`, { method: 'POST' });
    return res.json();
  },

  async listModelVersions(): Promise<ModelVersionInfo[]> {
    const res = await fetch(`${API_BASE}/models/versions`);
    return res.json();
  },

  async activateModelVersion(id: string) {
    const res = await fetch(`${API_BASE}/models/activate/${id}`, { method: 'POST' });
    return res.json();
  },

  async runModelTraining(datasetId: string, epochs: number, lr: number) {
    const form = new FormData();
    form.append('dataset_id', datasetId);
    form.append('epochs', epochs.toString());
    form.append('lr', lr.toString());
    const res = await fetch(`${API_BASE}/models/train`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Training failed');
    }
    return res.json();
  },

  async getTrainingStatus() {
    const res = await fetch(`${API_BASE}/models/train/status`);
    return res.json();
  },

  // Events & Settings
  async listEvents(subsystem: string = 'all', severity: string = 'all', limit: number = 50): Promise<EventItem[]> {
    const res = await fetch(`${API_BASE}/events?subsystem=${subsystem}&severity=${severity}&limit=${limit}`);
    return res.json();
  },

  async clearEvents() {
    const res = await fetch(`${API_BASE}/events?confirm=true`, { method: 'DELETE' });
    return res.json();
  },

  async getSettings(): Promise<Record<string, string>> {
    const res = await fetch(`${API_BASE}/settings`);
    return res.json();
  },

  async updateSettings(settings: Record<string, string>) {
    const res = await fetch(`${API_BASE}/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    });
    return res.json();
  }
};
