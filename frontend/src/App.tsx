import React, { useState, useEffect } from 'react';
import { Navigation } from './components/Navigation';
import { Header } from './components/Header';
import { OverviewPage } from './pages/OverviewPage';
import { ImageLabPage } from './pages/ImageLabPage';
import { AudioLabPage } from './pages/AudioLabPage';
import { LiveMonitorPage } from './pages/LiveMonitorPage';
import { LLMConsolePage } from './pages/LLMConsolePage';
import { BaselineManagerPage } from './pages/BaselineManagerPage';
import { DatasetsModelsPage } from './pages/DatasetsModelsPage';
import { EventHistoryPage } from './pages/EventHistoryPage';
import { SettingsPage } from './pages/SettingsPage';
import { SystemOverallStatus } from './types';
import { api } from './services/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [systemStatus, setSystemStatus] = useState<SystemOverallStatus | null>(null);
  const [activeProviderName, setActiveProviderName] = useState<string>('Ollama / Local');
  const [isBackendOnline, setIsBackendOnline] = useState<boolean>(true);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3500);

    // WebSocket real-time subscription
    let ws: WebSocket | null = null;
    let wsReconnectTimeout: any = null;

    const connectWebSocket = () => {
      try {
        ws = new WebSocket('ws://127.0.0.1:8000/ws/monitor');

        ws.onopen = () => {
          setIsBackendOnline(true);
          ws?.send(JSON.stringify({ type: 'get_status' }));
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'status_update' && data.overall) {
              setSystemStatus(data.overall);
              setIsBackendOnline(true);
              setLastUpdated(new Date());
            } else if ((data.type === 'image_update' || data.type === 'audio_update') && data.overall) {
              setSystemStatus(data.overall);
              setIsBackendOnline(true);
              setLastUpdated(new Date());
            }
          } catch (e) {
            // ignore malformed ws messages
          }
        };

        ws.onerror = () => {
          // let fetchStatus handle offline reporting
        };

        ws.onclose = () => {
          wsReconnectTimeout = setTimeout(connectWebSocket, 4000);
        };
      } catch (err) {
        // ws unsupported or blocked
      }
    };

    connectWebSocket();

    return () => {
      clearInterval(interval);
      if (wsReconnectTimeout) clearTimeout(wsReconnectTimeout);
      if (ws) ws.close();
    };
  }, []);

  const fetchStatus = async () => {
    try {
      const statusData = await api.getStatus();
      setSystemStatus(statusData);
      setIsBackendOnline(true);
      setLastUpdated(new Date());

      const provData = await api.getLLMProviders();
      const current = provData.providers?.find((p: any) => p.id === provData.active_provider);
      if (current) {
        setActiveProviderName(current.name.split(' ')[0]);
      }
    } catch (err) {
      // Backend disconnected / offline: DO NOT display falsely healthy status!
      setIsBackendOnline(false);
      setSystemStatus((prev) => ({
        overall_state: 'unavailable',
        overall_reliability_score: 0,
        image_state: 'unavailable',
        image_reliability_score: 0,
        audio_state: 'unavailable',
        audio_reliability_score: 0,
        llm_state: 'unavailable',
        active_warnings: prev?.active_warnings || [],
        recent_events_count: prev?.recent_events_count || 0,
        timestamp: Date.now() / 1000,
        last_updated_formatted: 'Backend Disconnected'
      }));
    }
  };

  const renderActivePage = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <OverviewPage
            status={systemStatus}
            isOnline={isBackendOnline}
            lastUpdated={lastUpdated}
            onNavigate={(tab) => setActiveTab(tab)}
            onRefresh={fetchStatus}
          />
        );
      case 'image-lab':
        return <ImageLabPage />;
      case 'audio-lab':
        return <AudioLabPage />;
      case 'live-monitor':
        return <LiveMonitorPage />;
      case 'llm-console':
        return (
          <LLMConsolePage
            currentInputRiskScore={
              systemStatus ? (100 - systemStatus.overall_reliability_score) : 0
            }
            currentInputSubsystem={
              systemStatus?.image_state === 'warning' ? 'image' : systemStatus?.audio_state === 'warning' ? 'audio' : 'sensor'
            }
          />
        );
      case 'baselines':
        return <BaselineManagerPage onBaselineChanged={fetchStatus} />;
      case 'datasets':
        return <DatasetsModelsPage />;
      case 'events':
        return <EventHistoryPage />;
      case 'settings':
        return <SettingsPage onSettingsSaved={fetchStatus} />;
      default:
        return (
          <OverviewPage
            status={systemStatus}
            isOnline={isBackendOnline}
            lastUpdated={lastUpdated}
            onNavigate={(tab) => setActiveTab(tab)}
            onRefresh={fetchStatus}
          />
        );
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0b0f19] text-slate-100 font-sans selection:bg-blue-600/30">
      {/* Sidebar Navigation */}
      <Navigation
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        overallState={isBackendOnline ? (systemStatus?.overall_state || 'normal') : 'unavailable'}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Top Header */}
        <Header
          status={systemStatus}
          activeProviderName={activeProviderName}
          isOnline={isBackendOnline}
        />

        {/* Scrollable Viewport */}
        <main className="flex-1 overflow-y-auto p-6 tech-grid">
          <div className="max-w-7xl mx-auto pb-12">
            {renderActivePage()}
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
