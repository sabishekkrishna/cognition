import React, { useRef, useEffect } from 'react';

interface WaveformVisualizerProps {
  audioBlob?: Blob | null;
  height?: number;
  barColor?: string;
}

export const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
  audioBlob,
  height = 90,
  barColor = '#06b6d4'
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    ctx.clearRect(0, 0, width, height);

    if (!audioBlob) {
      // Draw quiet baseline waveform placeholder
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      return;
    }

    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const reader = new FileReader();

    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result as ArrayBuffer;
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const rawData = audioBuffer.getChannelData(0); // Left channel

        ctx.clearRect(0, 0, width, height);
        const step = Math.ceil(rawData.length / width);
        const amp = height / 2;

        ctx.fillStyle = barColor;
        for (let i = 0; i < width; i++) {
          let min = 1.0;
          let max = -1.0;
          for (let j = 0; j < step; j++) {
            const datum = rawData[i * step + j];
            if (datum < min) min = datum;
            if (datum > max) max = datum;
          }
          const y1 = (1 + min) * amp;
          const y2 = Math.max(1, (max - min) * amp);
          ctx.fillRect(i, y1, 1, y2);
        }
      } catch (err) {
        console.error('Waveform render error:', err);
      } finally {
        audioContext.close();
      }
    };

    reader.readAsArrayBuffer(audioBlob);
  }, [audioBlob, height, barColor]);

  return (
    <div className="w-full bg-[#0d1322] p-2 rounded-xl border border-[#1f293d] overflow-hidden">
      <canvas
        ref={canvasRef}
        width={700}
        height={height}
        className="w-full h-auto block rounded"
      />
    </div>
  );
};
