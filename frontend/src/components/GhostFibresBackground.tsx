import React, { useEffect, useRef } from 'react';

/**
 * GhostFibresBackground renders an ethereal purple/violet particulate wave mesh
 * (ghost fibres) flowing across the dark canvas, matching reference image 2.
 */
interface BokehDot {
  x: number;
  y: number;
  r: number;
  alpha: number;
  speed: number;
}

export const GhostFibresBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    let width = 0;
    let height = 0;

    // Background bokeh particles
    const bokeh: BokehDot[] = [];
    const BOKEH_COUNT = 45;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);

      bokeh.length = 0;
      for (let i = 0; i < BOKEH_COUNT; i++) {
        bokeh.push({
          x: Math.random() * width,
          y: Math.random() * height,
          r: 2.5 + Math.random() * 5.5,
          alpha: 0.06 + Math.random() * 0.14,
          speed: 0.1 + Math.random() * 0.25,
        });
      }
    };

    resize();
    window.addEventListener('resize', resize);

    let t = 0;

    const render = () => {
      t += 1;

      // Dark moody background
      ctx.fillStyle = '#06050b';
      ctx.fillRect(0, 0, width, height);

      // Deep purple ambient glow in center-right
      const ambient = ctx.createRadialGradient(
        width * 0.65,
        height * 0.45,
        50,
        width * 0.65,
        height * 0.45,
        width * 0.75
      );
      ambient.addColorStop(0, 'rgba(88, 48, 150, 0.22)');
      ambient.addColorStop(0.5, 'rgba(45, 22, 85, 0.12)');
      ambient.addColorStop(1, 'rgba(6, 5, 11, 0)');
      ctx.fillStyle = ambient;
      ctx.fillRect(0, 0, width, height);

      // Render drifting soft bokeh circles
      for (const b of bokeh) {
        b.y -= b.speed;
        if (b.y < -10) {
          b.y = height + 10;
          b.x = Math.random() * width;
        }

        ctx.beginPath();
        ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(168, 110, 245, ${b.alpha})`;
        ctx.fill();
      }

      // Ghost fibres: multiple layers of particulate ribbons
      const ribbons = [
        // Main diagonal ribbon wave
        {
          lines: 26,
          baseYOffset: height * 0.38,
          slope: 0.15,
          waveAmp1: 75,
          waveFreq1: 0.0022,
          speed1: 0.008,
          waveAmp2: 38,
          waveFreq2: 0.0048,
          speed2: -0.006,
          colorCore: 'rgba(235, 215, 255, 0.85)',
          colorMid: 'rgba(175, 120, 250, 0.6)',
          colorOuter: 'rgba(120, 70, 200, 0.2)',
          stepX: 18,
          pointSize: 1.6,
        },
        // Secondary floating ribbon (lower right)
        {
          lines: 18,
          baseYOffset: height * 0.68,
          slope: -0.22,
          waveAmp1: 50,
          waveFreq1: 0.0018,
          speed1: 0.005,
          waveAmp2: 25,
          waveFreq2: 0.0035,
          speed2: 0.007,
          colorCore: 'rgba(215, 185, 255, 0.6)',
          colorMid: 'rgba(150, 95, 235, 0.4)',
          colorOuter: 'rgba(95, 50, 175, 0.15)',
          stepX: 24,
          pointSize: 1.3,
        },
        // Subtle tertiary upper ribbon
        {
          lines: 12,
          baseYOffset: height * 0.18,
          slope: 0.08,
          waveAmp1: 35,
          waveFreq1: 0.0026,
          speed1: -0.007,
          waveAmp2: 18,
          waveFreq2: 0.0052,
          speed2: 0.009,
          colorCore: 'rgba(200, 160, 255, 0.45)',
          colorMid: 'rgba(135, 80, 215, 0.25)',
          colorOuter: 'rgba(80, 40, 150, 0.08)',
          stepX: 26,
          pointSize: 1.1,
        },
      ];

      for (const r of ribbons) {
        for (let l = 0; l < r.lines; l++) {
          const lineFraction = l / (r.lines - 1); // 0 to 1
          const distFromCenter = Math.abs(lineFraction - 0.5) * 2; // 0 at core, 1 at edges

          // Line offset in ribbon space
          const ribbonOffset = (l - r.lines / 2) * 9;

          // Color calculation based on closeness to core
          const isCore = distFromCenter < 0.2;
          const isMid = distFromCenter < 0.65;
          const alpha = isCore
            ? 0.75 - distFromCenter * 0.4
            : isMid
            ? 0.45 - distFromCenter * 0.25
            : 0.22 - distFromCenter * 0.15;

          const ptSize = isCore ? r.pointSize * 1.35 : isMid ? r.pointSize : r.pointSize * 0.8;

          for (let x = 0; x <= width + 40; x += r.stepX) {
            const timeVal = t;
            // Compound undulating harmonic wave
            const w1 = Math.sin(x * r.waveFreq1 + timeVal * r.speed1 + l * 0.07) * r.waveAmp1;
            const w2 = Math.cos(x * r.waveFreq2 + timeVal * r.speed2 + l * 0.04) * r.waveAmp2;
            const y = r.baseYOffset + (x - width / 2) * r.slope + ribbonOffset + w1 + w2;

            // Draw glowing fiber point
            ctx.beginPath();
            ctx.arc(x, y, ptSize, 0, Math.PI * 2);

            if (isCore) {
              ctx.fillStyle = `rgba(240, 225, 255, ${alpha})`;
            } else if (isMid) {
              ctx.fillStyle = `rgba(180, 125, 255, ${alpha})`;
            } else {
              ctx.fillStyle = `rgba(130, 75, 215, ${alpha})`;
            }
            ctx.fill();
          }
        }
      }

      animId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animId);
    };
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden select-none" aria-hidden="true">
      <canvas ref={canvasRef} className="w-full h-full block" />
    </div>
  );
};
