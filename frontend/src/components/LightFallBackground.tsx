import React, { useEffect, useRef } from 'react';

/**
 * LightFallBackground renders falling light streaks along a curved 3D tunnel perspective
 * with glowing heads and smooth fading tails, matching the reference screenshot.
 */
interface Streak {
  startX: number;
  y: number;
  speed: number;
  length: number;
  size: number;
  alpha: number;
  layer: number; // 0: faint background, 1: mid, 2: bright foreground
}

export const LightFallBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    let width = 0;
    let height = 0;

    const streaks: Streak[] = [];
    const STREAK_COUNT = 95;

    // Mouse tracking for subtle interactive lens flare
    const mouse = { x: -1000, y: -1000, targetX: -1000, targetY: -1000, isDown: false };

    const getCurveX = (startX: number, yPos: number, w: number, h: number) => {
      const centerX = w * 0.5;
      const progress = Math.max(0, yPos / (h || 1));
      // Parabolic flare outwards from center towards bottom
      const displacement = (startX - centerX) * Math.pow(progress, 1.85) * 0.44;
      return startX + displacement;
    };

    const createStreak = (startY?: number): Streak => {
      // Concentrate slightly more streaks towards center and mid-edges
      const startX = Math.random() * width;
      const layer = Math.random() < 0.28 ? 2 : Math.random() < 0.68 ? 1 : 0;
      const speed =
        layer === 2
          ? 2.8 + Math.random() * 2.2
          : layer === 1
          ? 1.9 + Math.random() * 1.5
          : 1.1 + Math.random() * 1.1;
      const length =
        layer === 2
          ? 150 + Math.random() * 110
          : layer === 1
          ? 100 + Math.random() * 80
          : 60 + Math.random() * 55;
      const size =
        layer === 2
          ? 2.0 + Math.random() * 0.9
          : layer === 1
          ? 1.4 + Math.random() * 0.6
          : 0.9 + Math.random() * 0.4;
      const alpha =
        layer === 2
          ? 0.9 + Math.random() * 0.1
          : layer === 1
          ? 0.65 + Math.random() * 0.2
          : 0.35 + Math.random() * 0.2;

      return {
        startX,
        y: startY !== undefined ? startY : Math.random() * (height + length) - length,
        speed,
        length,
        size,
        alpha,
        layer,
      };
    };

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);

      streaks.length = 0;
      for (let i = 0; i < STREAK_COUNT; i++) {
        streaks.push(createStreak());
      }
    };

    resize();
    window.addEventListener('resize', resize);

    const onPointerMove = (e: PointerEvent) => {
      mouse.targetX = e.clientX;
      mouse.targetY = e.clientY;
    };
    const onPointerDown = () => {
      mouse.isDown = true;
    };
    const onPointerUp = () => {
      mouse.isDown = false;
    };
    const onPointerLeave = () => {
      mouse.targetX = -1000;
      mouse.targetY = -1000;
      mouse.isDown = false;
    };

    window.addEventListener('pointermove', onPointerMove, { passive: true });
    window.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointerup', onPointerUp);
    document.addEventListener('pointerleave', onPointerLeave);

    const render = () => {
      // Smooth mouse interpolation
      mouse.x += (mouse.targetX - mouse.x) * 0.1;
      mouse.y += (mouse.targetY - mouse.y) * 0.1;

      const speedMultiplier = mouse.isDown ? 2.8 : 1.0;

      // Dark background with subtle deep navy vignette
      ctx.fillStyle = '#06060a';
      ctx.fillRect(0, 0, width, height);

      // Atmospheric top-center soft illumination
      const topVignette = ctx.createRadialGradient(
        width * 0.5,
        0,
        20,
        width * 0.5,
        height * 0.1,
        width * 0.6
      );
      topVignette.addColorStop(0, 'rgba(40, 38, 62, 0.55)');
      topVignette.addColorStop(0.5, 'rgba(20, 18, 35, 0.25)');
      topVignette.addColorStop(1, 'rgba(6, 6, 10, 0)');
      ctx.fillStyle = topVignette;
      ctx.fillRect(0, 0, width, height * 0.6);

      // Render falling streaks
      for (let i = 0; i < streaks.length; i++) {
        const s = streaks[i];

        // Move streak downward
        s.y += s.speed * speedMultiplier;

        // Reset if past bottom
        if (s.y - s.length > height + 20) {
          streaks[i] = createStreak(-s.length - Math.random() * 40);
          continue;
        }

        const headY = s.y;
        const tailY = Math.max(-20, headY - s.length);
        const midY = (headY + tailY) * 0.5;

        const headX = getCurveX(s.startX, headY, width, height);
        const tailX = getCurveX(s.startX, tailY, width, height);
        const midX = getCurveX(s.startX, midY, width, height);

        // Distance to cursor for subtle lens interaction
        const distToMouse = Math.hypot(headX - mouse.x, headY - mouse.y);
        const mouseFactor = distToMouse < 180 ? (1 - distToMouse / 180) * 0.4 : 0;
        const effectiveAlpha = Math.min(1.0, s.alpha + mouseFactor);

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(tailX, tailY);
        ctx.quadraticCurveTo(midX, midY, headX, headY);

        // Gradient for fading tail
        const grad = ctx.createLinearGradient(tailX, tailY, headX, headY);
        grad.addColorStop(0, 'rgba(255, 255, 255, 0)');
        grad.addColorStop(0.65, `rgba(225, 235, 255, ${effectiveAlpha * 0.35})`);
        grad.addColorStop(0.88, `rgba(250, 252, 255, ${effectiveAlpha * 0.85})`);
        grad.addColorStop(1, `rgba(255, 255, 255, ${effectiveAlpha})`);

        ctx.strokeStyle = grad;
        ctx.lineWidth = s.size;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Glowing circular head node at the tip
        ctx.beginPath();
        ctx.arc(headX, headY, s.size * 1.35, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${effectiveAlpha})`;
        ctx.shadowColor = 'rgba(245, 250, 255, 0.98)';
        ctx.shadowBlur = s.layer === 2 ? 11 : s.layer === 1 ? 7 : 4;
        ctx.fill();

        // Outer bloom halo for foreground streaks
        if (s.layer === 2) {
          ctx.beginPath();
          ctx.arc(headX, headY, s.size * 3.2, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(220, 235, 255, ${effectiveAlpha * 0.32})`;
          ctx.fill();
        }

        ctx.restore();
      }

      animId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointerup', onPointerUp);
      document.removeEventListener('pointerleave', onPointerLeave);
      cancelAnimationFrame(animId);
    };
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden select-none" aria-hidden="true">
      <canvas ref={canvasRef} className="w-full h-full block" />
    </div>
  );
};

export default LightFallBackground;
