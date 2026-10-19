'use client';

import React, { useEffect, useRef, useState, useMemo } from 'react';

export interface DappledLightProps {
  shadowType?: 'king' | 'leaves';
  lightColor?: string;
  ambientColor?: string;
  swaySpeed?: number;
  swayIntensity?: number;
  blur?: number;
  windowFrame?: boolean;
  filmGrain?: boolean;
  dustParticles?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

export const DappledLight: React.FC<DappledLightProps> = ({
  shadowType = 'king',
  lightColor = '#9a87f5',
  ambientColor = '#07070a',
  swaySpeed = 1,
  swayIntensity = 1,
  blur = 2.2,
  windowFrame = true,
  filmGrain = true,
  dustParticles = true,
  className = '',
  style,
}) => {
  const [time, setTime] = useState(0);
  const animRef = useRef<number | null>(null);

  useEffect(() => {
    let start = performance.now();
    const loop = (now: number) => {
      const elapsed = (now - start) / 1000;
      setTime(elapsed * swaySpeed);
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [swaySpeed]);

  // Sway physics calculation: natural dual-harmonic wave
  const swayX = Math.sin(time * 0.9) * 8 * swayIntensity + Math.cos(time * 1.7) * 4 * swayIntensity;
  const swayY = Math.cos(time * 0.8) * 6 * swayIntensity + Math.sin(time * 1.5) * 3 * swayIntensity;
  const swayRot = Math.sin(time * 0.7) * 2.2 * swayIntensity;

  // Floating sun motes (dust particles) dancing in the sunbeam
  const motes = useMemo(() => {
    return Array.from({ length: 24 }, (_, i) => ({
      id: i,
      baseX: 750 + ((i * 37) % 650),
      baseY: 150 + ((i * 43) % 550),
      r: 1 + (i % 3) * 0.7,
      speedX: 0.3 + (i % 5) * 0.15,
      speedY: 0.2 + (i % 4) * 0.1,
      phase: i * 1.3,
    }));
  }, []);

  return (
    <div
      className={`fixed inset-0 pointer-events-none z-0 overflow-hidden select-none ${className}`}
      style={{ backgroundColor: ambientColor, ...style }}
      aria-hidden="true"
    >
      <svg
        className="w-full h-full object-cover"
        viewBox="0 0 1600 1000"
        preserveAspectRatio="xMidYMid slice"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          {/* Ambient Wall Light Gradient */}
          <radialGradient id="dappledAmbient" cx="72%" cy="38%" r="65%">
            <stop offset="0%" stopColor="#2b1f55" stopOpacity="0.45" />
            <stop offset="50%" stopColor="#14102c" stopOpacity="0.25" />
            <stop offset="100%" stopColor={ambientColor} stopOpacity="0" />
          </radialGradient>

          {/* Luminous Window Beam Gradient */}
          <linearGradient id="dappledBeam" x1="85%" y1="15%" x2="35%" y2="75%">
            <stop offset="0%" stopColor="#c2b6ff" stopOpacity="0.95" />
            <stop offset="25%" stopColor={lightColor} stopOpacity="0.88" />
            <stop offset="55%" stopColor="#705cdb" stopOpacity="0.75" />
            <stop offset="85%" stopColor="#4c39ad" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#231859" stopOpacity="0.05" />
          </linearGradient>

          {/* Core sunlight glow */}
          <radialGradient id="dappledCore" cx="68%" cy="40%" r="45%">
            <stop offset="0%" stopColor="#dcd4ff" stopOpacity="0.35" />
            <stop offset="60%" stopColor="#8771ea" stopOpacity="0.12" />
            <stop offset="100%" stopColor={ambientColor} stopOpacity="0" />
          </radialGradient>

          {/* Soft shadow blur (matching sunlight dappling in photo) */}
          <filter id="dappledShadowBlur" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation={blur} />
          </filter>

          {/* Outer beam softness */}
          <filter id="dappledEdge" x="-15%" y="-15%" width="130%" height="130%">
            <feGaussianBlur stdDeviation="8" />
          </filter>

          {/* Film Grain Filter */}
          {filmGrain && (
            <filter id="dappledGrain">
              <feTurbulence type="fractalNoise" baseFrequency="0.75" numOctaves="3" result="noise" />
              <feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.05 0" />
              <feComposite in2="SourceGraphic" in="glitch" operator="in" />
            </filter>
          )}
        </defs>

        {/* Ambient wall background */}
        <rect width="1600" height="1000" fill={ambientColor} />
        <rect width="1600" height="1000" fill="url(#dappledAmbient)" />

        {/* Outer soft ambient light bloom */}
        <g filter="url(#dappledEdge)">
          <polygon
            points="460,540 1360,-15 1500,160 600,720"
            fill="url(#dappledBeam)"
            opacity="0.5"
          />
        </g>

        {/* Sunlight Window Panes */}
        {windowFrame ? (
          <g>
            {/* Top-Left Window Pane */}
            <polygon points="680,410 930,260 1060,410 810,560" fill="url(#dappledBeam)" filter="url(#dappledEdge)" opacity="0.9" />
            <polygon points="680,410 930,260 1060,410 810,560" fill="url(#dappledBeam)" />

            {/* Top-Right Window Pane (where the King shadow falls) */}
            <polygon points="960,242 1340,15 1470,165 1090,392" fill="url(#dappledBeam)" filter="url(#dappledEdge)" opacity="0.92" />
            <polygon points="960,242 1340,15 1470,165 1090,392" fill="url(#dappledBeam)" />

            {/* Bottom-Left Window Pane */}
            <polygon points="470,540 720,390 850,540 600,690" fill="url(#dappledBeam)" filter="url(#dappledEdge)" opacity="0.85" />
            <polygon points="470,540 720,390 850,540 600,690" fill="url(#dappledBeam)" />

            {/* Bottom-Right Window Pane */}
            <polygon points="750,372 1130,145 1260,295 880,522" fill="url(#dappledBeam)" filter="url(#dappledEdge)" opacity="0.9" />
            <polygon points="750,372 1130,145 1260,295 880,522" fill="url(#dappledBeam)" />

            {/* Sunlight beam core haze */}
            <polygon points="470,540 1340,15 1470,165 600,690" fill="url(#dappledCore)" />
          </g>
        ) : (
          <polygon points="460,540 1360,-15 1500,160 600,720" fill="url(#dappledBeam)" />
        )}

        {/* ── SWAYING SHADOW (KING SHADOW ON TOP RIGHT) ── */}
        {shadowType === 'king' ? (
          <g
            filter="url(#dappledShadowBlur)"
            opacity="0.93"
            transform={`translate(${1080 + swayX}, ${20 + swayY}) rotate(${-16 + swayRot}) scale(1.18)`}
          >
            <path
              fill="#090710"
              d="
                M 120 18
                h 10 v 16 h 16 v 10 h -16 v 18 h -10 v -18 h -16 v -10 h 16 z

                M 125 64
                C 105 64, 88 78, 86 98
                c -1 12, 4 22, 12 30
                c 6 6, 16 10, 27 10
                s 21 -4, 27 -10
                c 8 -8, 13 -18, 12 -30
                C 162 78, 145 64, 125 64 z

                M 74 138
                c 0 -5, 12 -9, 51 -9
                s 51 4, 51 9
                c 0 6, -10 11, -22 13
                l 6 22
                c -8 4, -21 6, -35 6
                s -27 -2, -35 -6
                l 6 -22
                C 84 149, 74 144, 74 138 z

                M 92 174
                c 10 3, 22 4, 33 4
                s 23 -1, 33 -4
                c 12 40, 24 100, 28 170
                c -16 6, -38 9, -61 9
                s -45 -3, -61 -9
                c 4 -70, 16 -130, 28 -170 z

                M 60 348
                c 0 -6, 26 -11, 65 -11
                s 65 5, 65 11
                c 0 8, -12 14, -30 18
                l 12 40
                c -14 5, -31 7, -47 7
                s -33 -2, -47 -7
                l 12 -40
                C 72 362, 60 356, 60 348 z

                M 42 408
                c 0 -7, 34 -13, 83 -13
                s 83 6, 83 13
                c 0 10, -18 18, -44 23
                l 10 32
                c -15 4, -32 6, -49 6
                s -34 -2, -49 -6
                l 10 -32
                C 60 426, 42 418, 42 408 z

                M 20 464
                c 0 -10, 44 -18, 105 -18
                s 105 8, 105 18
                v 24
                c 0 10, -44 18, -105 18
                S 20 498, 20 488 z
              "
            />
          </g>
        ) : (
          /* Leaves Shadow fallback */
          <g
            filter="url(#dappledShadowBlur)"
            opacity="0.9"
            transform={`translate(${1120 + swayX}, ${40 + swayY}) rotate(${-12 + swayRot}) scale(1.1)`}
          >
            {/* Branch stem */}
            <path d="M 0 0 C 80 120, 180 260, 240 420" stroke="#090710" strokeWidth="12" fill="none" />
            {/* Swaying foliage leaves */}
            <path
              fill="#090710"
              d="
                M 80 100 Q 140 80, 150 140 Q 90 160, 80 100 Z
                M 120 180 Q 200 150, 210 220 Q 130 250, 120 180 Z
                M 160 270 Q 250 250, 260 330 Q 170 350, 160 270 Z
                M 50 150 Q -20 170, 0 230 Q 60 210, 50 150 Z
                M 90 240 Q 20 280, 50 340 Q 110 300, 90 240 Z
              "
            />
          </g>
        )}

        {/* Floating Sun Dust Motes in the light beam */}
        {dustParticles && (
          <g opacity="0.65">
            {motes.map((m) => {
              const dx = Math.sin(time * m.speedX + m.phase) * 16;
              const dy = Math.cos(time * m.speedY + m.phase) * 14;
              return (
                <circle
                  key={m.id}
                  cx={m.baseX + dx}
                  cy={m.baseY + dy}
                  r={m.r}
                  fill="#ffffff"
                  opacity={0.4 + Math.sin(time * 1.2 + m.phase) * 0.3}
                />
              );
            })}
          </g>
        )}
      </svg>
    </div>
  );
};

export default DappledLight;
