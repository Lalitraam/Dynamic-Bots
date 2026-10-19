import React from 'react';

/**
 * WindowKingBackground renders the angled periwinkle/violet window light beam
 * with a crisp chess King shadow in the top-right quadrant, matching the visual
 * style and angle of the reference image.
 */
export const WindowKingBackground: React.FC = () => {
  return (
    <div
      className="fixed inset-0 pointer-events-none z-0 overflow-hidden select-none"
      aria-hidden="true"
      style={{ backgroundColor: '#07070a' }}
    >
      <svg
        className="w-full h-full object-cover"
        viewBox="0 0 1600 1000"
        preserveAspectRatio="xMidYMid slice"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          {/* Subtle ambient wall glow */}
          <radialGradient id="ambientGlow" cx="72%" cy="38%" r="65%">
            <stop offset="0%" stopColor="#2b1f55" stopOpacity="0.45" />
            <stop offset="50%" stopColor="#14102c" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#07070a" stopOpacity="0" />
          </radialGradient>

          {/* Luminous window beam gradient */}
          <linearGradient id="windowLightGrad" x1="85%" y1="15%" x2="35%" y2="75%">
            <stop offset="0%" stopColor="#c2b6ff" stopOpacity="0.95" />
            <stop offset="25%" stopColor="#9a87f5" stopOpacity="0.88" />
            <stop offset="55%" stopColor="#705cdb" stopOpacity="0.75" />
            <stop offset="85%" stopColor="#4c39ad" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#231859" stopOpacity="0.05" />
          </linearGradient>

          {/* Atmospheric haze layer */}
          <radialGradient id="beamCoreGlow" cx="68%" cy="40%" r="45%">
            <stop offset="0%" stopColor="#dcd4ff" stopOpacity="0.3" />
            <stop offset="60%" stopColor="#8771ea" stopOpacity="0.1" />
            <stop offset="100%" stopColor="#07070a" stopOpacity="0" />
          </radialGradient>

          {/* Realistic soft shadow blur filter (matching the leaves blur in photo) */}
          <filter id="shadowBlur" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="2.2" />
          </filter>

          {/* Outer beam edge diffusion */}
          <filter id="beamEdgeSoft" x="-10%" y="-10%" width="120%" height="120%">
            <feGaussianBlur stdDeviation="7" />
          </filter>

          {/* Window Frame Panes Mask: creates the 4-pane cross mullion bars */}
          <mask id="windowPanesMask">
            {/* White background enables full visibility */}
            <rect width="1600" height="1000" fill="#000000" />

            {/* The 4 angled window panes */}
            {/* Top-Left Pane */}
            <polygon
              points="690,390 940,240 1070,390 820,540"
              fill="#ffffff"
            />
            {/* Top-Right Pane */}
            <polygon
              points="970,222 1350, -5 1480,145 1100,372"
              fill="#ffffff"
            />
            {/* Bottom-Left Pane */}
            <polygon
              points="480,520 730,370 860,520 610,670"
              fill="#ffffff"
            />
            {/* Bottom-Right Pane */}
            <polygon
              points="760,352 1140,125 1270,275 890,502"
              fill="#ffffff"
            />
          </mask>

          {/* Clip path for the whole beam area so King shadow stays within the light */}
          <clipPath id="windowBeamClip">
            <polygon points="450,560 1370,-30 1520,170 590,750" />
          </clipPath>
        </defs>

        {/* Ambient room backdrop */}
        <rect width="1600" height="1000" fill="#07070a" />
        <rect width="1600" height="1000" fill="url(#ambientGlow)" />

        {/* Outer soft light bloom behind window */}
        <g filter="url(#beamEdgeSoft)">
          <polygon
            points="460,540 1360,-15 1500,160 600,720"
            fill="url(#windowLightGrad)"
            opacity="0.5"
          />
        </g>

        {/* Main 4-Pane Angled Window Projection */}
        <g>
          {/* Top-Left Pane */}
          <polygon
            points="680,410 930,260 1060,410 810,560"
            fill="url(#windowLightGrad)"
            filter="url(#beamEdgeSoft)"
            opacity="0.9"
          />
          <polygon
            points="680,410 930,260 1060,410 810,560"
            fill="url(#windowLightGrad)"
          />

          {/* Top-Right Pane (where the King shadow falls clearly) */}
          <polygon
            points="960,242 1340,15 1470,165 1090,392"
            fill="url(#windowLightGrad)"
            filter="url(#beamEdgeSoft)"
            opacity="0.9"
          />
          <polygon
            points="960,242 1340,15 1470,165 1090,392"
            fill="url(#windowLightGrad)"
          />

          {/* Bottom-Left Pane */}
          <polygon
            points="470,540 720,390 850,540 600,690"
            fill="url(#windowLightGrad)"
            filter="url(#beamEdgeSoft)"
            opacity="0.85"
          />
          <polygon
            points="470,540 720,390 850,540 600,690"
            fill="url(#windowLightGrad)"
          />

          {/* Bottom-Right Pane */}
          <polygon
            points="750,372 1130,145 1260,295 880,522"
            fill="url(#windowLightGrad)"
            filter="url(#beamEdgeSoft)"
            opacity="0.9"
          />
          <polygon
            points="750,372 1130,145 1260,295 880,522"
            fill="url(#windowLightGrad)"
          />

          {/* Atmosphere core beam glow */}
          <polygon
            points="470,540 1340,15 1470,165 600,690"
            fill="url(#beamCoreGlow)"
          />
        </g>

        {/* ── KING CHESS SHADOW (Only on Top Right, clear like the photo) ── */}
        <g
          filter="url(#shadowBlur)"
          opacity="0.93"
          transform="translate(1080, 20) rotate(-16) scale(1.18)"
        >
          {/* Subtle king shadow cast direction */}
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
      </svg>
    </div>
  );
};
