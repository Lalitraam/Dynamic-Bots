import React, { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { NORECT_LOGO, svgToUri } from '../utils/logo';

interface Point {
  tx: number;
  ty: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  c: string;
  s: number;
  d: number;
}

interface Swarm {
  P: Point[];
  W: number;
  H: number;
  m: { x: number; y: number };
  t0: number;
  tok: number;
  cv: HTMLCanvasElement;
  glitch: boolean;
  build: () => void;
  scatter: (now: number) => void;
  draw: (now: number) => void;
}

export const Landing: React.FC = () => {
  const navigate = useNavigate();
  const logoCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const textCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const glowRef = useRef<HTMLDivElement | null>(null);
  const dotRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const VB = [56, 45, 98, 128];
    const BLUES = ['#1E40C8', '#1E40C8', '#2552F0', '#2552F0', '#2F7BFF'];

    // Cursor tracking
    let mx = window.innerWidth / 2;
    let my = window.innerHeight / 2;
    let gx = mx;
    let gy = my;

    const onPointerMove = (e: PointerEvent) => {
      mx = e.clientX;
      my = e.clientY;
      if (glowRef.current && dotRef.current) {
        glowRef.current.style.opacity = '1';
        dotRef.current.style.opacity = '1';
        dotRef.current.style.transform = `translate(${mx}px,${my}px)`;
      }
    };

    const onPointerLeave = () => {
      if (glowRef.current && dotRef.current) {
        glowRef.current.style.opacity = '0';
        dotRef.current.style.opacity = '0';
      }
    };

    window.addEventListener('pointermove', onPointerMove);
    document.addEventListener('pointerleave', onPointerLeave);

    let cursorAnimId: number;
    const cursorLoop = () => {
      gx += (mx - gx) * 0.14;
      gy += (my - gy) * 0.14;
      if (glowRef.current) {
        glowRef.current.style.transform = `translate(${gx}px,${gy}px)`;
      }
      cursorAnimId = requestAnimationFrame(cursorLoop);
    };
    cursorAnimId = requestAnimationFrame(cursorLoop);

    // Glitch bands state
    const G = {
      until: 0,
      next: 3500,
      bands: [] as [number, number, number][],
    };

    const makeSwarm = (
      cv: HTMLCanvasElement,
      sampler: (w: number, h: number, done: (pts: { x: number; y: number; c: string; s: number }[]) => void) => void
    ): Swarm => {
      const ctx = cv.getContext('2d')!;
      const S: Swarm = {
        P: [],
        W: 0,
        H: 0,
        m: { x: -999, y: -999 },
        t0: 0,
        tok: 0,
        cv,
        glitch: false,
        build: () => {},
        scatter: () => {},
        draw: () => {},
      };

      const handlePointerMove = (e: PointerEvent) => {
        const r = cv.getBoundingClientRect();
        S.m.x = e.clientX - r.left;
        S.m.y = e.clientY - r.top;
      };
      const handlePointerLeave = () => {
        S.m.x = -999;
        S.m.y = -999;
      };

      cv.addEventListener('pointermove', handlePointerMove);
      cv.addEventListener('pointerleave', handlePointerLeave);

      S.build = () => {
        const id = ++S.tok;
        const r = cv.getBoundingClientRect();
        S.W = r.width;
        S.H = r.height;
        cv.width = S.W * dpr;
        cv.height = S.H * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        sampler(S.W, S.H, (pts) => {
          if (id !== S.tok) return;
          S.P = pts.map((q) => ({
            tx: q.x,
            ty: q.y,
            x: Math.random() * S.W,
            y: Math.random() * S.H,
            vx: 0,
            vy: 0,
            c: q.c,
            s: q.s,
            d: Math.random() * 0.9,
          }));
          if (reduce) {
            S.P.forEach((p) => {
              p.x = p.tx;
              p.y = p.ty;
            });
          }
          S.t0 = performance.now();
        });
      };

      S.scatter = (now: number) => {
        S.P.forEach((p) => {
          const a = Math.random() * 6.28;
          const f = 10 + Math.random() * 16;
          p.vx = Math.cos(a) * f;
          p.vy = Math.sin(a) * f;
        });
        S.t0 = now + 900;
      };

      S.draw = (now: number) => {
        const W = S.W;
        const H = S.H;
        const P = S.P;
        const m = S.m;
        ctx.clearRect(0, 0, W, H);
        ctx.globalCompositeOperation = 'lighter';
        const age = (now - S.t0) / 1000;
        const pulse = 0.07 + 0.03 * Math.sin(now / 600);
        const gb = S.glitch && now < G.until && age > 2 ? G.bands : null;

        for (const p of P) {
          if (!reduce) {
            if (age > p.d) {
              const k = 0.045 + p.d * 0.02;
              p.vx += (p.tx - p.x) * k;
              p.vy += (p.ty - p.y) * k;
            } else {
              p.x += Math.sin(now / 300 + p.ty) * 0.6;
              p.y += Math.cos(now / 280 + p.tx) * 0.6;
            }
            const dx = p.x - m.x;
            const dy = p.y - m.y;
            const dd = dx * dx + dy * dy;
            if (dd < 7000) {
              const f = (1 - dd / 7000) * 2.4;
              const r = Math.sqrt(dd) || 1;
              p.vx += (dx / r) * f;
              p.vy += (dy / r) * f;
            }
            p.vx *= 0.86;
            p.vy *= 0.86;
            p.x += p.vx;
            p.y += p.vy;
          }
          let x = p.x;
          let c = p.c;
          if (gb) {
            for (const b of gb) {
              if (p.ty >= b[0] && p.ty < b[1]) {
                x += b[2];
                c = '#fff';
                break;
              }
            }
          }
          ctx.fillStyle = c;
          ctx.globalAlpha = pulse;
          ctx.fillRect(x - 3, p.y - 3, p.s + 6, p.s + 6);
          ctx.globalAlpha = 0.95;
          ctx.fillRect(x, p.y, p.s, p.s);
        }
        ctx.globalAlpha = 1;
        ctx.globalCompositeOperation = 'source-over';
      };

      return S;
    };

    const logoSampler = (
      W: number,
      H: number,
      done: (pts: { x: number; y: number; c: string; s: number }[]) => void
    ) => {
      const sc = Math.min((W * 0.8) / VB[2], (H * 0.8) / VB[3]);
      const iw = Math.round(VB[2] * sc);
      const ih = Math.round(VB[3] * sc);
      const svg = NORECT_LOGO.replace(
        /width="762" height="205" viewBox="0 0 762 205"/,
        `width="${iw}" height="${ih}" viewBox="${VB.join(' ')}"`
      );
      const img = new Image();
      img.onload = () => {
        const o = document.createElement('canvas');
        o.width = Math.ceil(W);
        o.height = Math.ceil(H);
        const g = o.getContext('2d')!;
        g.drawImage(img, (W - iw) / 2, (H - ih) / 2, iw, ih);
        const d = g.getImageData(0, 0, o.width, o.height).data;
        const step = Math.max(3, Math.round(sc * 1.05));
        const pts: { x: number; y: number; c: string; s: number }[] = [];
        for (let y = 0; y < o.height; y += step) {
          for (let x = 0; x < o.width; x += step) {
            const i = (y * o.width + x) * 4;
            const R = d[i];
            const G_ = d[i + 1];
            const B = d[i + 2];
            if (d[i + 3] > 140 && R + G_ + B > 90) {
              pts.push({
                x,
                y,
                c: `rgb(${R},${G_},${B})`,
                s: step * 0.55 + Math.random() * 0.8,
              });
            }
          }
        }
        done(pts);
      };
      img.src = svgToUri(svg);
    };

    const textSampler = (
      W: number,
      H: number,
      done: (pts: { x: number; y: number; c: string; s: number }[]) => void
    ) => {
      const font = (s: number) => `700 ${s}px "Space Grotesk","Helvetica Neue",Arial,sans-serif`;
      Promise.resolve(document.fonts && document.fonts.load(font(100)))
        .catch(() => {})
        .then(() => {
          const o = document.createElement('canvas');
          o.width = Math.ceil(W);
          o.height = Math.ceil(H);
          const g = o.getContext('2d')!;
          g.font = font(100);
          const ratio = g.measureText('DOPPELGAMBIT').width / 100;
          const fs = Math.min(H * 0.82, (W * 0.985) / ratio);
          g.font = font(fs);
          g.textBaseline = 'middle';
          g.fillStyle = '#fff';
          g.fillText('DOPPELGAMBIT', 2, H / 2 + fs * 0.04);
          const d = g.getImageData(0, 0, o.width, o.height).data;
          const pts: { x: number; y: number; c: string; s: number }[] = [];
          for (let y = 0; y < o.height; y += 3) {
            for (let x = 0; x < o.width; x += 3) {
              if (d[(y * o.width + x) * 4 + 3] > 140) {
                pts.push({
                  x,
                  y,
                  c: BLUES[Math.floor(Math.random() * BLUES.length)],
                  s: 2.2 + Math.random() * 0.5,
                });
              }
            }
          }
          done(pts);
        });
    };

    if (!logoCanvasRef.current || !textCanvasRef.current) return;

    const logoS = makeSwarm(logoCanvasRef.current, logoSampler);
    const textS = makeSwarm(textCanvasRef.current, textSampler);
    textS.glitch = true;
    const swarms = [logoS, textS];
    let nextAt = performance.now() + 10000;

    const scatterAll = (now: number) => {
      swarms.forEach((s) => s.scatter(now));
      nextAt = now + 10000;
    };

    const handleCanvasClick = () => scatterAll(performance.now());
    swarms.forEach((s) => s.cv.addEventListener('click', handleCanvasClick));

    let frameId: number;
    const frame = (now: number) => {
      if (!reduce && now > nextAt) scatterAll(now);
      if (now > G.next) {
        G.next = now + 2200 + Math.random() * 3200;
        G.until = now + 380;
        const h = textS.H;
        G.bands = [0, 1, 2].map(() => {
          const y = Math.random() * h * 0.8;
          return [y, y + h * (0.08 + Math.random() * 0.15), (Math.random() - 0.5) * 28];
        });
      }
      swarms.forEach((s) => s.draw(now));
      frameId = requestAnimationFrame(frame);
    };

    swarms.forEach((s) => s.build());
    frameId = requestAnimationFrame(frame);

    let rt: number;
    const handleResize = () => {
      window.clearTimeout(rt);
      rt = window.setTimeout(() => {
        swarms.forEach((s) => s.build());
        nextAt = performance.now() + 10000;
      }, 200);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerleave', onPointerLeave);
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(cursorAnimId);
      cancelAnimationFrame(frameId);
      swarms.forEach((s) => s.cv.removeEventListener('click', handleCanvasClick));
    };
  }, []);

  return (
    <div className="min-h-screen relative overflow-x-hidden bg-black text-white selection:bg-blue-600 selection:text-white flex flex-col justify-between">
      {/* Header */}
      <header className="relative z-10 flex justify-between items-center px-[clamp(20px,5vw,64px)] py-5">
        <img
          id="hl"
          src={svgToUri(NORECT_LOGO)}
          alt="Doppelgambit"
          className="h-10 w-auto filter drop-shadow-[0_0_10px_rgba(47,123,255,0.7)] drop-shadow-[0_0_22px_rgba(47,123,255,0.35)] cursor-pointer"
          onClick={() => navigate('/')}
        />
        <nav className="flex gap-7 items-center text-sm text-[#8b8b8b]">
          <a
            href="#play"
            onClick={(e) => {
              e.preventDefault();
              navigate('/login');
            }}
            className="hover:text-white transition-colors"
          >
            Play
          </a>
          <a
            href="#analysis"
            onClick={(e) => {
              e.preventDefault();
              navigate('/login');
            }}
            className="hover:text-white transition-colors"
          >
            Analysis
          </a>
          <a
            href="#ratings"
            onClick={(e) => {
              e.preventDefault();
              navigate('/login');
            }}
            className="hover:text-white transition-colors"
          >
            Ratings
          </a>
          <button
            className="pill"
            onClick={() => navigate('/login')}
          >
            Sign in
          </button>
        </nav>
      </header>

      {/* Main Hero */}
      <main className="relative z-10 grid grid-cols-1 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)] items-center flex-1 px-[clamp(20px,5vw,64px)] py-6">
        <section className="py-5">
          <div className="font-mono text-xs text-[#8b8b8b] flex gap-2.5 items-center mb-6">
            <b className="w-[7px] h-[7px] rounded-full bg-white shadow-[0_0_10px_#fff] animate-[blink_2s_infinite]"></b>
            Engine online
          </div>

          <h1 className="name mb-2">
            <span className="sr-only">DOPPELGAMBIT</span>
            <canvas
              ref={textCanvasRef}
              id="t"
              aria-hidden="true"
              className="block w-full h-[clamp(64px,8.5vw,130px)] cursor-pointer filter drop-shadow-[0_0_12px_rgba(37,82,240,0.65)]"
            />
          </h1>

          <p className="font-head font-medium text-[clamp(20px,2.4vw,30px)] my-4 text-white">
            It plays like you. Then it beats you.
          </p>

          <p className="text-[#8b8b8b] leading-[1.65] max-w-[46ch] text-base mb-8">
            Doppelgambit learns your openings and habits, then plays your own style back at you. Every
            game ends with a clear review of what to fix.
          </p>

          <div className="flex gap-3.5 flex-wrap">
            <button
              className="btn main"
              onClick={() => navigate('/login')}
            >
              Challenge the bot
            </button>
            <button
              className="btn ghost"
              onClick={() => navigate('/login')}
            >
              Play as guest
            </button>
          </div>
        </section>

        {/* Stage */}
        <section
          className="relative h-[min(76vh,700px)] min-h-[380px] order-first lg:order-none"
          aria-label="Doppelgambit logo forming from particles"
        >
          <canvas
            ref={logoCanvasRef}
            id="c"
            className="absolute inset-0 w-full h-full cursor-pointer filter drop-shadow-[0_0_16px_rgba(47,123,255,0.45)]"
          />
          <div className="absolute bottom-[2%] left-0 font-mono text-[11px] text-[#8b8b8b] pointer-events-none">
            click to scatter · re-forms every 10 seconds
          </div>
        </section>
      </main>

      {/* Tech Developer Attribution Footer */}
      <footer className="relative z-10 border-t border-[rgba(255,255,255,0.08)] bg-black/80 backdrop-blur-md px-[clamp(20px,5vw,64px)] py-4 mt-auto">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 font-mono text-xs">
          <div className="flex items-center gap-3 text-[#8b8b8b]">
            <span className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-[#777]">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#2F7BFF] shadow-[0_0_8px_#2F7BFF] animate-pulse" />
              SYS.BUILD // v1.0.4
            </span>
            <span className="text-white/20">|</span>
            <span className="text-[11px] uppercase tracking-widest text-[#a3a3a3]">
              DEVELOPED BY
            </span>
          </div>

          <div className="flex items-center gap-2.5 flex-wrap">
            <a
              href="https://github.com/abhx12"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="GitHub profile of abhx12"
              className="group flex items-center gap-2 px-3 py-1.5 rounded border border-[rgba(255,255,255,0.12)] bg-[rgba(255,255,255,0.02)] hover:border-[#2F7BFF] hover:bg-[rgba(47,123,255,0.08)] hover:shadow-[0_0_16px_rgba(47,123,255,0.35)] transition-all duration-200"
            >
              <svg
                className="w-3.5 h-3.5 text-[#8b8b8b] group-hover:text-white transition-colors"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  clipRule="evenodd"
                  d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                />
              </svg>
              <span className="font-mono text-xs tracking-tight text-white group-hover:text-blue-200">
                abhx12
              </span>
              <span className="text-[10px] text-[#2F7BFF] opacity-50 group-hover:opacity-100 transition-opacity">
                ↗
              </span>
            </a>

            <span className="text-[#555] font-mono text-xs select-none">&amp;</span>

            <a
              href="https://github.com/Lalitraam"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="GitHub profile of Lalitraam"
              className="group flex items-center gap-2 px-3 py-1.5 rounded border border-[rgba(255,255,255,0.12)] bg-[rgba(255,255,255,0.02)] hover:border-[#2F7BFF] hover:bg-[rgba(47,123,255,0.08)] hover:shadow-[0_0_16px_rgba(47,123,255,0.35)] transition-all duration-200"
            >
              <svg
                className="w-3.5 h-3.5 text-[#8b8b8b] group-hover:text-white transition-colors"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  clipRule="evenodd"
                  d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                />
              </svg>
              <span className="font-mono text-xs tracking-tight text-white group-hover:text-blue-200">
                Lalitraam
              </span>
              <span className="text-[10px] text-[#2F7BFF] opacity-50 group-hover:opacity-100 transition-opacity">
                ↗
              </span>
            </a>
          </div>
        </div>
      </footer>

      {/* Glow Cursor elements */}
      <div
        ref={glowRef}
        id="glow"
        className="fixed left-0 top-0 w-[420px] h-[420px] -ml-[210px] -mt-[210px] rounded-full pointer-events-none z-50 mix-blend-screen opacity-0 transition-opacity duration-300"
        style={{
          background: 'radial-gradient(circle, rgba(255,255,255,0.2), rgba(47,123,255,0.07) 38%, transparent 68%)',
        }}
      />
      <div
        ref={dotRef}
        id="dot"
        className="fixed left-0 top-0 w-2 h-2 -ml-1 -mt-1 rounded-full bg-white shadow-[0_0_14px_3px_rgba(255,255,255,0.8)] pointer-events-none z-[51] opacity-0"
      />
    </div>
  );
};

export default Landing;
