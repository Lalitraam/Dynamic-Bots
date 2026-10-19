import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { NORECT_LOGO, ICON_SVG, svgToUri } from '../utils/logo';

export const Auth: React.FC = () => {
  const navigate = useNavigate();
  const { login, register, sendOtp, verifyEmail } = useAuth();

  const [mode, setMode] = useState<'login' | 'signup' | 'verify'>('login');
  const [loginMethod, setLoginMethod] = useState<'password' | 'otp'>('password');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [otpCode, setOtpCode] = useState('');
  const [devOtp, setDevOtp] = useState<string | null>(null);
  const [resendCooldown, setResendCooldown] = useState(0);
  const [showPassword, setShowPassword] = useState(false);
  const [strength, setStrength] = useState(0);
  const [msg, setMsg] = useState<{ text: string; isError: boolean } | null>(null);
  const [loading, setLoading] = useState(false);

  const pcRef = useRef<HTMLCanvasElement | null>(null);
  const mirRef = useRef<HTMLDivElement | null>(null);
  const glowRef = useRef<HTMLDivElement | null>(null);
  const dotRef = useRef<HTMLDivElement | null>(null);

  // Resend cooldown timer
  useEffect(() => {
    if (resendCooldown <= 0) return;
    const t = setInterval(() => {
      setResendCooldown((prev) => (prev > 0 ? prev - 1 : 0));
    }, 1000);
    return () => clearInterval(t);
  }, [resendCooldown]);

  // Parallax on mirror & cursor glow
  useEffect(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let mx = window.innerWidth / 2;
    let my = window.innerHeight / 2;
    let gx = mx;
    let gy = my;

    const onPointerMove = (e: PointerEvent) => {
      mx = e.clientX;
      my = e.clientY;

      if (!reduce && mirRef.current) {
        const px = (e.clientX / window.innerWidth - 0.5).toFixed(3);
        const py = (e.clientY / window.innerHeight - 0.5).toFixed(3);
        mirRef.current.style.setProperty('--px', px);
        mirRef.current.style.setProperty('--py', py);
      }

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

    let glowAnimId: number;
    const glowLoop = () => {
      gx += (mx - gx) * 0.14;
      gy += (my - gy) * 0.14;
      if (glowRef.current) {
        glowRef.current.style.transform = `translate(${gx}px,${gy}px)`;
      }
      glowAnimId = requestAnimationFrame(glowLoop);
    };
    glowAnimId = requestAnimationFrame(glowLoop);

    // Drifting particles on the right-hand canvas
    const cv = pcRef.current;
    let animFrame: number;
    if (cv) {
      const ctx = cv.getContext('2d');
      let W = 0;
      let H = 0;
      let ps: { x: number; y: number; v: number; s: number; b: boolean }[] = [];

      const size = () => {
        const r = cv.getBoundingClientRect();
        W = cv.width = r.width;
        H = cv.height = r.height;
        ps = Array.from({ length: 70 }, () => ({
          x: Math.random() * W,
          y: Math.random() * H,
          v: 0.15 + Math.random() * 0.4,
          s: Math.random() * 1.8 + 0.6,
          b: Math.random() < 0.55,
        }));
      };

      const tick = () => {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        for (const p of ps) {
          p.y -= p.v;
          if (p.y < 0) {
            p.y = H;
            p.x = Math.random() * W;
          }
          ctx.fillStyle = p.b ? 'rgba(47,123,255,.75)' : 'rgba(255,255,255,.6)';
          ctx.fillRect(p.x, p.y, p.s, p.s);
        }
        if (!reduce) animFrame = requestAnimationFrame(tick);
      };

      size();
      tick();
      window.addEventListener('resize', size);

      return () => {
        window.removeEventListener('pointermove', onPointerMove);
        document.removeEventListener('pointerleave', onPointerLeave);
        window.removeEventListener('resize', size);
        cancelAnimationFrame(glowAnimId);
        cancelAnimationFrame(animFrame);
      };
    }

    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerleave', onPointerLeave);
      cancelAnimationFrame(glowAnimId);
    };
  }, []);

  const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = e.target.value;
    setPassword(v);
    let s = 0;
    if (v.length >= 8) s++;
    if (/[a-z]/.test(v) && /[A-Z]/.test(v)) s++;
    if (/\d/.test(v)) s++;
    if (/[^A-Za-z0-9]/.test(v)) s++;
    setStrength(s);
  };

  const handleSendOtp = async () => {
    if (!/^\S+@\S+\.\S+$/.test(email)) {
      setMsg({ text: 'Please enter a valid email address first.', isError: true });
      return;
    }
    setLoading(true);
    setMsg(null);
    try {
      const res = await sendOtp(email);
      setMode('verify');
      setResendCooldown(45);
      if (res.dev_otp) {
        setDevOtp(res.dev_otp);
      }
      setMsg({
        text: res.message || `Verification code sent to ${email}.`,
        isError: false,
      });
    } catch (err: any) {
      const detail =
        err?.response?.data?.detail ||
        err?.response?.data?.error ||
        err?.message ||
        'Failed to send verification code';
      setMsg({ text: String(detail), isError: true });
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMsg(null);

    if (mode === 'verify') {
      if (!otpCode.trim() || otpCode.trim().length !== 6) {
        setMsg({ text: 'Please enter the 6-digit verification code.', isError: true });
        return;
      }
      setLoading(true);
      try {
        await verifyEmail(email, otpCode.trim());
        navigate('/home');
      } catch (err: any) {
        const detail =
          err?.response?.data?.detail ||
          err?.response?.data?.error ||
          err?.message ||
          'Verification failed';
        setMsg({ text: String(detail), isError: true });
      } finally {
        setLoading(false);
      }
      return;
    }

    if (!/^\S+@\S+\.\S+$/.test(email)) {
      setMsg({ text: 'Enter a valid email address.', isError: true });
      return;
    }

    if (mode === 'signup') {
      if (password.length < 8) {
        setMsg({ text: 'Password needs at least 8 characters.', isError: true });
        return;
      }
    } else if (loginMethod === 'otp') {
      // Send OTP for login
      await handleSendOtp();
      return;
    } else {
      if (!password) {
        setMsg({ text: 'Enter your password.', isError: true });
        return;
      }
    }

    setLoading(true);
    try {
      if (mode === 'signup') {
        const res = await register(email, password);
        setMode('verify');
        setOtpCode('');
        setResendCooldown(45);
        if (res.dev_otp) {
          setDevOtp(res.dev_otp);
        }
        setMsg({
          text: res.message || 'Verification code sent to your email.',
          isError: false,
        });
      } else {
        await login(email, password);
        navigate('/home');
      }
    } catch (err: any) {
      const status = err?.response?.status;
      const detail =
        err?.response?.data?.detail ||
        err?.response?.data?.error ||
        err?.message ||
        'Authentication failed';

      if (status === 403 || String(detail).toLowerCase().includes('not verified')) {
        setMode('verify');
        setOtpCode('');
        setResendCooldown(45);
        setMsg({
          text: 'Email not verified. A new 6-digit code was sent to your email.',
          isError: false,
        });
      } else {
        setMsg({ text: String(detail), isError: true });
      }
    } finally {
      setLoading(false);
    }
  };

  const strengthLabels = ['Too weak', 'Weak', 'Okay', 'Good', 'Strong'];

  return (
    <div className="relative z-10 grid grid-cols-1 md:grid-cols-[minmax(0,560px)_minmax(0,1fr)] min-h-screen bg-black text-white">
      {/* Form Section */}
      <section className="flex flex-col justify-between p-[clamp(20px,4vw,44px)] px-[clamp(20px,4vw,56px)] z-10">
        <img
          src={svgToUri(NORECT_LOGO)}
          alt="Doppelgambit"
          className="h-[38px] w-auto filter drop-shadow-[0_0_10px_rgba(47,123,255,0.7)] drop-shadow-[0_0_22px_rgba(47,123,255,0.35)] cursor-pointer self-start"
          onClick={() => navigate('/')}
        />

        <div className="w-full max-w-[400px] my-9 mx-auto">
          {/* Segmented control */}
          <div className="relative grid grid-cols-2 border border-[rgba(255,255,255,0.16)] rounded-full p-1 mb-8">
            <i
              className={`absolute left-1 top-1 w-[calc(50%-4px)] h-[calc(100%-8px)] bg-white rounded-full transition-transform duration-300 ease-[cubic-bezier(0.2,0.8,0.2,1)] ${
                mode === 'signup' ? 'translate-x-full' : 'translate-x-0'
              }`}
            />
            <button
              type="button"
              className={`relative z-10 bg-transparent border-0 p-2.5 font-semibold text-[13px] cursor-pointer transition-colors duration-300 ${
                mode === 'login' ? 'text-black' : 'text-[#8b8b8b]'
              }`}
              onClick={() => {
                setMode('login');
                setMsg(null);
              }}
            >
              Sign in
            </button>
            <button
              type="button"
              className={`relative z-10 bg-transparent border-0 p-2.5 font-semibold text-[13px] cursor-pointer transition-colors duration-300 ${
                mode === 'signup' ? 'text-black' : 'text-[#8b8b8b]'
              }`}
              onClick={() => {
                setMode('signup');
                setMsg(null);
              }}
            >
              Create account
            </button>
          </div>

          <h1 className="font-head font-bold text-4xl leading-tight tracking-[-0.035em]">
            {mode === 'verify'
              ? 'Verify your email.'
              : mode === 'signup'
              ? 'Create your account.'
              : 'Welcome back.'}
          </h1>
          <p className="text-[#8b8b8b] text-sm leading-[1.55] my-2.5 mb-6">
            {mode === 'verify'
              ? 'Enter the 6-digit code sent to your email to complete verification.'
              : mode === 'signup'
              ? 'Join Doppelgambit and meet the engine that plays like you.'
              : 'Sign in to access your custom bots, training runs, and match history.'}
          </p>

          {/* Login method toggle (Password vs OTP) */}
          {mode === 'login' && (
            <div className="flex items-center gap-2 mb-5 p-1 bg-[#121212] border border-[rgba(255,255,255,0.1)] rounded-lg">
              <button
                type="button"
                onClick={() => setLoginMethod('password')}
                className={`flex-1 py-1.5 px-3 rounded-md text-xs font-mono transition-all ${
                  loginMethod === 'password'
                    ? 'bg-[#222] text-white font-semibold shadow-sm'
                    : 'text-[#8b8b8b] hover:text-white'
                }`}
              >
                Password Login
              </button>
              <button
                type="button"
                onClick={() => setLoginMethod('otp')}
                className={`flex-1 py-1.5 px-3 rounded-md text-xs font-mono transition-all ${
                  loginMethod === 'otp'
                    ? 'bg-[#222] text-white font-semibold shadow-sm'
                    : 'text-[#8b8b8b] hover:text-white'
                }`}
              >
                OTP Code Login
              </button>
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate>
            {mode === 'verify' ? (
              <div className="mb-4">
                <label className="block font-mono text-xs text-[#8b8b8b] mb-1.5 uppercase">
                  6-Digit Verification Code (OTP)
                </label>
                <input
                  type="text"
                  maxLength={6}
                  autoFocus
                  value={otpCode}
                  onChange={(e) => setOtpCode(e.target.value.replace(/\D/g, ''))}
                  placeholder="000000"
                  className="w-full bg-[#0b0b0b] border border-[rgba(255,255,255,0.16)] rounded px-3.5 py-3 text-center text-2xl tracking-[0.3em] font-mono text-white focus:border-white focus:shadow-[0_0_0_3px_rgba(255,255,255,0.12)] focus:outline-none transition-all"
                />

                {devOtp && (
                  <div
                    onClick={() => setOtpCode(devOtp)}
                    className="mt-2.5 p-2 bg-blue-950/40 border border-blue-500/30 rounded text-xs font-mono text-blue-300 flex items-center justify-between cursor-pointer hover:bg-blue-900/40 transition-colors"
                  >
                    <span>Verification Code: <strong className="text-white tracking-wider">{devOtp}</strong></span>
                    <span className="text-[11px] underline">Click to autofill</span>
                  </div>
                )}

                <div className="flex justify-between items-center mt-3 text-xs font-mono">
                  <span className="text-[#666]">Sent to {email}</span>
                  <button
                    type="button"
                    disabled={loading || resendCooldown > 0}
                    onClick={handleSendOtp}
                    className="text-[#2F7BFF] hover:underline bg-transparent border-0 cursor-pointer p-0 font-mono text-xs disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    {resendCooldown > 0 ? `Resend in ${resendCooldown}s` : 'Resend code'}
                  </button>
                </div>

                <div className="mt-4 pt-3 border-t border-[rgba(255,255,255,0.08)] flex justify-between">
                  <button
                    type="button"
                    onClick={() => {
                      setMode('login');
                      setMsg(null);
                    }}
                    className="text-xs text-[#888] hover:text-white transition-colors bg-transparent border-0 cursor-pointer font-mono"
                  >
                    ← Back to Sign In
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="mb-4">
                  <label className="block font-mono text-xs text-[#8b8b8b] mb-1.5">
                    Email
                  </label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    autoComplete="email"
                    className="w-full bg-[#0b0b0b] border border-[rgba(255,255,255,0.16)] rounded px-3.5 py-3 text-[15px] text-white focus:border-white focus:shadow-[0_0_0_3px_rgba(255,255,255,0.12)] focus:outline-none transition-all"
                  />
                </div>

                {mode === 'signup' || loginMethod === 'password' ? (
                  <div className="mb-2">
                    <label className="block font-mono text-xs text-[#8b8b8b] mb-1.5">
                      Password
                    </label>
                    <div className="relative">
                      <input
                        type={showPassword ? 'text' : 'password'}
                        value={password}
                        onChange={handlePasswordChange}
                        placeholder="••••••••"
                        autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
                        className="w-full bg-[#0b0b0b] border border-[rgba(255,255,255,0.16)] rounded px-3.5 py-3 pr-14 text-[15px] text-white focus:border-white focus:shadow-[0_0_0_3px_rgba(255,255,255,0.12)] focus:outline-none transition-all"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-2.5 top-1/2 -translate-y-1/2 bg-transparent border-0 text-[#8b8b8b] font-mono text-xs hover:text-white cursor-pointer px-1.5 py-1"
                      >
                        {showPassword ? 'Hide' : 'Show'}
                      </button>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-[#888] font-mono mb-4">
                    We'll email you a 6-digit one-time code to log in without needing a password.
                  </p>
                )}

                {mode === 'signup' && (
                  <>
                    <div className="flex gap-1.5 mt-2.5">
                      {[0, 1, 2, 3].map((idx) => (
                        <i
                          key={idx}
                          className={`flex-1 h-[3px] rounded transition-all duration-300 ${
                            idx < strength
                              ? strength === 4
                                ? 'bg-[#2F7BFF] shadow-[0_0_10px_#2F7BFF]'
                                : 'bg-white'
                              : 'bg-[rgba(255,255,255,0.16)]'
                          }`}
                        />
                      ))}
                    </div>
                    <span className="block font-mono text-[11px] text-[#8b8b8b] mt-1.5 min-h-[14px]">
                      {password ? strengthLabels[strength] : ''}
                    </span>
                  </>
                )}
              </>
            )}

            <button
              type="submit"
              disabled={loading}
              className="btn main full mt-5"
            >
              {loading
                ? 'One moment…'
                : mode === 'verify'
                ? 'Verify code'
                : mode === 'signup'
                ? 'Create account'
                : loginMethod === 'otp'
                ? 'Send One-Time Code'
                : 'Sign in'}
            </button>

            {mode !== 'verify' && (
              <a
                href="/"
                onClick={(e) => {
                  e.preventDefault();
                  navigate('/');
                }}
                className="block text-center mt-5 text-white text-sm hover:underline cursor-pointer"
              >
                <span className="border-b border-[rgba(255,255,255,0.16)] pb-0.5 hover:border-white transition-colors">
                  Play as guest
                </span>{' '}
                →
              </a>
            )}

            {msg && (
              <div
                id="msg"
                role="status"
                className={`font-mono text-xs mt-3.5 min-h-[18px] p-2.5 rounded border ${
                  msg.isError
                    ? 'text-red-400 border-red-900/50 bg-red-950/20'
                    : 'text-blue-400 border-blue-900/50 bg-blue-950/20'
                }`}
              >
                {msg.text}
              </div>
            )}
          </form>
        </div>

        <p className="font-mono text-[11px] text-[#8b8b8b] text-center">
          By continuing you agree to the Terms and Privacy Policy.
        </p>
      </section>

      {/* Art / Mirror Section */}
      <section
        className="relative overflow-hidden border-l border-[rgba(255,255,255,0.16)] hidden md:flex flex-col justify-end p-[clamp(24px,4vw,56px)] bg-[radial-gradient(70%_55%_at_50%_38%,rgba(47,123,255,0.2),transparent_70%)]"
        aria-hidden="true"
      >
        <canvas ref={pcRef} id="pc" className="absolute inset-0 w-full h-full" />
        <div
          className="absolute left-0 right-0 bottom-0 h-[46%]"
          style={{
            background:
              'conic-gradient(rgba(255,255,255,0.06) 25%, transparent 0 50%, rgba(255,255,255,0.06) 0 75%, transparent 0) 0 0/72px 72px',
            maskImage: 'linear-gradient(to top, #000, transparent)',
            WebkitMaskImage: 'linear-gradient(to top, #000, transparent)',
          }}
        />

        <div className="absolute font-mono text-[11px] text-[#8b8b8b] top-[26px] left-[30px]">
          // your style
        </div>
        <div className="absolute font-mono text-[11px] text-[#8b8b8b] top-[26px] right-[30px]">
          mirrored //
        </div>

        <div
          ref={mirRef}
          className="mir absolute left-1/2 top-[9%] w-[min(44%,250px)] -translate-x-1/2"
          style={{ '--px': 0, '--py': 0 } as React.CSSProperties}
        >
          <img
            src={svgToUri(ICON_SVG)}
            alt=""
            className="up block w-full filter drop-shadow-[0_0_26px_rgba(47,123,255,0.6)]"
            style={{
              transform: 'translate(calc(var(--px)*16px), calc(var(--py)*10px))',
              transition: 'transform 0.25s',
            }}
          />
          <div className="seam h-[1px] -mx-[45%] bg-gradient-to-r from-transparent via-white to-transparent shadow-[0_0_18px_1px_rgba(47,123,255,0.9)]" />
          <img
            src={svgToUri(ICON_SVG)}
            alt=""
            className="rf block w-full opacity-30 filter drop-shadow-[0_0_26px_rgba(47,123,255,0.6)]"
            style={{
              transform: 'translate(calc(var(--px)*16px), calc(var(--py)*-10px)) scaleY(-1)',
              transition: 'transform 0.25s',
              maskImage: 'linear-gradient(to bottom, transparent 0, #000 85%)',
              WebkitMaskImage: 'linear-gradient(to bottom, transparent 0, #000 85%)',
            }}
          />
        </div>

        <p className="relative font-head font-bold text-[clamp(26px,3.2vw,46px)] leading-[1.08] tracking-[-0.035em]">
          Every move you make,
          <br />
          <span className="text-[#8b8b8b]">it remembers.</span>
        </p>
      </section>

      {/* Glow Cursor elements */}
      <div
        ref={glowRef}
        id="glow"
        className="fixed left-0 top-0 w-[420px] h-[420px] -ml-[210px] -mt-[210px] rounded-full pointer-events-none z-50 mix-blend-screen opacity-0 transition-opacity duration-300"
        style={{
          background: 'radial-gradient(circle, rgba(255,255,255,0.16), rgba(47,123,255,0.06) 38%, transparent 68%)',
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

export default Auth;
