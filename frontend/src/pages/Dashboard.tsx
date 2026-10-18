import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import apiClient from '../api/client';
import { NORECT_LOGO, svgToUri } from '../utils/logo';

interface ModelCard {
  version: string;
  top1?: number | null;
  nll?: number | null;
  personalisation_status: string;
  personalized: boolean;
  decision?: string | null;
}

interface OpeningMove {
  move: string;
  count: number;
  probability: number;
}

interface Openings {
  white: OpeningMove[];
  black: OpeningMove[];
}

interface ErrorStat {
  bucket: string;
  clock_ratio: string;
  error_rate: number;
  count: number;
}

interface DashboardData {
  username: string;
  rating?: number | null;
  model_card: ModelCard;
  openings: Openings;
  time_management: Record<string, number>;
  errors: ErrorStat[];
}

export const Dashboard: React.FC = () => {
  const { username = '' } = useParams<{ username: string }>();
  const navigate = useNavigate();

  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [openingTab, setOpeningTab] = useState<'white' | 'black'>('white');

  useEffect(() => {
    const fetchDashboard = async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await apiClient.get<DashboardData>(`/api/bots/${username}/dashboard`);
        setData(res.data);
      } catch (err: any) {
        console.error('Failed to load dashboard:', err);
        setError(err?.response?.data?.detail || 'Failed to load telemetry data.');
      } finally {
        setLoading(false);
      }
    };

    if (username) {
      fetchDashboard();
    }
  }, [username]);

  return (
    <div className="min-h-screen relative bg-black text-white selection:bg-blue-600 selection:text-white pb-16">
      {/* Background violet light effect */}
      <div className="fx pointer-events-none" aria-hidden="true">
        <div className="pool" />
        <div className="beam">
          <i />
          <i className="b" />
        </div>
      </div>

      {/* Header */}
      <header className="relative z-10 flex justify-between items-center px-[clamp(20px,4vw,48px)] py-4 border-b border-[rgba(255,255,255,0.1)]">
        <div className="flex items-center gap-6">
          <img
            src={svgToUri(NORECT_LOGO)}
            alt="Doppelgambit"
            className="h-9 w-auto filter drop-shadow-[0_0_10px_rgba(47,123,255,0.7)] cursor-pointer"
            onClick={() => navigate('/home')}
          />
          <div className="flex items-center gap-2 font-mono text-xs text-[#8b8b8b]">
            <span
              onClick={() => navigate('/home')}
              className="hover:text-white cursor-pointer transition-colors"
            >
              Bots
            </span>
            <span>/</span>
            <span className="text-white font-medium">{username}</span>
            <span>/</span>
            <span className="text-[#2F7BFF]">Telemetry</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate(`/play/${username}`)}
            className="btn main sm"
          >
            Play Bot
          </button>
          <button
            onClick={() => navigate('/home')}
            className="btn ghost sm"
          >
            ← Back to Roster
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-6xl mx-auto px-6 py-8">
        {loading ? (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="font-mono text-sm text-[#8b8b8b] flex items-center gap-3">
              <span className="w-2.5 h-2.5 rounded-full bg-white animate-ping" />
              Loading telemetry for {username}…
            </div>
          </div>
        ) : error ? (
          <div className="bg-[#080808] border border-red-900/50 rounded-xl p-8 max-w-lg mx-auto my-12 text-center">
            <h2 className="font-head font-bold text-xl text-red-400 mb-2">Error Loading Dashboard</h2>
            <p className="font-mono text-xs text-[#8b8b8b] mb-6">{error}</p>
            <button
              onClick={() => navigate('/home')}
              className="btn ghost sm"
            >
              Return Home
            </button>
          </div>
        ) : data ? (
          <div className="space-y-8">
            {/* Title & Overview Banner */}
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-6 border-b border-[rgba(255,255,255,0.1)]">
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <h1 className="font-head font-bold text-3xl md:text-4xl tracking-tight">
                    {data.username}
                  </h1>
                  {data.rating && (
                    <span className="font-mono text-xs bg-[#161616] border border-[rgba(255,255,255,0.18)] px-2.5 py-1 rounded-md text-[#2F7BFF] font-semibold">
                      {Math.round(data.rating)} ELO
                    </span>
                  )}
                </div>
                <p className="font-mono text-xs text-[#8b8b8b]">
                  Neural weights, opening distributions, and time management telemetry.
                </p>
              </div>

              {/* Status Badge */}
              <div>
                {data.model_card.personalized ? (
                  <span className="inline-flex items-center gap-2 font-mono text-xs bg-emerald-950/40 text-emerald-300 border border-emerald-600/60 px-3.5 py-1.5 rounded-full shadow-[0_0_12px_rgba(16,185,129,0.2)]">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    Verified Personalized Model
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-2 font-mono text-xs bg-neutral-900 text-[#8b8b8b] border border-[rgba(255,255,255,0.16)] px-3.5 py-1.5 rounded-full">
                    <span className="w-2 h-2 rounded-full bg-[#8b8b8b]" />
                    {data.model_card.personalisation_status || 'Base Model'}
                  </span>
                )}
              </div>
            </div>

            {/* Grid 1: Model Card Metrics */}
            <section className="bg-[#080808] border border-[rgba(255,255,255,0.16)] rounded-xl p-6 shadow-xl">
              <div className="flex items-center justify-between mb-5">
                <h2 className="font-head font-bold text-lg text-white">Model Card</h2>
                <span className="font-mono text-xs text-[#8b8b8b] bg-white/5 px-2.5 py-1 rounded">
                  Checkpoint: {data.model_card.version}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-[#0e0e0e] border border-[rgba(255,255,255,0.08)] p-4 rounded-lg">
                  <span className="block font-mono text-[11px] text-[#8b8b8b] uppercase tracking-wider mb-1">
                    Top-1 Accuracy
                  </span>
                  <div className="font-head font-bold text-2xl text-white">
                    {data.model_card.top1 !== null && data.model_card.top1 !== undefined
                      ? `${(data.model_card.top1 * 100).toFixed(1)}%`
                      : '—'}
                  </div>
                  <small className="font-mono text-[10px] text-[#666]">Move prediction accuracy</small>
                </div>

                <div className="bg-[#0e0e0e] border border-[rgba(255,255,255,0.08)] p-4 rounded-lg">
                  <span className="block font-mono text-[11px] text-[#8b8b8b] uppercase tracking-wider mb-1">
                    Negative Log-Likelihood
                  </span>
                  <div className="font-head font-bold text-2xl text-white">
                    {data.model_card.nll !== null && data.model_card.nll !== undefined
                      ? data.model_card.nll.toFixed(3)
                      : '—'}
                  </div>
                  <small className="font-mono text-[10px] text-[#666]">Lower is sharper</small>
                </div>

                <div className="bg-[#0e0e0e] border border-[rgba(255,255,255,0.08)] p-4 rounded-lg">
                  <span className="block font-mono text-[11px] text-[#8b8b8b] uppercase tracking-wider mb-1">
                    Fine-Tuning Decision
                  </span>
                  <div className="font-head font-bold text-2xl text-[#2F7BFF]">
                    {data.model_card.decision || 'standard'}
                  </div>
                  <small className="font-mono text-[10px] text-[#666]">Training gate decision</small>
                </div>

                <div className="bg-[#0e0e0e] border border-[rgba(255,255,255,0.08)] p-4 rounded-lg">
                  <span className="block font-mono text-[11px] text-[#8b8b8b] uppercase tracking-wider mb-1">
                    Style Verification
                  </span>
                  <div className="font-head font-bold text-2xl text-emerald-400">
                    {data.model_card.personalized ? 'Passed' : 'Standard'}
                  </div>
                  <small className="font-mono text-[10px] text-[#666]">Outperformed base prior</small>
                </div>
              </div>
            </section>

            {/* Grid 2: Openings & Time Management */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Openings Repertoire */}
              <section className="bg-[#080808] border border-[rgba(255,255,255,0.16)] rounded-xl p-6 shadow-xl flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-head font-bold text-lg text-white">Opening Repertoire</h2>
                    {/* Toggle White / Black */}
                    <div className="flex border border-[rgba(255,255,255,0.16)] rounded-lg p-0.5 font-mono text-xs">
                      <button
                        onClick={() => setOpeningTab('white')}
                        className={`px-3 py-1 rounded transition-colors ${
                          openingTab === 'white' ? 'bg-white text-black font-semibold' : 'text-[#8b8b8b]'
                        }`}
                      >
                        White 1st Moves
                      </button>
                      <button
                        onClick={() => setOpeningTab('black')}
                        className={`px-3 py-1 rounded transition-colors ${
                          openingTab === 'black' ? 'bg-white text-black font-semibold' : 'text-[#8b8b8b]'
                        }`}
                      >
                        Black Replies
                      </button>
                    </div>
                  </div>

                  <p className="font-mono text-xs text-[#8b8b8b] mb-5">
                    {openingTab === 'white'
                      ? 'Top 5 White starting moves learned from player archives.'
                      : 'Top 5 Black replies across 1-ply responses.'}
                  </p>

                  <div className="space-y-3.5">
                    {(openingTab === 'white' ? data.openings.white : data.openings.black).length === 0 ? (
                      <div className="text-center py-8 font-mono text-xs text-[#8b8b8b]">
                        No opening book data recorded.
                      </div>
                    ) : (
                      (openingTab === 'white' ? data.openings.white : data.openings.black).map(
                        (op, idx) => (
                          <div key={op.move} className="font-mono text-xs">
                            <div className="flex justify-between items-center mb-1">
                              <span className="font-semibold text-white">
                                {idx + 1}. <span className="text-[#2F7BFF] font-bold">{op.move}</span>
                              </span>
                              <span className="text-[#8b8b8b]">
                                {op.count.toLocaleString()} games ({Math.round(op.probability * 100)}%)
                              </span>
                            </div>
                            <div className="w-full bg-[#181818] h-2 rounded-full overflow-hidden">
                              <div
                                className="bg-[#2F7BFF] h-full rounded-full transition-all duration-500 shadow-[0_0_8px_rgba(47,123,255,0.7)]"
                                style={{ width: `${Math.max(4, op.probability * 100)}%` }}
                              />
                            </div>
                          </div>
                        )
                      )
                    )}
                  </div>
                </div>
              </section>

              {/* Time Management by Phase */}
              <section className="bg-[#080808] border border-[rgba(255,255,255,0.16)] rounded-xl p-6 shadow-xl flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-head font-bold text-lg text-white">Time Management</h2>
                    <span className="font-mono text-xs text-[#8b8b8b]">Mean think time (sec)</span>
                  </div>

                  <p className="font-mono text-xs text-[#8b8b8b] mb-5">
                    Average seconds spent per move across distinct phases of the game.
                  </p>

                  <div className="space-y-4">
                    {[
                      { key: 'opening', label: 'Opening Phase', desc: 'Plies 1 to 20' },
                      { key: 'middlegame', label: 'Middlegame Phase', desc: 'Plies 21 to 40' },
                      { key: 'endgame', label: 'Endgame Phase', desc: 'Plies 41+' },
                    ].map((phase) => {
                      const val = data.time_management[phase.key] ?? 0;
                      return (
                        <div
                          key={phase.key}
                          className="bg-[#0e0e0e] border border-[rgba(255,255,255,0.08)] p-4 rounded-lg flex items-center justify-between"
                        >
                          <div>
                            <span className="font-head font-bold text-white block text-sm">
                              {phase.label}
                            </span>
                            <span className="font-mono text-[11px] text-[#666]">
                              {phase.desc}
                            </span>
                          </div>
                          <div className="text-right">
                            <span className="font-mono font-bold text-xl text-white">
                              {val > 0 ? `${val.toFixed(1)}s` : '—'}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </section>
            </div>

            {/* Section 3: Error Rates by Clock Ratio */}
            <section className="bg-[#080808] border border-[rgba(255,255,255,0.16)] rounded-xl p-6 shadow-xl">
              <div className="flex items-center justify-between mb-4">
                <h2 className="font-head font-bold text-lg text-white">Error Rate vs Clock Pressure</h2>
                <span className="font-mono text-xs text-[#8b8b8b]">Blunder threshold: ≥ 100 cp loss</span>
              </div>

              <p className="font-mono text-xs text-[#8b8b8b] mb-6">
                Frequency of severe mistakes as a function of remaining time ratio (clock / base time).
              </p>

              {data.errors.length === 0 ? (
                <div className="text-center py-6 font-mono text-xs text-[#8b8b8b]">
                  No analysis error statistics logged yet.
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {data.errors.map((err) => {
                    const isTimeTrouble = err.clock_ratio.includes('<0.1');
                    const isComfortable = err.clock_ratio.includes('>0.3');
                    return (
                      <div
                        key={err.bucket}
                        className="bg-[#0e0e0e] border border-[rgba(255,255,255,0.08)] p-5 rounded-lg flex flex-col justify-between"
                      >
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-mono font-bold text-xs uppercase text-[#8b8b8b]">
                              Ratio {err.clock_ratio}
                            </span>
                            <span
                              className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                                isTimeTrouble
                                  ? 'bg-rose-950/60 text-rose-400 border border-rose-800/60'
                                  : isComfortable
                                  ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/60'
                                  : 'bg-amber-950/60 text-amber-400 border border-amber-800/60'
                              }`}
                            >
                              {isTimeTrouble ? 'Scramble' : isComfortable ? 'Calm' : 'Moderate'}
                            </span>
                          </div>

                          <div className="font-head font-bold text-3xl my-2 text-white">
                            {(err.error_rate * 100).toFixed(1)}%
                          </div>
                        </div>

                        <div className="mt-4 pt-3 border-t border-[rgba(255,255,255,0.06)] font-mono text-[11px] text-[#666]">
                          Sampled Moves: {err.count.toLocaleString()}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
          </div>
        ) : null}
      </main>
    </div>
  );
};

export default Dashboard;
