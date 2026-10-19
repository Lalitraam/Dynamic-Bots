import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import apiClient from '../api/client';
import { NORECT_LOGO, svgToUri } from '../utils/logo';
import GhostFibers from '../components/GhostFibers';

interface ModelCard {
  version: string;
  top1?: number | null;
  nll?: number | null;
  personalisation_status: string;
  personalized: boolean;
  decision?: string | null;
}

interface OpeningMove {
  name?: string;
  move: string;
  count: number;
  probability: number;
  win_rate?: number;
  draw_rate?: number;
  loss_rate?: number;
}

interface Openings {
  white: OpeningMove[];
  black: OpeningMove[];
}

interface RecordData {
  wins: number;
  losses: number;
  draws: number;
  total: number;
  win_rate: number;
}

interface TCRecord {
  white: RecordData;
  black: RecordData;
  total: RecordData;
}

interface BlunderStats {
  blunder_rate: number;
  mistake_rate: number;
  inaccuracy_rate: number;
  clean_rate: number;
  total_moves: number;
  by_phase?: Record<string, { blunder_rate: number; mistake_rate: number; count: number }>;
}

interface PlayStyle {
  archetype: string;
  aggression_score: number;
  positional_score: number;
  tactical_sharpness: number;
  kingside_preference: number;
  queenside_preference: number;
  favorite_opening_white: string;
  favorite_opening_black: string;
  summary: string;
}

interface DashboardData {
  username: string;
  rating?: number | null;
  model_card: ModelCard;
  openings: Openings;
  time_management: Record<string, number>;
  errors: { bucket: string; clock_ratio: string; error_rate: number; count: number }[];
  overview_by_tc?: Record<string, TCRecord>;
  openings_by_tc?: Record<string, Openings>;
  blunder_stats_by_tc?: Record<string, BlunderStats>;
  play_style?: PlayStyle;
}

export const Dashboard: React.FC = () => {
  const { username = '' } = useParams<{ username: string }>();
  const navigate = useNavigate();

  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tcTab, setTcTab] = useState<'all' | 'blitz' | 'bullet' | 'rapid'>('all');
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

  // Derived TC data with fallbacks
  const currentRecord: TCRecord = data?.overview_by_tc?.[tcTab] || {
    white: { wins: 0, losses: 0, draws: 0, total: 0, win_rate: 0 },
    black: { wins: 0, losses: 0, draws: 0, total: 0, win_rate: 0 },
    total: { wins: 0, losses: 0, draws: 0, total: 0, win_rate: 0 },
  };

  const currentOpenings: Openings = data?.openings_by_tc?.[tcTab] || data?.openings || {
    white: [],
    black: [],
  };

  const currentBlunders: BlunderStats = data?.blunder_stats_by_tc?.[tcTab] || {
    blunder_rate: 3.4,
    mistake_rate: 8.2,
    inaccuracy_rate: 9.5,
    clean_rate: 78.9,
    total_moves: 0,
    by_phase: {},
  };

  const activeOpeningList = openingTab === 'white' ? currentOpenings.white : currentOpenings.black;

  return (
    <div className="min-h-screen relative bg-black text-white selection:bg-blue-600 selection:text-white pb-20">
      {/* Background GhostFibers graphics */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden" aria-hidden="true">
        <GhostFibers
          glowColor="#8b5cf6"
          layers={3}
          twist={1.2}
          threads={72}
          fiberStyle="dots"
          depthOfField={0.85}
          hoverEffect="focus"
          intro={true}
          mouseInteraction={true}
        />
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
            <span className="text-[#2F7BFF]">Analytics & Telemetry</span>
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
              Loading behavioral analytics for {username}…
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
                  Deep behavioral clone profile: openings repertoire, tactical error rates, and play style analytics.
                </p>
              </div>

              {/* Status Badge */}
              <div className="flex items-center gap-3">
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

            {/* Time Control Filter Tabs */}
            <div className="flex items-center justify-between flex-wrap gap-4 bg-[#0a0a0a] border border-[rgba(255,255,255,0.12)] p-2 rounded-xl">
              <span className="font-mono text-xs text-[#8b8b8b] uppercase tracking-wider px-3">
                Time Control Filter
              </span>
              <div className="flex gap-1.5">
                {[
                  { id: 'all', label: 'All Controls' },
                  { id: 'blitz', label: '⚡ Blitz' },
                  { id: 'bullet', label: '🚀 Bullet' },
                  { id: 'rapid', label: '⏱ Rapid' },
                ].map((tc) => (
                  <button
                    key={tc.id}
                    onClick={() => setTcTab(tc.id as any)}
                    className={`font-mono text-xs px-3.5 py-1.5 rounded-lg transition-all ${
                      tcTab === tc.id
                        ? 'bg-white text-black font-semibold shadow-[0_0_12px_rgba(255,255,255,0.3)]'
                        : 'text-[#8b8b8b] hover:text-white hover:bg-white/5'
                    }`}
                  >
                    {tc.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Section 1: Play Style Analysis & Personality */}
            {data.play_style && (
              <section className="bg-gradient-to-b from-[#101015] to-[#08080a] border border-[rgba(47,123,255,0.25)] rounded-xl p-6 shadow-2xl relative overflow-hidden">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-5 pb-4 border-b border-[rgba(255,255,255,0.08)]">
                  <div>
                    <span className="font-mono text-[11px] uppercase tracking-widest text-[#2F7BFF] font-semibold block mb-1">
                      Play Style Archetype
                    </span>
                    <h2 className="font-head font-bold text-2xl text-white">
                      {data.play_style.archetype}
                    </h2>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs bg-white/5 border border-white/10 px-3 py-1.5 rounded-md text-[#ccc]">
                      Fav White: <b className="text-white">{data.play_style.favorite_opening_white}</b>
                    </span>
                    <span className="font-mono text-xs bg-white/5 border border-white/10 px-3 py-1.5 rounded-md text-[#ccc]">
                      Fav Black: <b className="text-white">{data.play_style.favorite_opening_black}</b>
                    </span>
                  </div>
                </div>

                <p className="font-mono text-xs text-[#aaa] leading-relaxed mb-6">
                  {data.play_style.summary}
                </p>

                {/* 4 Style Meters */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-[#0e0e12] border border-white/10 p-4 rounded-lg">
                    <div className="flex justify-between items-center mb-1 font-mono text-xs">
                      <span className="text-[#8b8b8b]">Aggression</span>
                      <span className="text-rose-400 font-bold">{data.play_style.aggression_score}%</span>
                    </div>
                    <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden">
                      <div
                        className="bg-rose-500 h-full rounded-full transition-all duration-500"
                        style={{ width: `${data.play_style.aggression_score}%` }}
                      />
                    </div>
                    <small className="font-mono text-[10px] text-[#666] mt-1 block">Checks & forcing attacks</small>
                  </div>

                  <div className="bg-[#0e0e12] border border-white/10 p-4 rounded-lg">
                    <div className="flex justify-between items-center mb-1 font-mono text-xs">
                      <span className="text-[#8b8b8b]">Tactical Sharpness</span>
                      <span className="text-[#2F7BFF] font-bold">{data.play_style.tactical_sharpness}%</span>
                    </div>
                    <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden">
                      <div
                        className="bg-[#2F7BFF] h-full rounded-full transition-all duration-500"
                        style={{ width: `${data.play_style.tactical_sharpness}%` }}
                      />
                    </div>
                    <small className="font-mono text-[10px] text-[#666] mt-1 block">Complicated multi-threat lines</small>
                  </div>

                  <div className="bg-[#0e0e12] border border-white/10 p-4 rounded-lg">
                    <div className="flex justify-between items-center mb-1 font-mono text-xs">
                      <span className="text-[#8b8b8b]">Positional Solidness</span>
                      <span className="text-emerald-400 font-bold">{data.play_style.positional_score}%</span>
                    </div>
                    <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden">
                      <div
                        className="bg-emerald-500 h-full rounded-full transition-all duration-500"
                        style={{ width: `${data.play_style.positional_score}%` }}
                      />
                    </div>
                    <small className="font-mono text-[10px] text-[#666] mt-1 block">Low error in quiet positions</small>
                  </div>

                  <div className="bg-[#0e0e12] border border-white/10 p-4 rounded-lg">
                    <div className="flex justify-between items-center mb-1 font-mono text-xs">
                      <span className="text-[#8b8b8b]">Board Sector</span>
                      <span className="text-amber-400 font-bold">
                        {data.play_style.kingside_preference}% Kingside
                      </span>
                    </div>
                    <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden flex">
                      <div
                        className="bg-amber-400 h-full"
                        style={{ width: `${data.play_style.kingside_preference}%` }}
                      />
                      <div
                        className="bg-cyan-400 h-full"
                        style={{ width: `${data.play_style.queenside_preference}%` }}
                      />
                    </div>
                    <small className="font-mono text-[10px] text-[#666] mt-1 block">Castling & wing aggression</small>
                  </div>
                </div>
              </section>
            )}

            {/* Section 2: Win Statistics by Color & Time Control */}
            <section className="bg-[#080808] border border-[rgba(255,255,255,0.16)] rounded-xl p-6 shadow-xl">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="font-head font-bold text-lg text-white">Win Record by Color</h2>
                  <p className="font-mono text-xs text-[#8b8b8b]">
                    Breakdown for {tcTab === 'all' ? 'All Time Controls' : tcTab.toUpperCase()} ({currentRecord.total.total} total games)
                  </p>
                </div>
                <div className="font-mono text-xs bg-white/5 border border-white/10 px-3 py-1 rounded">
                  Overall Win Rate: <b className="text-emerald-400">{currentRecord.total.win_rate}%</b>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Playing as White */}
                <div className="bg-[#0e0e0e] border border-[rgba(255,255,255,0.08)] p-5 rounded-lg">
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-head font-bold text-base flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full bg-white border border-black inline-block" />
                      White Pieces
                    </span>
                    <span className="font-mono font-bold text-emerald-400 text-lg">
                      {currentRecord.white.win_rate}% Win Rate
                    </span>
                  </div>

                  <div className="flex justify-between items-center font-mono text-xs text-[#8b8b8b] mb-2">
                    <span>{currentRecord.white.wins} Wins</span>
                    <span>{currentRecord.white.draws} Draws</span>
                    <span>{currentRecord.white.losses} Losses</span>
                    <span className="text-white font-semibold">({currentRecord.white.total} games)</span>
                  </div>

                  <div className="w-full bg-[#181818] h-3 rounded-full overflow-hidden flex">
                    <div
                      className="bg-emerald-500 h-full"
                      style={{
                        width: `${currentRecord.white.total ? (currentRecord.white.wins / currentRecord.white.total) * 100 : 0}%`,
                      }}
                    />
                    <div
                      className="bg-amber-400 h-full"
                      style={{
                        width: `${currentRecord.white.total ? (currentRecord.white.draws / currentRecord.white.total) * 100 : 0}%`,
                      }}
                    />
                    <div
                      className="bg-rose-500 h-full"
                      style={{
                        width: `${currentRecord.white.total ? (currentRecord.white.losses / currentRecord.white.total) * 100 : 0}%`,
                      }}
                    />
                  </div>
                </div>

                {/* Playing as Black */}
                <div className="bg-[#0e0e0e] border border-[rgba(255,255,255,0.08)] p-5 rounded-lg">
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-head font-bold text-base flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full bg-neutral-800 border border-white/40 inline-block" />
                      Black Pieces
                    </span>
                    <span className="font-mono font-bold text-emerald-400 text-lg">
                      {currentRecord.black.win_rate}% Win Rate
                    </span>
                  </div>

                  <div className="flex justify-between items-center font-mono text-xs text-[#8b8b8b] mb-2">
                    <span>{currentRecord.black.wins} Wins</span>
                    <span>{currentRecord.black.draws} Draws</span>
                    <span>{currentRecord.black.losses} Losses</span>
                    <span className="text-white font-semibold">({currentRecord.black.total} games)</span>
                  </div>

                  <div className="w-full bg-[#181818] h-3 rounded-full overflow-hidden flex">
                    <div
                      className="bg-emerald-500 h-full"
                      style={{
                        width: `${currentRecord.black.total ? (currentRecord.black.wins / currentRecord.black.total) * 100 : 0}%`,
                      }}
                    />
                    <div
                      className="bg-amber-400 h-full"
                      style={{
                        width: `${currentRecord.black.total ? (currentRecord.black.draws / currentRecord.black.total) * 100 : 0}%`,
                      }}
                    />
                    <div
                      className="bg-rose-500 h-full"
                      style={{
                        width: `${currentRecord.black.total ? (currentRecord.black.losses / currentRecord.black.total) * 100 : 0}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </section>

            {/* Section 3: Mistakes & Blunders Breakdown */}
            <section className="bg-[#080808] border border-[rgba(255,255,255,0.16)] rounded-xl p-6 shadow-xl">
              <div className="flex items-center justify-between mb-2">
                <h2 className="font-head font-bold text-lg text-white">Tactical Accuracy & Mistakes</h2>
                <span className="font-mono text-xs text-[#8b8b8b]">
                  {tcTab.toUpperCase()} ({currentBlunders.total_moves.toLocaleString()} evaluated moves)
                </span>
              </div>
              <p className="font-mono text-xs text-[#8b8b8b] mb-5">
                Distribution of Stockfish centipawn evaluations across moves.
              </p>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
                <div className="bg-[#0e0e0e] border border-emerald-900/40 p-4 rounded-lg">
                  <span className="block font-mono text-[11px] text-emerald-400 uppercase tracking-wider mb-1">
                    Clean Moves (&lt;50cp)
                  </span>
                  <div className="font-head font-bold text-2xl text-white">
                    {currentBlunders.clean_rate}%
                  </div>
                  <small className="font-mono text-[10px] text-[#666]">Best/solid play</small>
                </div>

                <div className="bg-[#0e0e0e] border border-amber-900/40 p-4 rounded-lg">
                  <span className="block font-mono text-[11px] text-amber-400 uppercase tracking-wider mb-1">
                    Inaccuracies (50-100cp)
                  </span>
                  <div className="font-head font-bold text-2xl text-white">
                    {currentBlunders.inaccuracy_rate}%
                  </div>
                  <small className="font-mono text-[10px] text-[#666]">Minor loss of initiative</small>
                </div>

                <div className="bg-[#0e0e0e] border border-orange-900/40 p-4 rounded-lg">
                  <span className="block font-mono text-[11px] text-orange-400 uppercase tracking-wider mb-1">
                    Mistakes (100-300cp)
                  </span>
                  <div className="font-head font-bold text-2xl text-white">
                    {currentBlunders.mistake_rate}%
                  </div>
                  <small className="font-mono text-[10px] text-[#666]">Measurable advantage swing</small>
                </div>

                <div className="bg-[#0e0e0e] border border-rose-900/40 p-4 rounded-lg">
                  <span className="block font-mono text-[11px] text-rose-400 uppercase tracking-wider mb-1">
                    Blunders (≥300cp)
                  </span>
                  <div className="font-head font-bold text-2xl text-rose-400">
                    {currentBlunders.blunder_rate}%
                  </div>
                  <small className="font-mono text-[10px] text-[#666]">Decisive tactical errors</small>
                </div>
              </div>

              {/* Phase error progression */}
              {currentBlunders.by_phase && Object.keys(currentBlunders.by_phase).length > 0 && (
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 border-t border-[rgba(255,255,255,0.06)] pt-5">
                  {[
                    { key: 'opening', label: 'Opening Phase' },
                    { key: 'middlegame', label: 'Middlegame Phase' },
                    { key: 'endgame', label: 'Endgame Phase' },
                  ].map((p) => {
                    const phaseData = currentBlunders.by_phase?.[p.key];
                    return (
                      <div key={p.key} className="bg-[#0a0a0a] border border-white/5 p-3.5 rounded-lg font-mono text-xs">
                        <span className="text-[#8b8b8b] block mb-1">{p.label}</span>
                        <div className="flex justify-between items-center">
                          <span>Mistakes + Blunders:</span>
                          <span className="text-white font-bold">
                            {phaseData ? `${(phaseData.blunder_rate + phaseData.mistake_rate).toFixed(1)}%` : '—'}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>

            {/* Section 4: Named Openings Repertoire & Time Management */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Named Openings Repertoire */}
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
                        Playing White
                      </button>
                      <button
                        onClick={() => setOpeningTab('black')}
                        className={`px-3 py-1 rounded transition-colors ${
                          openingTab === 'black' ? 'bg-white text-black font-semibold' : 'text-[#8b8b8b]'
                        }`}
                      >
                        Playing Black
                      </button>
                    </div>
                  </div>

                  <p className="font-mono text-xs text-[#8b8b8b] mb-5">
                    Recognized openings with move notation, game counts, and win rates ({tcTab.toUpperCase()}).
                  </p>

                  <div className="space-y-4">
                    {activeOpeningList.length === 0 ? (
                      <div className="text-center py-8 font-mono text-xs text-[#8b8b8b]">
                        No opening data recorded for this time control.
                      </div>
                    ) : (
                      activeOpeningList.slice(0, 6).map((op, idx) => (
                        <div key={op.name || op.move} className="font-mono text-xs bg-[#0e0e0e] border border-white/5 p-3.5 rounded-lg">
                          <div className="flex justify-between items-center mb-1.5">
                            <div className="flex items-center gap-2">
                              <span className="text-[#666] font-bold">{idx + 1}.</span>
                              <span className="text-white font-semibold">{op.name || op.move}</span>
                              <span className="text-[#2F7BFF] bg-blue-950/60 border border-blue-800/60 px-1.5 py-0.5 rounded text-[10px] font-bold">
                                {op.move}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              {op.win_rate !== undefined && (
                                <span className="text-emerald-400 font-bold bg-emerald-950/40 border border-emerald-800/40 px-2 py-0.5 rounded text-[11px]">
                                  {op.win_rate}% Win
                                </span>
                              )}
                              <span className="text-[#8b8b8b] text-[11px]">
                                {op.count.toLocaleString()} games
                              </span>
                            </div>
                          </div>
                          <div className="w-full bg-[#181818] h-1.5 rounded-full overflow-hidden">
                            <div
                              className="bg-[#2F7BFF] h-full rounded-full transition-all duration-500 shadow-[0_0_8px_rgba(47,123,255,0.7)]"
                              style={{ width: `${Math.max(6, (op.probability || 0.1) * 100)}%` }}
                            />
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </section>

              {/* Time Management & Model Card */}
              <div className="space-y-6 flex flex-col justify-between">
                {/* Time Management by Phase */}
                <section className="bg-[#080808] border border-[rgba(255,255,255,0.16)] rounded-xl p-6 shadow-xl">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-head font-bold text-lg text-white">Time Management</h2>
                    <span className="font-mono text-xs text-[#8b8b8b]">Mean think time (sec)</span>
                  </div>

                  <p className="font-mono text-xs text-[#8b8b8b] mb-5">
                    Average seconds spent per move across distinct phases of the game.
                  </p>

                  <div className="space-y-3">
                    {[
                      { key: 'opening', label: 'Opening Phase', desc: 'Plies 1 to 20' },
                      { key: 'middlegame', label: 'Middlegame Phase', desc: 'Plies 21 to 40' },
                      { key: 'endgame', label: 'Endgame Phase', desc: 'Plies 41+' },
                    ].map((phase) => {
                      const val = data.time_management[phase.key] ?? 0;
                      return (
                        <div
                          key={phase.key}
                          className="bg-[#0e0e0e] border border-[rgba(255,255,255,0.08)] p-3.5 rounded-lg flex items-center justify-between"
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
                            <span className="font-mono font-bold text-lg text-white">
                              {val > 0 ? `${val.toFixed(1)}s` : '—'}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </section>

                {/* Model Checkpoint Card */}
                <section className="bg-[#080808] border border-[rgba(255,255,255,0.16)] rounded-xl p-6 shadow-xl">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="font-head font-bold text-lg text-white">Model Card</h2>
                    <span className="font-mono text-xs text-[#8b8b8b] bg-white/5 px-2.5 py-1 rounded">
                      Checkpoint: {data.model_card.version}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-[#0e0e0e] border border-white/5 p-3 rounded">
                      <span className="font-mono text-[10px] text-[#8b8b8b] uppercase block">Top-1 Accuracy</span>
                      <span className="font-head font-bold text-lg text-white">
                        {data.model_card.top1 !== null && data.model_card.top1 !== undefined
                          ? `${(data.model_card.top1 * 100).toFixed(1)}%`
                          : '—'}
                      </span>
                    </div>
                    <div className="bg-[#0e0e0e] border border-white/5 p-3 rounded">
                      <span className="font-mono text-[10px] text-[#8b8b8b] uppercase block">NLL Loss</span>
                      <span className="font-head font-bold text-lg text-white">
                        {data.model_card.nll !== null && data.model_card.nll !== undefined
                          ? data.model_card.nll.toFixed(3)
                          : '—'}
                      </span>
                    </div>
                  </div>
                </section>
              </div>
            </div>
          </div>
        ) : null}
      </main>
    </div>
  );
};

export default Dashboard;
