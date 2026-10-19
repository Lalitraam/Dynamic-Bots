import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient from '../api/client';
import { useAuth } from '../context/AuthContext';
import { NORECT_LOGO, svgToUri } from '../utils/logo';
import { LightFallBackground } from '../components/LightFallBackground';

export interface Bot {
  username: string;
  status: string;
  active_version: number | null;
  added_at: string | null;
  archived: boolean;
  last_trained_at: string | null;
  games_used: number | null;
  tier: string | null;
  error: string | null;
  job_step?: string | null;
  job_progress?: number | null;
}

export const Home: React.FC = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const [bots, setBots] = useState<Bot[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newBotUsername, setNewBotUsername] = useState('');
  const [modalLoading, setModalLoading] = useState(false);
  const [modalError, setModalError] = useState<string | null>(null);

  const fetchBots = async (showLoader = false) => {
    try {
      if (showLoader) setLoading(true);
      const res = await apiClient.get<Bot[]>('/api/bots');
      setBots(res.data);
    } catch (err: any) {
      console.error('Failed to fetch bots:', err);
    } finally {
      if (showLoader) setLoading(false);
    }
  };

  useEffect(() => {
    fetchBots(true);
  }, []);

  useEffect(() => {
    const hasPending = bots.some(
      (b) => b.status === 'queued' || b.status === 'building'
    );
    if (!hasPending) return;

    const timer = setInterval(() => {
      fetchBots(false);
    }, 3000);

    return () => clearInterval(timer);
  }, [bots]);

  const handleCreateBot = async (e: React.FormEvent) => {
    e.preventDefault();
    const uname = newBotUsername.trim();
    if (!uname) {
      setModalError('Please enter a Lichess username.');
      return;
    }

    setModalLoading(true);
    setModalError(null);

    try {
      await apiClient.post('/api/bots', { username: uname });
      setNewBotUsername('');
      setShowCreateModal(false);
      await fetchBots();
    } catch (err: any) {
      const detail =
        err?.response?.data?.detail ||
        err?.response?.data?.error ||
        err?.message ||
        'Failed to add bot';
      setModalError(String(detail));
    } finally {
      setModalLoading(false);
    }
  };

  return (
    <div className="min-h-screen relative bg-black text-white selection:bg-blue-600 selection:text-white">
      {/* Background Light Fall graphics matching reference screenshot */}
      <LightFallBackground />

      {/* Header */}
      <header className="relative z-10 flex justify-between items-center px-[clamp(20px,4vw,48px)] py-4 border-b border-[rgba(255,255,255,0.1)]">
        <div className="flex items-center gap-6">
          <img
            src={svgToUri(NORECT_LOGO)}
            alt="Doppelgambit"
            className="h-9 w-auto filter drop-shadow-[0_0_10px_rgba(47,123,255,0.7)] cursor-pointer"
            onClick={() => navigate('/home')}
          />
          <span className="font-mono text-xs text-[#8b8b8b] hidden sm:inline-block">
            // My Cloned Bots
          </span>
        </div>

        <div className="flex items-center gap-4">
          {user && (
            <span className="font-mono text-xs text-[#8b8b8b] hidden md:inline">
              {user.email}
            </span>
          )}
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn main sm"
          >
            + New Bot
          </button>
          <button
            onClick={() => {
              logout();
              navigate('/login');
            }}
            className="btn ghost sm"
          >
            Sign out
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="relative z-10 max-w-6xl mx-auto px-6 py-10">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
          <div>
            <h1 className="font-head font-bold text-3xl tracking-tight">Active Bots</h1>
            <p className="text-[#8b8b8b] text-sm mt-1 font-mono">
              Manage your trained neural style clones and challenge them anytime.
            </p>
          </div>
          <div className="font-mono text-xs text-[#8b8b8b] bg-[#0c0c0c] border border-[rgba(255,255,255,0.16)] px-3 py-1.5 rounded-md self-start sm:self-auto">
            Slots: <span className="text-white font-semibold">{bots.filter((b) => !b.archived).length} / 10</span>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center min-h-[300px]">
            <div className="font-mono text-xs text-[#8b8b8b] flex items-center gap-3">
              <span className="w-2.5 h-2.5 rounded-full bg-white animate-ping" />
              Loading your roster…
            </div>
          </div>
        ) : bots.length === 0 ? (
          <div className="bg-[#080808] border border-[rgba(255,255,255,0.16)] rounded-xl p-12 text-center max-w-lg mx-auto my-12">
            <div className="w-14 h-14 mx-auto mb-4 rounded-full bg-[#121212] border border-[rgba(255,255,255,0.16)] flex items-center justify-center font-head text-2xl text-[#2F7BFF]">
              ♟
            </div>
            <h2 className="font-head font-bold text-xl mb-2">No bots connected yet</h2>
            <p className="text-sm text-[#8b8b8b] mb-6 font-mono leading-relaxed">
              Clone any public Lichess player or your own handle to practice against their exact openings and behavioral traits.
            </p>
            <button
              onClick={() => setShowCreateModal(true)}
              className="btn main"
            >
              Clone Your First Bot
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {bots.map((bot) => (
              <div
                key={bot.username}
                className="bg-[#080808] border border-[rgba(255,255,255,0.16)] hover:border-[rgba(255,255,255,0.35)] rounded-xl p-6 transition-all duration-300 flex flex-col justify-between shadow-lg relative overflow-hidden group"
              >
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[#2F7BFF] to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                <div>
                  {/* Top info row */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-lg bg-[#141414] border border-[rgba(255,255,255,0.12)] flex items-center justify-center text-xl text-white">
                        ♞
                      </div>
                      <div>
                        <h2 className="font-head font-bold text-lg text-white group-hover:text-[#2F7BFF] transition-colors">
                          {bot.username}
                        </h2>
                        <span className="font-mono text-[11px] text-[#8b8b8b]">
                          Tier: {bot.tier || 'free'}
                        </span>
                      </div>
                    </div>

                    <span
                      className={`font-mono text-[11px] px-2.5 py-1 rounded-full border ${
                        bot.status === 'ready'
                          ? 'bg-emerald-950/40 text-emerald-400 border-emerald-800/60'
                          : bot.status === 'failed' || bot.status === 'error'
                          ? 'bg-rose-950/40 text-rose-400 border-rose-800/60'
                          : 'bg-amber-950/40 text-amber-400 border-amber-800/60 animate-pulse'
                      }`}
                    >
                      {bot.status}
                    </span>
                  </div>

                  {/* Pipeline Progress Display */}
                  {(bot.status === 'building' || bot.status === 'queued') && (
                    <div className="my-3 p-3 rounded-lg bg-[rgba(255,255,255,0.03)] border border-[rgba(255,255,255,0.08)]">
                      <div className="flex justify-between items-center text-[11px] font-mono mb-1.5">
                        <span className="text-[#8b8b8b] truncate max-w-[190px]" title={bot.job_step || 'Processing'}>
                          {bot.job_step || (bot.status === 'queued' ? 'Queued in pipeline' : 'Building model...')}
                        </span>
                        <span className="text-[#2F7BFF] font-semibold">
                          {Math.round((bot.job_progress ?? 0) * 100)}%
                        </span>
                      </div>
                      <div className="w-full bg-[rgba(255,255,255,0.08)] h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-[#2F7BFF] to-[#38bdf8] h-full rounded-full transition-all duration-500 ease-out"
                          style={{ width: `${Math.max(5, Math.min(100, Math.round((bot.job_progress ?? 0) * 100)))}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {bot.status === 'failed' && (
                    <div className="my-3 p-2.5 rounded-lg bg-rose-950/20 border border-rose-800/40 text-[11px] font-mono text-rose-300">
                      <div className="font-semibold text-rose-400 mb-0.5">Build Failed</div>
                      <p className="truncate text-[10px] text-rose-400/80">{bot.error || 'Pipeline encountered an error'}</p>
                    </div>
                  )}

                  {/* Bot metrics */}
                  <div className="grid grid-cols-2 gap-2 my-4 pt-3 border-t border-[rgba(255,255,255,0.08)] font-mono text-xs text-[#8b8b8b]">
                    <div>
                      <span className="block text-[10px] uppercase text-[#555]">Version</span>
                      <span className="text-white">v{bot.active_version ?? 1}</span>
                    </div>
                    <div>
                      <span className="block text-[10px] uppercase text-[#555]">Games Sampled</span>
                      <span className="text-white">{bot.games_used ?? 0}</span>
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="grid grid-cols-2 gap-3 mt-4 pt-4 border-t border-[rgba(255,255,255,0.08)]">
                  <button
                    onClick={() => bot.status === 'ready' && navigate(`/play/${bot.username}`)}
                    disabled={bot.status !== 'ready'}
                    className={`btn main sm ${
                      bot.status !== 'ready'
                        ? 'opacity-50 cursor-not-allowed pointer-events-none'
                        : ''
                    }`}
                  >
                    {bot.status === 'building'
                      ? 'Building...'
                      : bot.status === 'queued'
                      ? 'Queued'
                      : bot.status === 'failed'
                      ? 'Failed'
                      : 'Play'}
                  </button>
                  <button
                    onClick={() => navigate(`/dashboard/${bot.username}`)}
                    className="btn ghost sm"
                  >
                    Dashboard
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* Create New Bot Modal */}
      {showCreateModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4"
          onClick={() => setShowCreateModal(false)}
        >
          <div
            className="mc bg-[#080808] border border-[rgba(255,255,255,0.2)] rounded-xl w-full max-w-[450px] p-6 shadow-2xl relative"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-4">
              <h2 className="font-head font-bold text-2xl tracking-tight">Clone New Bot</h2>
              <button
                onClick={() => setShowCreateModal(false)}
                className="text-[#8b8b8b] hover:text-white text-xl font-mono"
              >
                ×
              </button>
            </div>

            <p className="text-xs text-[#8b8b8b] font-mono leading-relaxed mb-6">
              Enter any valid Lichess username. We will verify their profile and initialize their opening repertoire model.
            </p>

            <form onSubmit={handleCreateBot}>
              <div className="mb-4">
                <label className="block font-mono text-xs text-[#8b8b8b] mb-2 uppercase">
                  Lichess Username
                </label>
                <input
                  type="text"
                  autoFocus
                  value={newBotUsername}
                  onChange={(e) => setNewBotUsername(e.target.value)}
                  placeholder="e.g. MagnusCarlsen, hikaru, or your handle"
                  className="w-full bg-[#0c0c0c] border border-[rgba(255,255,255,0.16)] rounded px-3.5 py-3 text-sm text-white focus:border-white focus:shadow-[0_0_0_3px_rgba(255,255,255,0.12)] focus:outline-none transition-all"
                />
              </div>

              {modalError && (
                <div className="font-mono text-xs text-red-400 bg-red-950/20 border border-red-900/40 p-2.5 rounded mb-4">
                  {modalError}
                </div>
              )}

              <div className="flex gap-3 mt-6">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="btn ghost full"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={modalLoading}
                  className="btn main full"
                >
                  {modalLoading ? 'Creating…' : 'Clone Bot'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default Home;
