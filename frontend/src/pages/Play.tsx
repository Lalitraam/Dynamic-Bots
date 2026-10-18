import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Chessboard } from 'react-chessboard';
import { Chess } from 'chess.js';
import apiClient from '../api/client';
import { NORECT_LOGO, svgToUri } from '../utils/logo';

interface MoveRecord {
  uci: string;
  san?: string;
  fen?: string;
  from?: string;
  to?: string;
  captured?: string;
}

interface Clocks {
  white: number;
  black: number;
}

interface SnapshotMsg {
  t: 'snapshot';
  game_id: string;
  fen: string;
  status: string;
  result?: string | null;
  termination?: string | null;
  human_color: string;
  bot_username: string;
  moves: MoveRecord[];
  clocks: Clocks;
  turn: 'white' | 'black';
  ply: number;
}

const BD_THEMES = {
  brown: { lt: '#f0d9b5', dk: '#b58863', name: 'Classic brown' },
  walnut: { lt: '#ead7b0', dk: '#8a5a3b', name: 'Walnut' },
  marble: { lt: '#ececef', dk: '#8f9eb0', name: 'Marble' },
};

const TIME_CONTROLS = [
  { label: '1+0', sub: 'Bullet', base_s: 60, inc_s: 0 },
  { label: '3+0', sub: 'Blitz', base_s: 180, inc_s: 0 },
  { label: '3+2', sub: 'Blitz', base_s: 180, inc_s: 2 },
  { label: '5+0', sub: 'Blitz', base_s: 300, inc_s: 0 },
  { label: '10+0', sub: 'Rapid', base_s: 600, inc_s: 0 },
  { label: '15+10', sub: 'Rapid', base_s: 900, inc_s: 10 },
  { label: '0+0', sub: 'Casual', base_s: 600, inc_s: 0 },
];

export const Play: React.FC = () => {
  const { username = 'bot' } = useParams<{ username: string }>();
  const navigate = useNavigate();

  // Settings & Theme
  const [boardTheme, setBoardTheme] = useState<keyof typeof BD_THEMES>('brown');
  const [frameAnim, setFrameAnim] = useState<boolean>(true);
  const [showSettings, setShowSettings] = useState<boolean>(false);

  // New Game Modal state
  const [showModal, setShowModal] = useState<boolean>(true);
  const [selectedTc, setSelectedTc] = useState(TIME_CONTROLS[4]); // 10+0
  const [selectedStrength, setSelectedStrength] = useState<number>(2); // 1: Rookie, 2: Club, 3: Expert
  const [selectedColor, setSelectedColor] = useState<'w' | 'r' | 'b'>('w');
  const [isStartingGame, setIsStartingGame] = useState<boolean>(false);

  // Active Game State
  const [gameId, setGameId] = useState<string | null>(null);
  const [humanColor, setHumanColor] = useState<'white' | 'black'>('white');
  const [flipped, setFlipped] = useState<boolean>(false);
  const [fen, setFen] = useState<string>('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const [moves, setMoves] = useState<MoveRecord[]>([]);
  const [clocks, setClocks] = useState<Clocks>({ white: 600000, black: 600000 });
  const [turn, setTurn] = useState<'white' | 'black'>('white');
  const [ply, setPly] = useState<number>(0);
  const [gameStatus, setGameStatus] = useState<string>('in_progress');
  const [gameResult, setGameResult] = useState<string | null>(null);
  const [gameTermination, setGameTermination] = useState<string | null>(null);
  const [gameOverOverlay, setGameOverOverlay] = useState<boolean>(false);
  const [statusText, setStatusText] = useState<string>('Ready');

  // Move review index
  const [viewIndex, setViewIndex] = useState<number | null>(null);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const movesContainerRef = useRef<HTMLDivElement | null>(null);
  const boardWrapRef = useRef<HTMLDivElement | null>(null);
  const lastMovesCountRef = useRef<number>(0);

  // Clock countdown interval
  useEffect(() => {
    if (showModal || gameOverOverlay || gameStatus !== 'in_progress') return;
    if (moves.length < 2 && selectedTc.base_s > 0) {
      // Clocks only tick after first 2 plies
      return;
    }

    const interval = setInterval(() => {
      setClocks((prev) => {
        const next = { ...prev };
        if (turn === 'white') {
          next.white = Math.max(0, next.white - 100);
        } else {
          next.black = Math.max(0, next.black - 100);
        }
        return next;
      });
    }, 100);

    return () => clearInterval(interval);
  }, [turn, gameStatus, moves.length, showModal, gameOverOverlay, selectedTc.base_s]);

  // Particle Burst on capture
  const triggerBurst = useCallback(() => {
    const bw = boardWrapRef.current;
    if (!bw) return;
    const rect = bw.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    for (let i = 0; i < 24; i++) {
      const p = document.createElement('i');
      p.className = 'bp';
      p.style.left = `${centerX}px`;
      p.style.top = `${centerY}px`;
      bw.appendChild(p);

      const angle = Math.random() * 6.28;
      const dist = 40 + Math.random() * 80;
      const targetX = Math.cos(angle) * dist;
      const targetY = Math.sin(angle) * dist;

      p.animate(
        [
          { transform: 'translate(0, 0)', opacity: 1 },
          { transform: `translate(${targetX}px, ${targetY}px) scale(0.2)`, opacity: 0 },
        ],
        {
          duration: 600 + Math.random() * 300,
          easing: 'cubic-bezier(0.2, 0.8, 0.2, 1)',
        }
      ).onfinish = () => p.remove();
    }
  }, []);

  // Connect WebSocket when gameId and secret are established
  const connectWs = useCallback(
    (gId: string, secret: string) => {
      if (wsRef.current) {
        wsRef.current.close();
      }

      const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const wsProto = apiBase.startsWith('https') ? 'wss:' : 'ws:';
      const wsHost = apiBase.replace(/^https?:\/\//, '');
      const wsUrl = `${wsProto}//${wsHost}/ws/${gId}`;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ t: 'hello', secret }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.t === 'snapshot') {
            const snap = data as SnapshotMsg;
            setFen(snap.fen);
            setClocks(snap.clocks);
            setTurn(snap.turn);
            setPly(snap.ply);
            setGameStatus(snap.status);
            setMoves(snap.moves || []);

            // Check if capture occurred
            if (snap.moves && snap.moves.length > lastMovesCountRef.current) {
              const lastMove = snap.moves[snap.moves.length - 1];
              if (lastMove.captured) {
                triggerBurst();
              }
            }
            lastMovesCountRef.current = (snap.moves || []).length;

            if (snap.status === 'completed' || snap.status === 'resigned' || snap.result) {
              setGameResult(snap.result || null);
              setGameTermination(snap.termination || snap.status);
              setGameOverOverlay(true);
            }

            // Scroll moves to bottom
            setTimeout(() => {
              if (movesContainerRef.current) {
                movesContainerRef.current.scrollTop = movesContainerRef.current.scrollHeight;
              }
            }, 50);
          } else if (data.t === 'end') {
            setGameResult(data.result);
            setGameTermination(data.termination);
            setGameOverOverlay(true);
          } else if (data.t === 'error') {
            console.error('WS Error:', data.error);
            setStatusText(data.error);
          }
        } catch (e) {
          console.error('Error parsing WS message:', e);
        }
      };

      ws.onerror = (err) => {
        console.error('WebSocket encountered error:', err);
      };

      ws.onclose = () => {
        console.log('WebSocket closed');
      };
    },
    [triggerBurst]
  );

  // Clean up WS on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Start new game via API
  const handleStartGame = async () => {
    setIsStartingGame(true);
    let chosenColor = selectedColor;
    if (chosenColor === 'r') {
      chosenColor = Math.random() < 0.5 ? 'w' : 'b';
    }
    const colorStr = chosenColor === 'w' ? 'white' : 'black';

    try {
      const res = await apiClient.post('/api/games', {
        username,
        human_color: colorStr,
        base_s: selectedTc.base_s,
        inc_s: selectedTc.inc_s,
      });

      const { game_id, secret, human_color: assignedColor, fen: startFen } = res.data;
      setGameId(game_id);
      setHumanColor(assignedColor === 'black' ? 'black' : 'white');
      setFlipped(assignedColor === 'black');
      setFen(startFen);
      setClocks({
        white: selectedTc.base_s * 1000,
        black: selectedTc.base_s * 1000,
      });
      setMoves([]);
      lastMovesCountRef.current = 0;
      setPly(0);
      setGameStatus('in_progress');
      setGameResult(null);
      setGameTermination(null);
      setGameOverOverlay(false);
      setShowModal(false);
      setViewIndex(null);

      // Connect to websocket
      connectWs(game_id, secret);
    } catch (err: any) {
      console.error('Failed to start game:', err);
      alert(err?.response?.data?.detail || 'Failed to start game.');
    } finally {
      setIsStartingGame(false);
    }
  };

  // Move submission handler from react-chessboard
  const handlePieceDrop = ({
    sourceSquare,
    targetSquare,
  }: {
    sourceSquare: string;
    targetSquare: string | null;
  }): boolean => {
    if (!targetSquare) return false;
    if (gameStatus !== 'in_progress' || gameOverOverlay) return false;
    if (turn !== humanColor) return false;

    // Validate legality using chess.js locally
    try {
      const tempGame = new Chess(fen);
      const possibleMoves = tempGame.moves({ verbose: true });
      const foundMove = possibleMoves.find(
        (m) => m.from === sourceSquare && m.to === targetSquare
      );

      if (!foundMove) {
        return false;
      }

      // Format UCI (e.g., e2e4 or e7e8q)
      const uci = `${sourceSquare}${targetSquare}${foundMove.promotion || ''}`;

      // Send authoritatively via WebSocket
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            t: 'move',
            uci,
            n: ply,
          })
        );
      }

      // Do not update board locally; wait for server snapshot!
      return false;
    } catch (err) {
      console.error('Error validating move:', err);
      return false;
    }
  };

  const handleResign = () => {
    if (gameStatus !== 'in_progress') return;
    if (window.confirm('Are you sure you want to resign?')) {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ t: 'resign' }));
      }
    }
  };

  const formatClock = (ms: number) => {
    if (ms <= 0) return '0:00';
    const s = ms / 1000;
    if (s < 10) return '0:0' + s.toFixed(1);
    const mins = Math.floor(s / 60);
    const secs = Math.floor(s % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Move history pairs
  const movePairs: { white?: MoveRecord; black?: MoveRecord }[] = [];
  for (let i = 0; i < moves.length; i += 2) {
    movePairs.push({
      white: moves[i],
      black: moves[i + 1],
    });
  }

  // Display FEN (either live or historical review)
  const displayFen = viewIndex !== null && moves[viewIndex]?.fen ? moves[viewIndex].fen! : fen;

  const currentTheme = BD_THEMES[boardTheme];
  const isHumanTurn = turn === humanColor;

  return (
    <div className="min-h-screen relative bg-black text-white selection:bg-blue-600 selection:text-white flex flex-col justify-between">
      {/* Background violet window light + king shadow */}
      <div className="fx" aria-hidden="true">
        <div className="pool" />
        <div className="beam">
          <i />
          <i className="b" />
        </div>
        <svg className="ks" viewBox="0 0 100 150" preserveAspectRatio="xMidYMax meet">
          <path d="M47 2h6v7h9v6h-9v11h-6V15h-9V9h9zM30 52Q26 28 50 26Q74 28 70 52zM26 52h48v8H26zM34 60Q38 90 26 124h48Q62 90 66 60zM20 124h60v10H20zM14 134h72v12H14z" />
        </svg>
      </div>

      {/* Header */}
      <header className="relative z-20 flex justify-between items-center px-[clamp(16px,4vw,48px)] py-4 border-b border-[rgba(255,255,255,0.1)]">
        <div className="flex items-center gap-4">
          <img
            src={svgToUri(NORECT_LOGO)}
            alt="Doppelgambit"
            className="h-9 w-auto filter drop-shadow-[0_0_10px_rgba(47,123,255,0.7)] cursor-pointer"
            onClick={() => navigate('/home')}
          />
          <span className="font-mono text-xs text-[#8b8b8b] hidden sm:inline">
            // vs {username} {gameId ? `· Session #${gameId.slice(0, 6)}` : ''}
          </span>
        </div>

        <div className="flex gap-3 relative">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="btn ghost sm"
          >
            Board &amp; pieces
          </button>
          <button
            onClick={() => setShowModal(true)}
            className="btn main sm"
          >
            New game
          </button>

          {/* Settings Popover */}
          {showSettings && (
            <div className="pop absolute top-12 right-0 z-50 w-72 bg-[#080808] border border-[rgba(255,255,255,0.16)] rounded-xl p-4 shadow-2xl">
              <div className="text-[11px] font-mono text-[#8b8b8b] uppercase tracking-wider mb-2">
                Board Theme
              </div>
              <div className="grid grid-cols-3 gap-2 mb-4">
                {(Object.keys(BD_THEMES) as Array<keyof typeof BD_THEMES>).map((key) => (
                  <button
                    key={key}
                    onClick={() => setBoardTheme(key)}
                    className={`opt p-2 rounded border font-mono text-xs text-center flex flex-col items-center gap-1.5 transition-all ${
                      boardTheme === key
                        ? 'border-white text-white shadow-[0_0_12px_rgba(255,255,255,0.2)]'
                        : 'border-[rgba(255,255,255,0.1)] text-[#8b8b8b] hover:border-[#666]'
                    }`}
                  >
                    <div className="flex w-7 h-4 rounded overflow-hidden">
                      <div className="w-1/2 h-full" style={{ background: BD_THEMES[key].lt }} />
                      <div className="w-1/2 h-full" style={{ background: BD_THEMES[key].dk }} />
                    </div>
                    <span className="text-[10px] truncate max-w-full">{BD_THEMES[key].name}</span>
                  </button>
                ))}
              </div>

              <div className="text-[11px] font-mono text-[#8b8b8b] uppercase tracking-wider mb-2">
                Visual Effects
              </div>
              <button
                onClick={() => setFrameAnim(!frameAnim)}
                className={`tg w-full flex items-center justify-between font-mono text-xs py-2 px-1 text-left ${
                  frameAnim ? 'on text-white' : 'text-[#8b8b8b]'
                }`}
              >
                <span>Frame marching light</span>
                <i />
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Main Play Area */}
      <main className="relative z-10 flex flex-wrap gap-[clamp(16px,3vw,40px)] justify-center items-start px-[clamp(16px,4vw,48px)] py-6 flex-1">
        {/* Animated Chessboard Wrapper */}
        <div
          className={`frame my-auto ${!frameAnim ? 'noanim' : ''}`}
          style={{ width: 'clamp(280px, min(calc(100vh - 220px), calc(100vw - 440px)), 580px)' }}
        >
          <div ref={boardWrapRef} className="bw relative shadow-2xl">
            <Chessboard
              options={{
                position: displayFen,
                boardOrientation: flipped ? 'black' : 'white',
                onPieceDrop: handlePieceDrop,
                boardStyle: {
                  borderRadius: '0px',
                  boxShadow: '0 10px 40px rgba(0,0,0,.6)',
                },
                darkSquareStyle: { backgroundColor: currentTheme.dk },
                lightSquareStyle: { backgroundColor: currentTheme.lt },
                canDragPiece: ({ piece }) => {
                  if (gameStatus !== 'in_progress' || gameOverOverlay) return false;
                  if (!isHumanTurn) return false;
                  const pt = piece.pieceType || '';
                  return humanColor === 'white' ? pt.startsWith('w') : pt.startsWith('b');
                },
              }}
            />

            {/* Game Over Overlay */}
            {gameOverOverlay && (
              <div
                id="over"
                className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm text-center p-6"
              >
                <h2 className="font-head font-bold text-3xl md:text-4xl tracking-tight mb-2">
                  {gameResult === '1-0'
                    ? humanColor === 'white'
                      ? 'Victory!'
                      : 'Defeat'
                    : gameResult === '0-1'
                    ? humanColor === 'black'
                      ? 'Victory!'
                      : 'Defeat'
                    : 'Game Drawn'}
                </h2>
                <p className="text-sm font-mono text-[#8b8b8b] mb-6">
                  {gameTermination || 'Match concluded'} · {gameResult || '½-½'}
                </p>
                <div className="flex gap-3 flex-wrap justify-center">
                  <button onClick={handleStartGame} className="btn main sm">
                    Rematch
                  </button>
                  <button onClick={() => setShowModal(true)} className="btn blue sm">
                    New Game
                  </button>
                  <button onClick={() => setGameOverOverlay(false)} className="btn ghost sm">
                    Review Board
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Side Panel: Opponent profile, Clocks, Moves, Actions, Human profile */}
        <aside className="side w-full max-w-[340px] flex flex-col gap-3 min-h-[500px]">
          {/* Top Profile (Bot / Opponent) */}
          <div className="pr flex items-center gap-3 p-1">
            <div
              className={`av w-9 h-9 rounded-lg flex items-center justify-center font-bold text-lg border border-[rgba(255,255,255,0.15)] ${
                humanColor === 'white' ? 'bg-[#151515] text-[#f4f4f4]' : 'bg-[#f4f4f4] text-[#111]'
              }`}
            >
              ♟
            </div>
            <div className="flex-1 min-w-0">
              <b className="font-head text-sm block truncate">{username} (Clone)</b>
              <small className="font-mono text-[11px] text-[#8b8b8b] block">
                {selectedStrength === 1 ? 'Rookie' : selectedStrength === 2 ? 'Club' : 'Expert'} Engine
              </small>
            </div>
            <div
              className={`clk font-mono font-semibold text-2xl px-3 py-1.5 rounded min-w-[96px] text-center transition-colors ${
                turn !== humanColor && gameStatus === 'in_progress'
                  ? 'bg-white text-black shadow-[0_0_15px_rgba(255,255,255,0.4)]'
                  : 'bg-white/5 text-[#8b8b8b]'
              } ${
                turn !== humanColor &&
                (humanColor === 'white' ? clocks.black : clocks.white) < 20000 &&
                gameStatus === 'in_progress'
                  ? '!bg-[#e5484d] !text-white'
                  : ''
              }`}
            >
              {selectedTc.base_s ? formatClock(humanColor === 'white' ? clocks.black : clocks.white) : '∞'}
            </div>
          </div>

          {/* Move History Panel */}
          <div className="mvw flex-1 min-h-[220px] max-h-[300px] flex flex-col border border-[rgba(255,255,255,0.16)] rounded-lg overflow-hidden bg-white/[0.025]">
            <div
              ref={movesContainerRef}
              id="mv"
              className="flex-1 overflow-y-auto grid grid-cols-[36px_1fr_1fr] font-mono text-xs content-start"
            >
              {movePairs.map((pair, index) => (
                <React.Fragment key={index}>
                  <b className="bg-white/5 text-[#8b8b8b] font-normal py-1.5 text-center">
                    {index + 1}
                  </b>
                  <span
                    onClick={() => setViewIndex(index * 2)}
                    className={`py-1.5 px-2.5 cursor-pointer hover:bg-white/10 transition-colors ${
                      viewIndex === index * 2 || (viewIndex === null && moves.length - 1 === index * 2)
                        ? 'bg-white/20 text-white font-semibold'
                        : 'text-[#ccc]'
                    }`}
                  >
                    {pair.white?.san || pair.white?.uci || ''}
                  </span>
                  <span
                    onClick={() => pair.black && setViewIndex(index * 2 + 1)}
                    className={`py-1.5 px-2.5 cursor-pointer hover:bg-white/10 transition-colors ${
                      viewIndex === index * 2 + 1 ||
                      (viewIndex === null && moves.length - 1 === index * 2 + 1)
                        ? 'bg-white/20 text-white font-semibold'
                        : 'text-[#ccc]'
                    }`}
                  >
                    {pair.black?.san || pair.black?.uci || ''}
                  </span>
                </React.Fragment>
              ))}

              {gameResult && (
                <div className="col-span-3 text-center py-2.5 font-mono text-sm font-semibold border-t border-[rgba(255,255,255,0.16)] text-[#2F7BFF]">
                  {gameResult}
                </div>
              )}
            </div>

            {/* Move Nav Buttons */}
            <div className="nav grid grid-cols-4 border-t border-[rgba(255,255,255,0.16)] text-xs">
              <button
                onClick={() => setViewIndex(0)}
                className="p-2 text-[#8b8b8b] hover:bg-white/10 hover:text-white transition-colors"
                title="Start"
              >
                ⏮
              </button>
              <button
                onClick={() => setViewIndex((prev) => Math.max(0, (prev ?? moves.length) - 1))}
                className="p-2 text-[#8b8b8b] hover:bg-white/10 hover:text-white transition-colors"
                title="Back"
              >
                ◀
              </button>
              <button
                onClick={() =>
                  setViewIndex((prev) =>
                    prev === null || prev >= moves.length - 1 ? null : prev + 1
                  )
                }
                className="p-2 text-[#8b8b8b] hover:bg-white/10 hover:text-white transition-colors"
                title="Forward"
              >
                ▶
              </button>
              <button
                onClick={() => setViewIndex(null)}
                className="p-2 text-[#8b8b8b] hover:bg-white/10 hover:text-white transition-colors"
                title="Live"
              >
                ⏭
              </button>
            </div>
          </div>

          {/* Engine Status Line */}
          <div className="stat font-mono text-xs flex items-center gap-2.5 px-1 min-h-[22px] text-[#8b8b8b]">
            <span
              className={`led w-2 h-2 rounded-full ${
                turn !== humanColor && gameStatus === 'in_progress'
                  ? 'bg-[#2F7BFF] shadow-[0_0_10px_#2F7BFF] animate-pulse'
                  : 'bg-white shadow-[0_0_6px_#fff]'
              }`}
            />
            <span className="truncate">
              {viewIndex !== null
                ? 'Reviewing — Click ⏭ for live board'
                : turn !== humanColor && gameStatus === 'in_progress'
                ? `${username} is calculating…`
                : gameStatus === 'in_progress'
                ? 'Your turn'
                : statusText}
            </span>
          </div>

          {/* Action Buttons */}
          <div className="acts grid grid-cols-4 gap-2">
            <button
              onClick={() => setFlipped(!flipped)}
              className="btn ghost sm text-xs !p-2"
              title="Flip board orientation"
            >
              ⇅ Flip
            </button>
            <button
              onClick={() => setViewIndex(null)}
              className="btn ghost sm text-xs !p-2"
              title="Jump to current position"
            >
              ↺ Reset
            </button>
            <button
              onClick={() => {
                if (window.confirm('Offer a draw?')) {
                  alert('Doppelgambit evaluates the position and declines the draw.');
                }
              }}
              className="btn ghost sm text-xs !p-2"
              title="Offer draw"
            >
              ½ Draw
            </button>
            <button
              onClick={handleResign}
              disabled={gameStatus !== 'in_progress'}
              className="btn ghost sm text-xs !p-2 hover:!border-red-500 hover:!text-red-400"
              title="Resign game"
            >
              ⚑ Resign
            </button>
          </div>

          {/* Bottom Profile (Human / You) */}
          <div className="pr flex items-center gap-3 p-1">
            <div
              className={`av w-9 h-9 rounded-lg flex items-center justify-center font-bold text-lg border border-[rgba(255,255,255,0.15)] ${
                humanColor === 'white' ? 'bg-[#f4f4f4] text-[#111]' : 'bg-[#151515] text-[#f4f4f4]'
              }`}
            >
              ♟
            </div>
            <div className="flex-1 min-w-0">
              <b className="font-head text-sm block">You</b>
              <small className="font-mono text-[11px] text-[#8b8b8b] block">
                Playing {humanColor}
              </small>
            </div>
            <div
              className={`clk font-mono font-semibold text-2xl px-3 py-1.5 rounded min-w-[96px] text-center transition-colors ${
                turn === humanColor && gameStatus === 'in_progress'
                  ? 'bg-white text-black shadow-[0_0_15px_rgba(255,255,255,0.4)]'
                  : 'bg-white/5 text-[#8b8b8b]'
              } ${
                turn === humanColor &&
                (humanColor === 'white' ? clocks.white : clocks.black) < 20000 &&
                gameStatus === 'in_progress'
                  ? '!bg-[#e5484d] !text-white'
                  : ''
              }`}
            >
              {selectedTc.base_s ? formatClock(humanColor === 'white' ? clocks.white : clocks.black) : '∞'}
            </div>
          </div>
        </aside>
      </main>

      {/* Game Creation Modal */}
      {showModal && (
        <div className="modal fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="mc bg-[#080808] border border-[rgba(255,255,255,0.2)] rounded-xl w-full max-w-[450px] p-6 shadow-2xl relative">
            <h2 className="font-head font-bold text-2xl tracking-tight mb-4">
              Challenge {username}
            </h2>

            {/* Time control */}
            <div className="font-mono text-[11px] text-[#8b8b8b] uppercase tracking-wider mb-2">
              Time Control
            </div>
            <div className="grid grid-cols-3 gap-2 mb-4">
              {TIME_CONTROLS.map((tc) => (
                <button
                  key={tc.label}
                  type="button"
                  onClick={() => setSelectedTc(tc)}
                  className={`opt p-2.5 rounded border font-mono text-center transition-all ${
                    selectedTc.label === tc.label
                      ? 'border-white text-white shadow-[0_0_12px_rgba(255,255,255,0.2)]'
                      : 'border-[rgba(255,255,255,0.16)] text-[#8b8b8b] hover:border-[#666]'
                  }`}
                >
                  <b className="block font-head text-lg text-white font-bold">{tc.label}</b>
                  <small className="text-[10px] text-[#8b8b8b] block">{tc.sub}</small>
                </button>
              ))}
            </div>

            {/* Opponent strength */}
            <div className="font-mono text-[11px] text-[#8b8b8b] uppercase tracking-wider mb-2">
              Opponent Strength
            </div>
            <div className="grid grid-cols-3 gap-2 mb-4">
              {[
                { lvl: 1, label: 'Rookie' },
                { lvl: 2, label: 'Club' },
                { lvl: 3, label: 'Expert' },
              ].map((s) => (
                <button
                  key={s.lvl}
                  type="button"
                  onClick={() => setSelectedStrength(s.lvl)}
                  className={`opt py-2 px-3 rounded border font-mono text-xs text-center transition-all ${
                    selectedStrength === s.lvl
                      ? 'border-white text-white shadow-[0_0_12px_rgba(255,255,255,0.2)]'
                      : 'border-[rgba(255,255,255,0.16)] text-[#8b8b8b] hover:border-[#666]'
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>

            {/* Play as */}
            <div className="font-mono text-[11px] text-[#8b8b8b] uppercase tracking-wider mb-2">
              Play As
            </div>
            <div className="grid grid-cols-3 gap-2 mb-6">
              {[
                { code: 'w', label: 'White' },
                { code: 'r', label: 'Random' },
                { code: 'b', label: 'Black' },
              ].map((c) => (
                <button
                  key={c.code}
                  type="button"
                  onClick={() => setSelectedColor(c.code as any)}
                  className={`opt py-2 px-3 rounded border font-mono text-xs text-center transition-all ${
                    selectedColor === c.code
                      ? 'border-white text-white shadow-[0_0_12px_rgba(255,255,255,0.2)]'
                      : 'border-[rgba(255,255,255,0.16)] text-[#8b8b8b] hover:border-[#666]'
                  }`}
                >
                  {c.label}
                </button>
              ))}
            </div>

            <button
              onClick={handleStartGame}
              disabled={isStartingGame}
              className="btn main full py-3"
            >
              {isStartingGame ? 'Setting up board…' : 'Play'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Play;
