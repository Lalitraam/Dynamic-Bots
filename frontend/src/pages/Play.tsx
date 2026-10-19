import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Chessboard as BaseChessboard } from 'react-chessboard';
import { Chess } from 'chess.js';
import apiClient from '../api/client';
import { NORECT_LOGO, svgToUri } from '../utils/logo';
import { LightFallBackground } from '../components/LightFallBackground';

const Chessboard = BaseChessboard as any;

export const pieceMap: Record<string, string> = {
  p: 'Pawn',
  n: 'Knight',
  b: 'Bishop',
  r: 'Rook',
  q: 'Queen',
  k: 'King',
};

export const formatMove = (game: any, move: any): string => {
  if (!move) return '';
  const from = move.from || (move.uci && move.uci.length >= 4 ? move.uci.slice(0, 2) : undefined);
  const to = move.to || (move.uci && move.uci.length >= 4 ? move.uci.slice(2, 4) : '');
  const piece = game && from ? game.get(from) : null;
  if (!piece) return move.san || 'Move';
  return `${pieceMap[piece.type] || piece.type} ${to || move.to}`;
};

export const formatTerminationReason = (reason: string | null | undefined): string => {
  if (!reason) return 'Concluded';
  const r = reason.toLowerCase().trim();
  if (r === 'time forfeit' || r === 'timeout') return 'Time forfeit';
  if (r === 'checkmate') return 'Checkmate';
  if (r === 'resignation') return 'Resignation';
  if (r === 'stalemate') return 'Stalemate';
  if (r === 'insufficient material') return 'Insufficient material';
  if (r === 'threefold repetition') return 'Threefold repetition';
  if (r === '50-move rule') return '50-move rule';
  return reason.charAt(0).toUpperCase() + reason.slice(1);
};


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
  const [selectedColor, setSelectedColor] = useState<'w' | 'r' | 'b'>('w');
  const [isStartingGame, setIsStartingGame] = useState<boolean>(false);

  // Active Game State
  const [gameId, setGameId] = useState<string | null>(null);
  const [humanColor, setHumanColor] = useState<'white' | 'black'>('white');
  const [flipped, setFlipped] = useState<boolean>(false);
  const [fen, setFen] = useState<string>('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const [moves, setMoves] = useState<MoveRecord[]>([]);
  const [clocks, setClocks] = useState<Clocks>({ white: 600000, black: 600000 });
  const [ts, setTs] = useState<number>(() => performance.now());
  const turn: 'white' | 'black' = fen.split(' ')[1] === 'w' ? 'white' : 'black';
  const [, setPly] = useState<number>(0);
  const [gameStatus, setGameStatus] = useState<string>('in_progress');
  const [gameResult, setGameResult] = useState<string | null>(null);
  const [gameTermination, setGameTermination] = useState<string | null>(null);
  const [gameOverOverlay, setGameOverOverlay] = useState<boolean>(false);
  const [statusText, setStatusText] = useState<string>('Ready');
  const [isWaitingForBot, setIsWaitingForBot] = useState<boolean>(false);

  // Move review index
  const [viewIndex, setViewIndex] = useState<number | null>(null);

  // Legal move highlight state
  const [selectedSquare, setSelectedSquare] = useState<string | null>(null);
  const [optionSquares, setOptionSquares] = useState<Record<string, React.CSSProperties>>({});

  // Premove state (single-move queue like Lichess)
  const [premove, setPremove] = useState<{ from: string; to: string; promotion?: string } | null>(null);
  const premoveRef = useRef<{ from: string; to: string; promotion?: string } | null>(null);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const movesContainerRef = useRef<HTMLDivElement | null>(null);
  const boardWrapRef = useRef<HTMLDivElement | null>(null);
  const lastMovesCountRef = useRef<number>(0);
  const isWaitingForBotRef = useRef<boolean>(false);

  // Live state refs to guarantee callbacks never suffer from stale closures
  const fenRef = useRef(fen);
  fenRef.current = fen;
  const movesRef = useRef(moves);
  movesRef.current = moves;
  const humanColorRef = useRef(humanColor);
  humanColorRef.current = humanColor;
  const gameStatusRef = useRef(gameStatus);
  gameStatusRef.current = gameStatus;
  const gameOverOverlayRef = useRef(gameOverOverlay);
  gameOverOverlayRef.current = gameOverOverlay;
  const viewIndexRef = useRef(viewIndex);
  viewIndexRef.current = viewIndex;

  // Live clock references for drift-free real-time ticking
  const clocksRef = useRef<Clocks>({ white: 600000, black: 600000 });
  clocksRef.current = clocks;
  const turnRef = useRef<'white' | 'black'>(turn);
  turnRef.current = turn;
  const lastTickTimeRef = useRef<number>(performance.now());

  // Real-time Lichess-style clock countdown interval
  useEffect(() => {
    if (showModal || gameOverOverlay || (gameStatus !== 'in_progress' && gameStatus !== 'active')) {
      return;
    }
    if (selectedTc.base_s <= 0) {
      return; // Unlimited / casual
    }

    lastTickTimeRef.current = performance.now();

    const interval = setInterval(() => {
      const now = performance.now();
      const delta = now - lastTickTimeRef.current;
      lastTickTimeRef.current = now;

      const humanCol = humanColorRef.current;
      const botCol = humanCol === 'white' ? 'black' : 'white';

      // Active player: while waiting for bot, it is authoritatively the bot's turn
      const currentActive = isWaitingForBotRef.current
        ? botCol
        : (fenRef.current.split(' ')[1] === 'w' ? 'white' : 'black');

      // Never tick human's clock while waiting for the bot
      if (isWaitingForBotRef.current && currentActive === humanCol) {
        return;
      }

      // Move-1 rule: White does not tick before move 1
      const movesCount = movesRef.current.length;
      if (movesCount === 0 && currentActive === 'white') {
        return;
      }

      setClocks((prev) => {
        const next = { ...prev };
        if (currentActive === 'white') {
          next.white = Math.max(0, next.white - delta);
        } else {
          next.black = Math.max(0, next.black - delta);
        }
        clocksRef.current = next;
        return next;
      });
    }, 50);

    return () => clearInterval(interval);
  }, [showModal, gameOverOverlay, gameStatus, selectedTc.base_s, ts, isWaitingForBot]);

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
    (gameId: string, secret: string) => {
      if (wsRef.current) {
        wsRef.current.close();
      }

      const wsUrl = `ws://localhost:8000/ws/${gameId}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ t: 'hello', secret }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.t === 'snapshot') {
            setIsWaitingForBot(false);
            isWaitingForBotRef.current = false;
            const snap = data as SnapshotMsg;
            setFen(snap.fen);
            fenRef.current = snap.fen;

            if (snap.clocks) {
              setClocks(snap.clocks);
              clocksRef.current = { ...snap.clocks };
              lastTickTimeRef.current = performance.now();
              setTs(performance.now());
            }

            turnRef.current = snap.turn;
            setPly(snap.ply);
            setGameStatus(snap.status);
            gameStatusRef.current = snap.status;
            setMoves(snap.moves || []);
            movesRef.current = snap.moves || [];
            if (snap.human_color) {
              const hColor = snap.human_color.toLowerCase() === 'black' ? 'black' : 'white';
              setHumanColor(hColor);
              humanColorRef.current = hColor;
            }

            // Check if capture occurred
            if (snap.moves && snap.moves.length > lastMovesCountRef.current) {
              const lastMove = snap.moves[snap.moves.length - 1];
              if (lastMove.captured) {
                triggerBurst();
              }
            }
            lastMovesCountRef.current = (snap.moves || []).length;

            // Handle premove execution if queued
            const pm = premoveRef.current;
            if (pm && snap.status !== 'completed' && snap.status !== 'resigned' && !snap.result) {
              const nextGame = new Chess(snap.fen);
              const humanChessColor = (snap.human_color ? snap.human_color.toLowerCase() : humanColorRef.current) === 'black' ? 'b' : 'w';
              if (nextGame.turn() === humanChessColor) {
                let executedPremove: any = null;
                try {
                  executedPremove = nextGame.move({
                    from: pm.from,
                    to: pm.to,
                    promotion: pm.promotion || 'q',
                  });
                } catch {
                  executedPremove = null;
                }

                setPremove(null);
                premoveRef.current = null;

                if (executedPremove) {
                  // Execute premove immediately
                  const uci = `${pm.from}${pm.to}${executedPremove.promotion || ''}`;
                  const nextPly = (snap.moves || []).length + 1;
                  setFen(nextGame.fen());
                  fenRef.current = nextGame.fen();
                  turnRef.current = nextGame.fen().split(' ')[1] === 'w' ? 'white' : 'black';
                  setIsWaitingForBot(true);
                  isWaitingForBotRef.current = true;

                  const humanMoveObj: MoveRecord = {
                    uci,
                    san: executedPremove.san,
                    from: pm.from,
                    to: pm.to,
                  };
                  setMoves((prev) => [...prev, humanMoveObj]);
                  movesRef.current = [...movesRef.current, humanMoveObj];

                  if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                    wsRef.current.send(JSON.stringify({ t: 'move', uci, n: nextPly }));
                  }
                }
              } else {
                setPremove(null);
                premoveRef.current = null;
              }
            }

            if (snap.status === 'completed' || snap.status === 'resigned' || snap.result) {
              setGameStatus('completed');
              gameStatusRef.current = 'completed';
              setGameResult(snap.result || null);
              setGameTermination(snap.termination || snap.status);
              setGameOverOverlay(true);
              gameOverOverlayRef.current = true;
              setStatusText(`Game over · ${formatTerminationReason(snap.termination || snap.status)} (${snap.result || '*'})`);
            }

            // Scroll moves to bottom
            setTimeout(() => {
              if (movesContainerRef.current) {
                movesContainerRef.current.scrollTop = movesContainerRef.current.scrollHeight;
              }
            }, 50);
          } else if (data.t === 'end') {
            setIsWaitingForBot(false);
            isWaitingForBotRef.current = false;
            setGameStatus('completed');
            gameStatusRef.current = 'completed';
            setGameResult(data.result || null);
            setGameTermination(data.termination || 'completed');
            setGameOverOverlay(true);
            gameOverOverlayRef.current = true;
            setStatusText(`Game over · ${formatTerminationReason(data.termination)} (${data.result || '*'})`);
          } else if (data.t === 'error') {
            setIsWaitingForBot(false);
            isWaitingForBotRef.current = false;
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
      const initialMs = selectedTc.base_s * 1000;
      const initialClocks = {
        white: initialMs,
        black: initialMs,
      };
      setGameId(game_id);
      const isBlack = assignedColor === 'black';
      setHumanColor(isBlack ? 'black' : 'white');
      humanColorRef.current = isBlack ? 'black' : 'white';
      setFlipped(isBlack);
      setFen(startFen);
      fenRef.current = startFen;
      setClocks(initialClocks);
      clocksRef.current = { ...initialClocks };
      lastTickTimeRef.current = performance.now();
      setTs(performance.now());
      turnRef.current = 'white';
      setMoves([]);
      movesRef.current = [];
      lastMovesCountRef.current = 0;
      setPly(0);
      setGameStatus('in_progress');
      gameStatusRef.current = 'in_progress';
      setGameResult(null);
      setGameTermination(null);
      setGameOverOverlay(false);
      gameOverOverlayRef.current = false;
      const botPlaysFirst = isBlack;
      setIsWaitingForBot(botPlaysFirst);
      isWaitingForBotRef.current = botPlaysFirst;
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

  // ── Helpers ──────────────────────────────────────────────────────────────

  const clearHighlights = () => {
    setSelectedSquare(null);
    setOptionSquares({});
  };

  /** Build a customSquareStyles map showing legal move dots/rings for a given square */
  const getLegalSquares = (square: string): Record<string, React.CSSProperties> => {
    const game = new Chess(fenRef.current);
    let legalMoves: any[] = [];
    try {
      legalMoves = game.moves({ square: square as any, verbose: true }) as any[];
    } catch {
      return {};
    }
    if (!legalMoves.length) return {};

    const styles: Record<string, React.CSSProperties> = {
      [square]: {
        backgroundColor: 'rgba(255, 255, 100, 0.4)',
        boxShadow: 'inset 0 0 0 2px rgba(255, 255, 0, 0.8)',
      },
    };

    legalMoves.forEach((m) => {
      const isCapture = !!game.get(m.to);
      styles[m.to] = isCapture
        ? {
            background: 'radial-gradient(circle, rgba(229, 72, 77, 0.35) 75%, transparent 75%)',
            boxShadow: 'inset 0 0 0 3px #e5484d',
            cursor: 'pointer',
          }
        : {
            background: 'radial-gradient(circle, #2F7BFF 24%, transparent 25%)',
            cursor: 'pointer',
          };
    });
    return styles;
  };

  // ── Drag eligibility check: allow dragging human player's pieces on turn or for premove ──
  const canDragPiece = ({ piece, square }: { isSparePiece: boolean; piece: any; square: string | null }): boolean => {
    if (!gameId || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return false;
    if (viewIndexRef.current !== null) return false;
    const currentStatus = gameStatusRef.current;
    if (gameOverOverlayRef.current || (currentStatus !== 'in_progress' && currentStatus !== 'active')) {
      return false;
    }
    const humanChessColor = humanColorRef.current === 'white' ? 'w' : 'b';

    // Allow dragging player's pieces (either for immediate move on turn or queued premove)
    if (square) {
      const game = new Chess(fenRef.current);
      const p = game.get(square as any);
      return !!p && p.color === humanChessColor;
    }
    if (piece?.pieceType) {
      return piece.pieceType.startsWith(humanChessColor);
    }
    return true;
  };

  // ── Move submission — react-chessboard v5: onPieceDrop({ piece, sourceSquare, targetSquare }) ──
  const onPieceDrop = ({ sourceSquare, targetSquare }: { sourceSquare: string; targetSquare: string | null; piece: any }): boolean => {
    clearHighlights();
    if (!sourceSquare || !targetSquare) return false;
    if (!gameId || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return false;

    const game = new Chess(fenRef.current);
    const humanChessColor = humanColorRef.current === 'white' ? 'w' : 'b';
    const currentStatus = gameStatusRef.current;
    const isGameOver = gameOverOverlayRef.current || (currentStatus !== 'in_progress' && currentStatus !== 'active') || game.isGameOver();
    const isMyTurn = game.turn() === humanChessColor && !isWaitingForBotRef.current;

    console.log('[DROP]', { sourceSquare, targetSquare, turn: game.turn(), humanChessColor, isMyTurn, isGameOver });

    if (isGameOver) return false;

    // PREMOVE SUPPORT: if it is the bot's turn, queue a single premove (Lichess style)
    if (!isMyTurn) {
      if (sourceSquare !== targetSquare) {
        const pm = { from: sourceSquare, to: targetSquare, promotion: 'q' };
        setPremove(pm);
        premoveRef.current = pm;
      }
      return false;
    }

    // Normal move: clear any pending premove
    setPremove(null);
    premoveRef.current = null;

    let move: any = null;
    try {
      move = game.move({ from: sourceSquare, to: targetSquare, promotion: 'q' });
    } catch {
      return false;
    }
    if (!move) return false;

    setFen(game.fen()); // This flips the turn immediately
    setTs(performance.now()); // Reset the tick timestamp
    fenRef.current = game.fen();
    lastTickTimeRef.current = performance.now();
    turnRef.current = game.fen().split(' ')[1] === 'w' ? 'white' : 'black';
    setIsWaitingForBot(true);
    isWaitingForBotRef.current = true;

    const humanColorVal = humanColorRef.current;

    // Lichess increment rule: add increment immediately upon human move completion
    if (selectedTc.inc_s > 0) {
      setClocks((prev) => {
        const next = {
          ...prev,
          [humanColorVal]: prev[humanColorVal] + selectedTc.inc_s * 1000,
        };
        clocksRef.current = next;
        return next;
      });
    }

    const uci = `${sourceSquare}${targetSquare}${move.promotion || ''}`;
    const nextPly = movesRef.current.length + 1;

    // Optimistically record human move in moves list
    const humanMoveObj: MoveRecord = {
      uci,
      san: move.san,
      from: sourceSquare,
      to: targetSquare,
    };
    setMoves((prev) => [...prev, humanMoveObj]);
    movesRef.current = [...movesRef.current, humanMoveObj];

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ t: 'move', uci, n: nextPly }));
    }
    return true;
  };

  // ── Unified square/piece selection for Click-to-Move ──
  const handleSquareSelection = (square: string) => {
    if (!gameId || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      clearHighlights();
      return;
    }
    if (viewIndexRef.current !== null || isWaitingForBotRef.current) {
      clearHighlights();
      return;
    }
    const game = new Chess(fenRef.current);
    const humanChessColor = humanColorRef.current === 'white' ? 'w' : 'b';
    const currentStatus = gameStatusRef.current;
    const isGameOver = gameOverOverlayRef.current || (currentStatus !== 'in_progress' && currentStatus !== 'active') || game.isGameOver();
    const isMyTurn = game.turn() === humanChessColor;

    if (isGameOver || !isMyTurn) {
      clearHighlights();
      return;
    }

    if (selectedSquare) {
      if (selectedSquare === square) {
        clearHighlights();
        return;
      }

      let move: any = null;
      try {
        move = game.move({ from: selectedSquare, to: square, promotion: 'q' });
      } catch {
        move = null;
      }

      if (move) {
        setFen(game.fen()); // This flips the turn immediately
        setTs(performance.now()); // Reset the tick timestamp
        fenRef.current = game.fen();
        lastTickTimeRef.current = performance.now();
        turnRef.current = game.fen().split(' ')[1] === 'w' ? 'white' : 'black';
        setIsWaitingForBot(true);
        isWaitingForBotRef.current = true;

        const humanColorVal = humanColorRef.current;

        // Lichess increment rule: add increment immediately upon human move completion
        if (selectedTc.inc_s > 0) {
          setClocks((prev) => {
            const next = {
              ...prev,
              [humanColorVal]: prev[humanColorVal] + selectedTc.inc_s * 1000,
            };
            clocksRef.current = next;
            return next;
          });
        }

        const uci = `${selectedSquare}${square}${move.promotion || ''}`;
        const nextPly = movesRef.current.length + 1;

        // Optimistically record human move in moves list
        const humanMoveObj: MoveRecord = {
          uci,
          san: move.san,
          from: selectedSquare,
          to: square,
        };
        setMoves((prev) => [...prev, humanMoveObj]);
        movesRef.current = [...movesRef.current, humanMoveObj];

        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ t: 'move', uci, n: nextPly }));
        }
        clearHighlights();
        return;
      }

      // Check if clicking another friendly piece to switch selection
      const pieceOnSquare = game.get(square as any);
      if (pieceOnSquare && pieceOnSquare.color === humanChessColor) {
        const styles = getLegalSquares(square);
        setSelectedSquare(square);
        setOptionSquares(styles);
        return;
      }

      clearHighlights();
      return;
    }

    const piece = game.get(square as any);
    if (piece && piece.color === humanChessColor) {
      const styles = getLegalSquares(square);
      setSelectedSquare(square);
      setOptionSquares(styles);
    } else {
      clearHighlights();
    }
  };

  // ── Click-to-move — react-chessboard v5 handlers ──
  const onSquareClick = ({ square }: { piece: any; square: string }) => {
    handleSquareSelection(square);
  };

  const onPieceClick = ({ square }: { isSparePiece: boolean; piece: any; square: string | null }) => {
    if (square) {
      handleSquareSelection(square);
    }
  };

  // ── Show legal dots on drag start — react-chessboard v5: onPieceDrag({ isSparePiece, piece, square }) ──
  const onPieceDrag = ({ square }: { isSparePiece: boolean; piece: any; square: string | null }) => {
    if (viewIndexRef.current !== null || isWaitingForBotRef.current || !square) return;
    const humanChessColor = humanColorRef.current === 'white' ? 'w' : 'b';
    const game = new Chess(fenRef.current);
    if (game.turn() !== humanChessColor) return;
    setSelectedSquare(square);
    setOptionSquares(getLegalSquares(square));
  };

  const onPieceDragCancel = () => clearHighlights();

  const handleResign = () => {
    if (gameStatus !== 'in_progress' && gameStatus !== 'active') return;
    if (window.confirm('Are you sure you want to resign?')) {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ t: 'resign' }));
      }
    }
  };

  const formatClock = (ms: number) => {
    if (ms <= 0) return '0:00';
    const totalSecs = ms / 1000;
    const mins = Math.floor(totalSecs / 60);
    const secs = totalSecs % 60;
    if (totalSecs < 20) {
      const secStr = secs < 10 ? `0${secs.toFixed(1)}` : secs.toFixed(1);
      return `${mins}:${secStr}`;
    }
    return `${mins}:${Math.floor(secs).toString().padStart(2, '0')}`;
  };

  // Move history pairs
  const movePairs: { white?: MoveRecord; black?: MoveRecord }[] = [];
  for (let i = 0; i < moves.length; i += 2) {
    movePairs.push({
      white: moves[i],
      black: moves[i + 1],
    });
  }

  // Pre-calculate human-readable moves & FEN snapshots using chess.js replay
  const { formattedMoves, moveFens } = useMemo(() => {
    const replayGame = new Chess();
    const fens: string[] = [];
    const formatted = moves.map((m) => {
      const from = m.from || (m.uci && m.uci.length >= 4 ? m.uci.slice(0, 2) : '');
      const to = m.to || (m.uci && m.uci.length >= 4 ? m.uci.slice(2, 4) : '');
      const moveObj = { ...m, from, to };
      const f = formatMove(replayGame, moveObj);
      try {
        if (from && to) {
          replayGame.move({ from, to, promotion: m.uci?.[4] || 'q' });
        } else if (m.san) {
          replayGame.move(m.san);
        }
      } catch {
        // fallback if move cannot be executed in replay
      }
      fens.push(replayGame.fen());
      return f;
    });
    return { formattedMoves: formatted, moveFens: fens };
  }, [moves]);

  // Display FEN (either live or historical review)
  const displayFen = viewIndex !== null && moveFens[viewIndex] ? moveFens[viewIndex] : fen;

  const currentTheme = BD_THEMES[boardTheme];

  // Combine legal move dots and active premove square highlights (Lichess style)
  const boardSquareStyles = useMemo(() => {
    const styles: Record<string, React.CSSProperties> = { ...optionSquares };
    if (premove) {
      styles[premove.from] = {
        backgroundColor: 'rgba(47, 123, 255, 0.45)',
        boxShadow: 'inset 0 0 0 3px #2F7BFF',
      };
      styles[premove.to] = {
        backgroundColor: 'rgba(47, 123, 255, 0.65)',
        boxShadow: 'inset 0 0 0 3px #2F7BFF',
      };
    }
    return styles;
  }, [optionSquares, premove]);

  return (
    <div className="min-h-screen relative bg-black text-white selection:bg-blue-600 selection:text-white flex flex-col justify-between">
      {/* Background Light Fall graphics matching screenshot */}
      <LightFallBackground />

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
            {/* Active Premove Indicator Badge */}
            {premove && (
              <div className="absolute top-2 left-2 z-20 bg-[#2F7BFF]/95 backdrop-blur-md text-white text-xs px-3 py-1.5 rounded shadow-lg flex items-center gap-2 font-mono border border-white/25">
                <span className="w-2 h-2 rounded-full bg-white animate-ping" />
                <span>Premove: <b>{premove.from}→{premove.to}</b></span>
                <button
                  onClick={() => { setPremove(null); premoveRef.current = null; }}
                  className="text-white/80 hover:text-white font-bold ml-1 hover:scale-110 transition-transform"
                  title="Cancel premove (or right-click board)"
                >
                  ✕
                </button>
              </div>
            )}

            <Chessboard
              options={{
                id: 'play-board',
                position: displayFen,
                boardOrientation: flipped ? 'black' : 'white',
                allowDragging: viewIndex === null,
                canDragPiece,
                onPieceDrop,
                onSquareClick,
                onSquareRightClick: () => {
                  setPremove(null);
                  premoveRef.current = null;
                  clearHighlights();
                },
                onPieceClick,
                onPieceDrag,
                onPieceDragCancel,
                squareStyles: boardSquareStyles,
                boardStyle: {
                  borderRadius: '0px',
                  boxShadow: '0 10px 40px rgba(0,0,0,.6)',
                },
                darkSquareStyle: { backgroundColor: currentTheme.dk },
                lightSquareStyle: { backgroundColor: currentTheme.lt },
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
                  {formatTerminationReason(gameTermination)} · {gameResult || '½-½'}
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
                AI Clone Engine
              </small>
            </div>
            <div
              className={`clk font-mono font-semibold text-2xl px-3 py-1.5 rounded min-w-[96px] text-center transition-colors ${
                (isWaitingForBot || turn !== humanColor) && (gameStatus === 'in_progress' || gameStatus === 'active')
                  ? 'bg-white text-black shadow-[0_0_15px_rgba(255,255,255,0.4)]'
                  : 'bg-white/5 text-[#8b8b8b]'
              } ${
                (isWaitingForBot || turn !== humanColor) &&
                (humanColor === 'white' ? clocks.black : clocks.white) < 20000 &&
                (gameStatus === 'in_progress' || gameStatus === 'active')
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
                    {formattedMoves[index * 2] || pair.white?.san || pair.white?.uci || ''}
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
                    {formattedMoves[index * 2 + 1] || pair.black?.san || pair.black?.uci || ''}
                  </span>
                </React.Fragment>
              ))}

              {gameResult && (
                <div className="col-span-3 text-center py-2.5 font-mono text-sm font-semibold border-t border-[rgba(255,255,255,0.16)] text-[#2F7BFF]">
                  {gameResult} {gameTermination ? `· ${formatTerminationReason(gameTermination)}` : ''}
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
                (isWaitingForBot || turn !== humanColor) && (gameStatus === 'in_progress' || gameStatus === 'active')
                  ? 'bg-[#2F7BFF] shadow-[0_0_10px_#2F7BFF] animate-pulse'
                  : 'bg-white shadow-[0_0_6px_#fff]'
              }`}
            />
            <span className="truncate">
              {viewIndex !== null
                ? 'Reviewing — Click ⏭ for live board'
                : (gameStatus === 'completed' || gameResult || gameOverOverlay)
                ? `Game over · ${formatTerminationReason(gameTermination)} (${gameResult || '½-½'})`
                : (isWaitingForBot || turn !== humanColor) && (gameStatus === 'in_progress' || gameStatus === 'active')
                ? `${username} is calculating…`
                : (gameStatus === 'in_progress' || gameStatus === 'active')
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
              disabled={gameStatus !== 'in_progress' && gameStatus !== 'active'}
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
                !isWaitingForBot && turn === humanColor && (gameStatus === 'in_progress' || gameStatus === 'active')
                  ? 'bg-white text-black shadow-[0_0_15px_rgba(255,255,255,0.4)]'
                  : 'bg-white/5 text-[#8b8b8b]'
              } ${
                !isWaitingForBot &&
                turn === humanColor &&
                (humanColor === 'white' ? clocks.white : clocks.black) < 20000 &&
                (gameStatus === 'in_progress' || gameStatus === 'active')
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
