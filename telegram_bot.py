"""
Telegram control surface for the trading bot.

Read-only by design: exposes status, balance, positions, market metrics, trade
history, and logs. No command can place, modify, or close an order, so even a
compromised chat cannot move funds.

Access is restricted to an explicit chat-id allowlist. Messages from any other
chat are ignored, and their contents are never echoed back or written to logs.
"""
import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime

import requests

from live_testnet_monitor_v7 import (
    COLLATERAL_PER_TRADE,
    DB_PATH,
    LEVERAGE,
    SCHEDULED_SCAN_HOUR,
    TOTAL_BOT_BUDGET,
    TRADES_SELECT,
    ensure_local_schema,
    rows_to_trades,
)

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org/bot{token}/{method}"
POLL_TIMEOUT = 30           # seconds Telegram holds the long-poll request open
HISTORY_DEFAULT = 5
HISTORY_MAX = 20
LOG_LINES = 10
MAX_MESSAGE_CHARS = 3900    # Telegram hard limit is 4096; leave headroom


def parse_allowed_ids(raw):
    """Parses TELEGRAM_ALLOWED_CHAT_IDS ("111,222") into a set of ints."""
    ids = set()
    for chunk in (raw or "").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.add(int(chunk))
        except ValueError:
            logger.warning("Melewati chat id non-numerik di TELEGRAM_ALLOWED_CHAT_IDS")
    return ids


def _truncate(text):
    """Keeps a message under the Telegram size limit."""
    if len(text) <= MAX_MESSAGE_CHARS:
        return text
    return text[:MAX_MESSAGE_CHARS - 20] + "\n... (dipotong)"


# ---------------------------------------------------------------------------
# Emoji helpers
# ---------------------------------------------------------------------------
# Emoji are decorative only: every value they accompany is also printed as text,
# so the message stays readable if a client cannot render them.

def emo_pnl(value):
    """Profit/loss indicator."""
    if value > 0:
        return "🟢"
    if value < 0:
        return "🔴"
    return "⚪"


def emo_side(side):
    """Position direction."""
    return "📈" if side == "LONG" else "📉"


def emo_trend(trend):
    """EMA200 trend regime."""
    if trend == "BULLISH":
        return "🐂"
    if trend == "BEARISH":
        return "🐻"
    return "❔"


def emo_adx(adx):
    """ADX regime: trending vs choppy."""
    return "🔥" if adx >= 25 else "😴"


def emo_vol(vol_ratio):
    """Volume confirmation against its 20-period SMA."""
    return "🔊" if vol_ratio >= 1.5 else "🔇"


def emo_signal(signal):
    """Current strategy state."""
    if "BUY" in signal:
        return "🚀"
    if "SELL" in signal:
        return "🩸"
    if "HOLDING" in signal:
        return "✋"
    if "ERROR" in signal or "NO_DATA" in signal:
        return "⚠️"
    return "⏳"


def emo_reason(reason):
    """Exit reason category."""
    if "TAKE_PROFIT" in reason:
        return "🎯"
    if "TRAILING_STOP" in reason:
        return "🛑"
    if "REGIME_EXIT" in reason:
        return "🌫️"
    return "🏁"


def emo_health(ok):
    """Store/service health."""
    return "✅" if ok else "❌"


class TelegramBridge:
    """
    Long-polling Telegram client running in its own daemon thread.

    Bot state is read through providers (callables) rather than a direct
    reference, because the trading bot instance is created later by the worker
    thread and may be replaced after a restart.
    """

    def __init__(self, token, allowed_chat_ids, bot_provider, running_provider):
        self.token = token
        self.allowed = set(allowed_chat_ids)
        self._bot_provider = bot_provider
        self._running_provider = running_provider
        self._offset = None
        self._stop = threading.Event()
        self._session = requests.Session()
        self.started_at = datetime.now()

    # ------------------------------------------------------------- transport
    def _call(self, method, **params):
        url = API_BASE.format(token=self.token, method=method)
        try:
            resp = self._session.post(url, json=params, timeout=POLL_TIMEOUT + 15)
            data = resp.json()
            if not data.get("ok"):
                # `description` can echo our own payload, so log the code only.
                logger.error("Telegram %s gagal: error_code=%s", method, data.get("error_code"))
                return None
            return data.get("result")
        except Exception as e:
            logger.error("Telegram %s error transport: %s", method, type(e).__name__)
            return None

    def send(self, chat_id, text):
        # No parse_mode on purpose: symbols and exchange error strings are sent
        # verbatim instead of being interpreted as Markdown/HTML markup.
        self._call(
            "sendMessage",
            chat_id=chat_id,
            text=_truncate(text),
            disable_web_page_preview=True,
        )

    def broadcast(self, text):
        """Pushes an event to every allowlisted chat. Used by bot._notify()."""
        for chat_id in self.allowed:
            self.send(chat_id, text)

    # ------------------------------------------------------------- lifecycle
    def stop(self):
        self._stop.set()

    def run(self):
        if not self.allowed:
            logger.error("Allowlist kosong — Telegram bridge tidak dijalankan.")
            return
        logger.info("Telegram bridge aktif untuk %d chat terotorisasi", len(self.allowed))
        backoff = 1
        while not self._stop.is_set():
            updates = self._call("getUpdates", offset=self._offset, timeout=POLL_TIMEOUT)
            if updates is None:
                # Network/API failure: back off so a sustained outage does not
                # spin the loop, but stay capped so recovery is quick.
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            backoff = 1
            for update in updates:
                self._offset = update["update_id"] + 1
                try:
                    self._handle(update)
                except Exception as e:
                    logger.exception("Gagal memproses update Telegram: %s", e)

    # -------------------------------------------------------------- dispatch
    def _handle(self, update):
        message = update.get("message") or update.get("edited_message")
        if not message:
            return
        chat_id = (message.get("chat") or {}).get("id")
        if chat_id not in self.allowed:
            # Log the id only; never the message body.
            logger.warning("Pesan Telegram dari chat tidak berizin diabaikan: id=%s", chat_id)
            return

        text = (message.get("text") or "").strip()
        if not text.startswith("/"):
            return

        parts = text.split()
        command = parts[0].split("@", 1)[0].lower()
        args = parts[1:]

        handler = self.COMMANDS.get(command)
        if handler is None:
            self.send(chat_id, f"❓ Command {command} tidak dikenal. Kirim /help.")
            return
        self.send(chat_id, handler(self, args))

    # -------------------------------------------------------------- commands
    def _bot(self):
        return self._bot_provider()

    def cmd_help(self, args):
        return (
            "🤖 PANTAU BOT TRADING\n\n"
            "📊 /status   - status bot, aset aktif, jadwal scan, penyimpanan\n"
            "📈 /market   - metrik 15m aset aktif (harga, ADX, volume, signal)\n"
            "💼 /posisi   - posisi terbuka + P&L live\n"
            "💰 /saldo    - saldo bot, realized P&L, margin per trade\n"
            "📜 /history  - trade terakhir (contoh: /history 10, maks 20)\n"
            "🏆 /stats    - rekap win rate, profit, loss\n"
            "📋 /log      - 10 log terbaru\n"
            "❓ /help     - pesan ini\n\n"
            "🔔 Notifikasi otomatis dikirim saat posisi dibuka dan ditutup.\n"
            "🔒 Bot ini hanya baca — tidak ada command untuk kirim order."
        )

    def cmd_status(self, args):
        bot = self._bot()
        running = self._running_provider()
        if bot is None:
            return (
                f"{'🟢' if running else '🔴'} Bot worker : {'AKTIF' if running else 'BERHENTI'}\n"
                "⏳ Instance belum siap (masih inisialisasi / pre-fetch candle)."
            )

        lines = [
            f"{'🟢' if running else '🔴'} Status     : {'RUNNING' if running else 'STOPPED'}",
            f"🎯 Aset aktif : {', '.join(s.split('/')[0] for s in bot.active_assets) or '-'}",
            f"💼 Posisi buka: {len(bot.positions)}",
        ]
        last = bot.last_scan_time
        lines.append(f"🕐 Scan akhir : {last.strftime('%Y-%m-%d %H:%M:%S') if last else 'Belum'}")

        nxt = bot.next_scan_time
        if nxt:
            secs = int((nxt - datetime.now()).total_seconds())
            if secs > 0:
                hours, rem = divmod(secs, 3600)
                minutes, _ = divmod(rem, 60)
                lines.append(f"⏰ Scan next  : {hours:02d}j {minutes:02d}m (pukul {SCHEDULED_SCAN_HOUR:02d}:00)")
            else:
                lines.append("🔄 Scan next  : sedang berjalan")

        local_ok = bool(getattr(bot, "local_db_ready", False))
        store = [f"{emo_health(local_ok)} SQLite lokal {'OK' if local_ok else 'GAGAL'}"]
        if getattr(bot, "db_conn", None) is not None:
            store.append("✅ Turso OK")
        elif getattr(bot, "turso_configured", False):
            store.append("⚠️ Turso terputus")
        else:
            store.append("➖ Turso tidak aktif")
        lines.append(f"🗄️ Penyimpanan: {', '.join(store)}")

        uptime = datetime.now() - self.started_at
        hours, rem = divmod(int(uptime.total_seconds()), 3600)
        minutes, _ = divmod(rem, 60)
        lines.append(f"⏱️ Uptime     : {hours}j {minutes}m")
        return "\n".join(lines)

    def cmd_market(self, args):
        bot = self._bot()
        if bot is None:
            return "⏳ Instance bot belum siap."
        if not bot.active_assets:
            return "🔍 Belum ada aset aktif. Menunggu scan harian."

        blocks = []
        for symbol in bot.active_assets:
            state = bot.markets_state.get(symbol)
            name = symbol.split("/")[0]
            if not state:
                blocks.append(f"⚠️ {name}: belum ada data")
                continue
            adx = state["adx"]
            vol = state["vol_ratio"]
            trend = state["trend"]
            signal = state["signal"]
            blocks.append(
                f"🪙 {name} @ {state['price']:.5f}\n"
                f"  {emo_trend(trend)} Trend  : {trend}\n"
                f"  {emo_adx(adx)} ADX    : {adx:.1f} ({'TREND' if adx >= 25 else 'CHOP'})\n"
                f"  {emo_vol(vol)} Volume : {vol:.2f}x ({'OK' if vol >= 1.5 else 'lemah'})\n"
                f"  {emo_signal(signal)} Signal : {signal}\n"
                f"  🎚️ Trigger: L {state['trigger_long']:.5f} / S {state['trigger_short']:.5f}"
            )
            pend = state.get("pending_orders")
            if pend:
                for line in pend:
                    blocks.append(f"  ⏰ {line}")
        return "📈 METRIK 15M\n\n" + "\n\n".join(blocks)

    def cmd_posisi(self, args):
        bot = self._bot()
        if bot is None:
            return "⏳ Instance bot belum siap."
        if not bot.positions:
            return "😴 Tidak ada posisi terbuka."

        blocks = []
        total_pnl = 0.0
        for symbol, pos in bot.positions.items():
            state = bot.markets_state.get(symbol, {})
            price = state.get("price") or pos["entry_price"]
            entry = pos["entry_price"]
            side = pos["type"]
            raw = (price - entry) / entry if side == "LONG" else (entry - price) / entry
            pnl_pct = raw * LEVERAGE * 100
            pnl_usdt = COLLATERAL_PER_TRADE * raw * LEVERAGE
            total_pnl += pnl_usdt
            blocks.append(
                f"{emo_side(side)} {symbol.split('/')[0]} {side}\n"
                f"  🎬 Entry : {entry:.5f} ({pos.get('entry_time', '-')})\n"
                f"  💵 Now   : {price:.5f}\n"
                f"  📦 Qty   : {pos['quantity']}\n"
                f"  🛑 SL/TP : {pos['sl']:.5f} / {pos['tp']:.5f}\n"
                f"  {emo_pnl(pnl_pct)} P&L   : {pnl_pct:+.2f}% ({pnl_usdt:+.2f} USDT)"
            )
        header = (
            f"💼 POSISI TERBUKA ({len(bot.positions)})\n"
            f"{emo_pnl(total_pnl)} Total unrealized: {total_pnl:+.2f} USDT\n\n"
        )
        return header + "\n\n".join(blocks)

    def cmd_saldo(self, args):
        bot = self._bot()
        realized = getattr(bot, "realized_pnl", 0.0) if bot else 0.0
        equity = TOTAL_BOT_BUDGET + realized
        pct = (realized / TOTAL_BOT_BUDGET * 100) if TOTAL_BOT_BUDGET else 0.0

        unrealized = 0.0
        if bot and bot.positions:
            for symbol, pos in bot.positions.items():
                state = bot.markets_state.get(symbol, {})
                price = state.get("price") or pos["entry_price"]
                entry = pos["entry_price"]
                raw = ((price - entry) / entry) if pos["type"] == "LONG" else ((entry - price) / entry)
                unrealized += COLLATERAL_PER_TRADE * raw * LEVERAGE

        lines = [
            f"🏦 Budget awal  : {TOTAL_BOT_BUDGET:.2f} USDT",
            f"{emo_pnl(realized)} Realized P&L : {realized:+.2f} USDT ({pct:+.2f}%)",
            f"💵 Saldo bot    : {equity:.2f} USDT",
        ]
        if bot and bot.positions:
            lines.append(f"{emo_pnl(unrealized)} Unrealized   : {unrealized:+.2f} USDT")
            lines.append(f"📊 Equity kini  : {equity + unrealized:.2f} USDT")
        lines.append(f"🎫 Margin/trade : {COLLATERAL_PER_TRADE:.2f} USDT")
        lines.append(f"⚡ Leverage     : {LEVERAGE}x")
        return "💰 SALDO BOT\n\n" + "\n".join(lines)

    def cmd_history(self, args):
        limit = HISTORY_DEFAULT
        if args:
            try:
                limit = max(1, min(HISTORY_MAX, int(args[0])))
            except ValueError:
                return f"⚠️ Jumlah tidak valid. Contoh: /history 10 (maks {HISTORY_MAX})"

        if not os.path.exists(DB_PATH):
            return "📭 Belum ada database history."
        try:
            with closing(sqlite3.connect(DB_PATH)) as conn:
                ensure_local_schema(conn)
                conn.commit()
                rows = conn.execute(TRADES_SELECT, (limit,)).fetchall()
                total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
                pending = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE synced = 0"
                ).fetchone()[0]
        except Exception as e:
            logger.error("Gagal baca history: %s", e)
            return "❌ Gagal membaca history — store tidak terbaca."

        if not rows:
            return "📭 Belum ada trade tertutup."

        blocks = []
        for t in rows_to_trades(rows):
            ret = t["return_pct"]
            blocks.append(
                f"🕐 {t['timestamp']}\n"
                f"  {emo_side(t['side'])} {t['symbol'].split('/')[0]} {t['side']} x{t['quantity']}\n"
                f"  ➡️ {t['entry']:.5f} → {t['exit']:.5f}\n"
                f"  {emo_pnl(ret)} Net   : {ret:+.2f}%\n"
                f"  {emo_reason(t['reason'])} Alasan: {t['reason']}"
            )
        footer = ""
        if pending:
            footer = f"\n\n⏳ {pending} baris belum tersinkron ke Turso."
        return f"📜 HISTORY ({len(blocks)} dari {total})\n\n" + "\n\n".join(blocks) + footer

    def cmd_stats(self, args):
        if not os.path.exists(DB_PATH):
            return "📭 Belum ada database history."
        try:
            with closing(sqlite3.connect(DB_PATH)) as conn:
                ensure_local_schema(conn)
                conn.commit()
                row = conn.execute(
                    "SELECT COUNT(*), "
                    "SUM(CASE WHEN net_return_pct > 0 THEN 1 ELSE 0 END), "
                    "SUM(net_return_pct), AVG(net_return_pct), "
                    "MAX(net_return_pct), MIN(net_return_pct) "
                    "FROM trades"
                ).fetchone()
                by_reason = conn.execute(
                    "SELECT exit_reason, COUNT(*) FROM trades "
                    "GROUP BY exit_reason ORDER BY COUNT(*) DESC"
                ).fetchall()
        except Exception as e:
            logger.error("Gagal baca stats: %s", e)
            return "❌ Gagal membaca statistik — store tidak terbaca."

        total, wins, sum_pct, avg_pct, best, worst = row
        if not total:
            return "📭 Belum ada trade tertutup."

        wins = wins or 0
        losses = total - wins
        win_rate = wins / total * 100
        # Each trade is sized at COLLATERAL_PER_TRADE, so a percentage return
        # converts to USDT at that fixed notional.
        pnl_usdt = (sum_pct or 0.0) / 100 * COLLATERAL_PER_TRADE
        total_pct = sum_pct or 0.0

        lines = [
            f"🔢 Total trade : {total}",
            f"⚔️ Menang/kalah: {wins}/{losses}",
            f"{'🟢' if win_rate >= 50 else '🔴'} Win rate    : {win_rate:.1f}%",
            f"{emo_pnl(total_pct)} Total net   : {total_pct:+.2f}% ({pnl_usdt:+.2f} USDT)",
            f"{emo_pnl(avg_pct or 0.0)} Rata-rata   : {avg_pct or 0.0:+.2f}% per trade",
            f"🥇 Terbaik     : {best or 0.0:+.2f}%",
            f"🥶 Terburuk    : {worst or 0.0:+.2f}%",
        ]
        if by_reason:
            lines.append("\n🚪 Alasan keluar:")
            for reason, count in by_reason:
                lines.append(f"  {reason}: {count}")
        return "🏆 STATISTIK\n\n" + "\n".join(lines)

    def cmd_log(self, args):
        bot = self._bot()
        if bot is None or not getattr(bot, "logs", None):
            return "📭 Belum ada log."
        return "📋 LOG TERBARU\n\n" + "\n".join(bot.logs[-LOG_LINES:])

    COMMANDS = {
        "/start": cmd_help,
        "/help": cmd_help,
        "/status": cmd_status,
        "/market": cmd_market,
        "/posisi": cmd_posisi,
        "/position": cmd_posisi,
        "/saldo": cmd_saldo,
        "/balance": cmd_saldo,
        "/history": cmd_history,
        "/stats": cmd_stats,
        "/log": cmd_log,
    }


def build_bridge(bot_provider, running_provider):
    """
    Returns a configured TelegramBridge, or None when credentials are missing
    so the trading loop still runs without Telegram.
    """
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    allowed = parse_allowed_ids(os.getenv("TELEGRAM_ALLOWED_CHAT_IDS"))
    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN belum diset — Telegram bridge dilewati.")
        return None
    if not allowed:
        # Refuse to run without an allowlist: anyone who found the bot could
        # otherwise read positions, balance, and history.
        logger.error(
            "TELEGRAM_ALLOWED_CHAT_IDS belum diset — bridge dimatikan demi keamanan."
        )
        return None
    return TelegramBridge(token, allowed, bot_provider, running_provider)
