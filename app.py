import os
import time
import sqlite3
import threading
import logging
from datetime import datetime
from contextlib import closing
from flask import Flask, request, Response, jsonify
from functools import wraps
from dotenv import load_dotenv

# Import bot class + shared config/SQL from main script so the writer and the
# reader can never disagree about schema, column order, or JSON keys.
from live_testnet_monitor_v6 import (
    RealExecutionRotatorBot,
    TOTAL_BOT_BUDGET,
    DB_PATH,
    TRADES_SELECT,
    ensure_local_schema,
    rows_to_trades,
)

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Basic auth config
WEB_USERNAME = os.getenv("WEB_USERNAME", "admin")
WEB_PASSWORD = os.getenv("WEB_PASSWORD", "changeme")

def check_auth(username, password):
    return username == WEB_USERNAME and password == WEB_PASSWORD

def authenticate():
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# Bot instance
bot = None
bot_thread = None
bot_running = False

def bot_worker():
    global bot, bot_running
    try:
        # Sanitize API credentials: strip whitespace/newlines/quoting noise
        api_key = (os.getenv("BINANCE_API_KEY") or "").strip()
        api_secret = (os.getenv("BINANCE_API_SECRET") or "").strip()
        if not api_key or not api_secret:
            logger.error("Missing BINANCE_API_KEY or BINANCE_API_SECRET")
            return
        bot = RealExecutionRotatorBot(api_key=api_key, api_secret=api_secret, use_render_mode=True)
        bot_running = True
        logger.info("Bot worker started")

        # Replicate the main loop from v6 __main__ (lines 779-826)
        bot.seconds_until_refresh = 0
        markets_state = {}
        # Virtual balance: start with TOTAL_BOT_BUDGET and track realized P&L
        balance_info = {'free': TOTAL_BOT_BUDGET, 'total': TOTAL_BOT_BUDGET}

        while True:
            # Update market data every second from cache (WebSocket)
            try:
                for symbol in bot.active_assets:
                    df = bot.fetch_market_data(symbol)
                    if df is not None:
                        last_row = df.iloc[-1]
                        vol_sma = last_row['vol_sma']
                        markets_state[symbol] = {
                            'price': last_row['close'],
                            'trend': 'BULLISH' if last_row['close'] > last_row['ema_200'] else 'BEARISH',
                            'adx': last_row['adx'],
                            'vol_ratio': last_row['volume'] / vol_sma if vol_sma > 0 else 1.0,
                            'signal': 'WAITING',  # will be updated on refresh
                            'trigger_long': last_row['donchian_high'],
                            'trigger_short': last_row['donchian_low']
                        }
            except Exception as e:
                logger.error("Error in market data update: %s", e)

            # Refresh signals and balance periodically (every 5 seconds)
            if bot.seconds_until_refresh <= 0:
                try:
                    if bot.last_scan_time is None or datetime.now() >= bot.next_scan_time:
                        bot.scan_daily_market()
                except Exception as e:
                    logger.error("Error in daily scan: %s", e)

                # Update signals (may open/close positions)
                try:
                    for symbol in bot.active_assets:
                        df = bot.fetch_market_data(symbol)
                        if df is not None:
                            signal = bot.check_signals(symbol, df)
                            if symbol in markets_state:
                                markets_state[symbol]['signal'] = signal
                except Exception as e:
                    logger.error("Error in signal check: %s", e)

                # Update virtual balance (based on realized P&L)
                bot.balance_info = {'free': TOTAL_BOT_BUDGET + bot.realized_pnl, 'total': TOTAL_BOT_BUDGET + bot.realized_pnl}
                balance_info = bot.balance_info

                bot.seconds_until_refresh = 5  # refresh signals and balance every 5 seconds

            bot.markets_state = markets_state
            bot.render_dashboard(markets_state, balance_info)
            time.sleep(1)
            bot.seconds_until_refresh -= 1

    except Exception as e:
        logger.exception("Bot worker crashed: %s", e)
        bot_running = False

# Routes
@app.route('/')
@requires_auth
def index():
    status = "running" if bot_running else "stopped"
    assets = ', '.join(bot.active_assets) if bot and bot.active_assets else 'None'
    last_scan = bot.last_scan_time.strftime('%Y-%m-%d %H:%M:%S') if bot and bot.last_scan_time else 'Never'
    positions_count = len(bot.positions) if bot else 0
    markets_state = bot.markets_state if bot and hasattr(bot, 'markets_state') else {}
    balance_info = bot.balance_info if bot and hasattr(bot, 'balance_info') else {'free': 0, 'total': 0}

    # Countdown to next scan
    countdown = ""
    if bot and bot.next_scan_time:
        now = datetime.now()
        if now < bot.next_scan_time:
            diff = bot.next_scan_time - now
            hours, rem = divmod(diff.total_seconds(), 3600)
            minutes, seconds = divmod(rem, 60)
            countdown = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
        else:
            countdown = "Scanning now..."

    # Market table rows
    market_rows = ""
    for symbol in bot.active_assets if bot else []:
        if symbol in markets_state:
            s = markets_state[symbol]
            market_rows += f"""
            <tr>
                <td>{symbol.split('/')[0]}</td>
                <td>{s['price']:.5f}</td>
                <td>{s['trend']}</td>
                <td>{s['adx']:.1f}</td>
                <td>{s['vol_ratio']:.2f}</td>
                <td>{s['signal']}</td>
                <td>{s['trigger_long']:.5f}</td>
                <td>{s['trigger_short']:.5f}</td>
            </tr>
            """
        else:
            market_rows += f"""
            <tr>
                <td>{symbol.split('/')[0]}</td>
                <td colspan="7">No data</td>
            </tr>
            """

    # Positions table rows
    pos_rows = ""
    if bot:
        for sym, pos in bot.positions.items():
            curr_price = markets_state[sym]['price'] if sym in markets_state else pos['entry_price']
            entry = pos['entry_price']
            side = pos['type']
            qty = pos['quantity']
            if side == 'LONG':
                raw_pnl = (curr_price - entry) / entry
            else:
                raw_pnl = (entry - curr_price) / entry
            pnl_pct = raw_pnl * 10 * 100  # leverage 10x
            pnl_color = "green" if pnl_pct >= 0 else "red"
            pos_rows += f"""
            <tr>
                <td>{sym.split('/')[0]}</td>
                <td>{side}</td>
                <td>{qty}</td>
                <td style="color:{pnl_color};">{pnl_pct:+.2f}%</td>
            </tr>
            """
        if not pos_rows:
            pos_rows = "<tr><td colspan='4'>No active positions</td></tr>"

    return f"""
    <html>
        <head>
            <title>Bot Status</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .card {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 900px; margin: auto; }}
                h1 {{ color: #333; }}
                .status {{ font-weight: bold; }}
                .ok {{ color: green; }}
                .stop {{ color: red; }}
                .info {{ margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
        <div class="card">
            <h1>🤖 Trading Bot Status</h1>
            <div class="info"><span class="status">Status:</span> <span class="{ 'ok' if status == 'running' else 'stop' }">{status.upper()}</span></div>
            <div class="info"><span class="status">Active assets:</span> {assets}</div>
            <div class="info"><span class="status">Last scan:</span> <span id="lastScan">{last_scan}</span></div>
            <div class="info"><span class="status">Next scan in:</span> {countdown}</div>
            <div class="info"><span class="status">Open positions:</span> <span id="posCount">{positions_count}</span></div>
            <div class="info"><span class="status">USDT Balance (Bot):</span> <span id="balanceFree">{balance_info['free']:.2f}</span></div>
            <div class="info"><span class="status">Live Time:</span> <span id="liveClock">--:--:--</span></div>
            <div class="info"><span class="status">Bot worker:</span> {'Active' if bot_running else 'Stopped'}</div>
            <hr>
            <h2>Market Data</h2>
            <table>
                <tr>
                    <th>Asset</th>
                    <th>Price</th>
                    <th>Trend</th>
                    <th>ADX</th>
                    <th>Vol Ratio</th>
                    <th>Signal</th>
                    <th>Trigger LONG</th>
                    <th>Trigger SHORT</th>
                </tr>
                <tbody id="marketRows">
                {market_rows}
                </tbody>
            </table>
            <hr>
            <h2>Active Positions</h2>
            <table>
                <tr>
                    <th>Asset</th>
                    <th>Side</th>
                    <th>Quantity</th>
                    <th>P&L (%)</th>
                </tr>
                <tbody id="posRows">
                {pos_rows}
                </tbody>
            </table>
            <hr>
            <h2>Trade History <span id="pendingSync" style="font-size:12px;font-weight:normal;color:#666"></span></h2>
            <table>
                <tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Exit</th><th>Return %</th><th>Reason</th></tr>
                <tbody id="tradeRows"><tr><td colspan="8">Loading...</td></tr></tbody>
            </table>
            <hr>
            <div class="info">📊 <a href="/status">JSON status</a> &nbsp;|&nbsp; 🏓 <a href="/ping">Health check</a></div>
        </div>
        </body>
        <script>
        function fmt(n) {{ return n.toFixed(5); }}
        async function refresh() {{
            try {{
                const r = await fetch('/api/live');
                const d = await r.json();
                document.getElementById('balanceFree').textContent = (d.balance.free || 0).toFixed(2);
                document.getElementById('posCount').textContent = d.positions.length;
                document.getElementById('lastScan').textContent = d.last_scan;

                let mhtml = '';
                for (const a of d.assets) {{
                    const trend = a.trend === 'BULLISH' ? '<span style="color:#22c55e">BULLISH</span>' :
                                  (a.trend === 'BEARISH' ? '<span style="color:#ef4444">BEARISH</span>' : a.trend);
                    mhtml += '<tr><td>' + a.symbol + '</td><td style="font-weight:bold">' + fmt(a.price) +
                        '</td><td>' + trend + '</td><td>' + a.adx.toFixed(1) + '</td><td>' +
                        a.vol_ratio.toFixed(2) + '</td><td>' + a.signal + '</td><td>' + fmt(a.trigger_long) +
                        '</td><td>' + fmt(a.trigger_short) + '</td></tr>';
                }}
                document.getElementById('marketRows').innerHTML = mhtml || '<tr><td colspan="8">No data</td></tr>';

                let phtml = '';
                for (const p of d.positions) {{
                    const color = p.pnl_pct >= 0 ? '#22c55e' : '#ef4444';
                    const sign = p.pnl_pct >= 0 ? '+' : '';
                    phtml += '<tr><td>' + p.symbol + '</td><td>' + p.side + '</td><td>' + p.qty +
                        '</td><td style="color:' + color + ';font-weight:bold">' + sign + p.pnl_pct +
                        '%</td></tr>';
                }}
                document.getElementById('posRows').innerHTML = phtml || '<tr><td colspan="4">No active positions</td></tr>';

                // Fetch trade history
                const tr = await fetch('/api/trades');
                const td = await tr.json();
                const tbody = document.getElementById('tradeRows');
                if (td.error) {{
                    tbody.innerHTML = '<tr><td colspan="8" style="color:#ef4444">History unavailable</td></tr>';
                }} else {{
                    let thtml = '';
                    for (const t of (td.trades || [])) {{
                        const retColor = t.return_pct >= 0 ? 'green' : 'red';
                        thtml += '<tr><td>' + t.timestamp + '</td><td>' + t.symbol.split('/')[0] + '</td><td>' + t.side + '</td><td>' + t.quantity.toFixed(4) + '</td><td>' + t.entry.toFixed(5) + '</td><td>' + t.exit.toFixed(5) + '</td><td style="color:' + retColor + '">' + t.return_pct.toFixed(2) + '%</td><td>' + t.reason + '</td></tr>';
                    }}
                    tbody.innerHTML = thtml || '<tr><td colspan="8">No trades yet</td></tr>';
                    const sync = document.getElementById('pendingSync');
                    if (sync) {{
                        sync.textContent = td.pending_sync > 0 ? (td.pending_sync + ' menunggu sync ke Turso') : 'tersinkron';
                    }}
                }}
            }} catch (e) {{}}
        }}
        setInterval(refresh, 1000);
        refresh();
        // Client-side clock
        function updateClock() {{
            const now = new Date();
            document.getElementById('liveClock').textContent = now.toLocaleTimeString();
        }}
        setInterval(updateClock, 1000);
        updateClock();
        </script>
    </html>
    """

@app.route('/api/live')
@requires_auth
def api_live():
    assets = []
    if bot and bot.active_assets:
        for symbol in bot.active_assets:
            s = bot.markets_state.get(symbol, {})
            assets.append({
                'symbol': symbol.split('/')[0],
                'price': s.get('price', 0.0),
                'trend': s.get('trend', 'UNKNOWN'),
                'adx': s.get('adx', 0.0),
                'vol_ratio': s.get('vol_ratio', 0.0),
                'signal': s.get('signal', 'NO_DATA'),
                'trigger_long': s.get('trigger_long', 0.0),
                'trigger_short': s.get('trigger_short', 0.0),
            })
    positions = []
    if bot:
        for sym, pos in bot.positions.items():
            markets_state = bot.markets_state.get(sym, {})
            price = markets_state.get('price', pos['entry_price'])
            side = pos['type']
            raw = (price - pos['entry_price']) / pos['entry_price'] if side == 'LONG' else (pos['entry_price'] - price) / pos['entry_price']
            positions.append({
                'symbol': sym.split('/')[0],
                'side': side,
                'qty': pos['quantity'],
                'pnl_pct': round(raw * 10 * 100, 2),
            })
    balance = bot.balance_info if bot and hasattr(bot, 'balance_info') else {'free': 0.0, 'total': 0.0}
    return jsonify({
        'status': 'running' if bot_running else 'stopped',
        'balance': balance,
        'assets': assets,
        'positions': positions,
        'last_scan': bot.last_scan_time.strftime('%Y-%m-%d %H:%M:%S') if bot and bot.last_scan_time else 'Never',
    })

@app.route('/ping')
def ping():
    return "pong", 200

TRADES_LIMIT = 100

@app.route('/api/trades')
@requires_auth
def api_trades():
    """
    Reads trade history from the local SQLite store, which is authoritative:
    every closed trade is written there first, so it is never behind Turso.

    Uses a fresh per-request connection instead of the bot's Turso handle:
    that handle belongs to the bot worker thread and is not safe to share.

    Response shape: {'trades': [...], 'error': str|None, 'pending_sync': int}
    `error` lets the dashboard distinguish "store unavailable" from "no trades".
    """
    if not os.path.exists(DB_PATH):
        return jsonify({'trades': [], 'error': None, 'pending_sync': 0})
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            ensure_local_schema(conn)
            conn.commit()
            rows = conn.execute(TRADES_SELECT, (TRADES_LIMIT,)).fetchall()
            pending = conn.execute("SELECT COUNT(*) FROM trades WHERE synced = 0").fetchone()[0]
        return jsonify({
            'trades': rows_to_trades(rows),
            'error': None,
            'pending_sync': pending,
        })
    except Exception as e:
        logger.error("Trade history read failed: %s", e)
        # Report the failure instead of an empty list, so an unreachable store
        # is never rendered as a legitimately empty history.
        return jsonify({
            'trades': [],
            'error': 'history unavailable',
            'pending_sync': 0,
        }), 503

@app.route('/status')
@requires_auth
def status():
    if bot is None:
        return jsonify({"status": "not_initialized"})
    return jsonify({
        "status": "running" if bot_running else "stopped",
        "active_assets": bot.active_assets if bot else [],
        "last_scan": bot.last_scan_time.isoformat() if bot.last_scan_time else None,
        "positions": list(bot.positions.keys()) if bot else []
    })

def start_bot_thread():
    global bot_thread
    if bot_thread is None or not bot_thread.is_alive():
        bot_thread = threading.Thread(target=bot_worker, daemon=True)
        bot_thread.start()
        logger.info("Bot thread started")

if __name__ == "__main__":
    start_bot_thread()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)