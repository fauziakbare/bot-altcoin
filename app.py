import os
import time
import threading
import logging
from datetime import datetime
from flask import Flask, request, Response, jsonify
from functools import wraps
from dotenv import load_dotenv

# Import bot class from main script
from live_testnet_monitor_v6 import RealExecutionRotatorBot

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
        balance_info = {'free': 5000.0, 'total': 5000.0}

        while True:
            if bot.seconds_until_refresh <= 0:
                try:
                    if bot.last_scan_time is None or datetime.now() >= bot.next_scan_time:
                        bot.scan_daily_market()

                    balance = bot.trade_client.fetch_balance()
                    usdt_free = balance['free'].get('USDT', 5000.0)
                    usdt_total = balance['total'].get('USDT', 5000.0)
                    balance_info = {'free': usdt_free, 'total': usdt_total}

                    for symbol in bot.active_assets:
                        df = bot.fetch_market_data(symbol)
                        if df is not None:
                            last_row = df.iloc[-1]
                            signal = bot.check_signals(symbol, df)
                            vol_sma = last_row['vol_sma']
                            markets_state[symbol] = {
                                'price': last_row['close'],
                                'trend': 'BULLISH' if last_row['close'] > last_row['ema_200'] else 'BEARISH',
                                'adx': last_row['adx'],
                                'vol_ratio': last_row['volume'] / vol_sma if vol_sma > 0 else 1.0,
                                'signal': signal,
                                'trigger_long': last_row['donchian_high'],
                                'trigger_short': last_row['donchian_low']
                            }
                except Exception as e:
                    logger.error("Error in market tick: %s", e)

                bot.seconds_until_refresh = 15

            bot.markets_state = markets_state
            bot.balance_info = balance_info
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
            <meta http-equiv="refresh" content="15">
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
            <div class="info"><span class="status">Last scan:</span> {last_scan}</div>
            <div class="info"><span class="status">Next scan in:</span> {countdown}</div>
            <div class="info"><span class="status">Open positions:</span> {positions_count}</div>
            <div class="info"><span class="status">USDT Balance:</span> {balance_info['free']:.2f} (free) / {balance_info['total']:.2f} (total)</div>
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
                {market_rows}
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
                {pos_rows}
            </table>
            <hr>
            <div class="info">📊 <a href="/status">JSON status</a> &nbsp;|&nbsp; 🏓 <a href="/ping">Health check</a></div>
        </div>
        </body>
    </html>
    """

@app.route('/ping')
def ping():
    return "pong", 200

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