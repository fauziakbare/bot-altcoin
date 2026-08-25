"""
Runs the trading bot with Telegram as the only control surface.

No HTTP server is started here, so nothing listens on a network port: the bot
is monitored entirely through Telegram long polling (outbound connections only).

Usage:  python run_bot.py
"""
import logging
import os
import sys
import threading

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "bot.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# `app` owns the trading loop (bot_worker). Importing it does not start Flask:
# app.run() only executes under app.py's own __main__ guard. Reusing it keeps
# the loop defined in exactly one place.
import app  # noqa: E402
from telegram_bot import build_bridge  # noqa: E402


def main():
    bridge = build_bridge(
        bot_provider=lambda: app.bot,
        running_provider=lambda: app.bot_running,
    )

    if bridge is None:
        logger.warning("Telegram tidak aktif — bot jalan tanpa kanal pantau.")
    else:
        # Publish the notifier before the worker builds the bot, so position
        # events are pushed from the very first trade.
        app.notifier = bridge
        threading.Thread(target=bridge.run, name="telegram", daemon=True).start()
        bridge.broadcast(
            "🤖 BOT DIMULAI\n"
            "✅ Trading loop aktif.\n"
            "❓ Kirim /help untuk daftar command."
        )

    try:
        app.bot_worker()
    except KeyboardInterrupt:
        logger.info("Shutdown diminta pengguna.")
    finally:
        if bridge is not None:
            bridge.stop()


if __name__ == "__main__":
    main()
