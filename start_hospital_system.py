#!/usr/bin/env python3
"""
Master Launcher for Emergency Hospital Response System
1. Starts the Hospital Emergency Dispatch Server (http://127.0.0.1:5001)
2. Opens the Hospital Dashboard UI in default browser
"""

import sys
import time
import webbrowser
from pathlib import Path

# Add project directory to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from server.app import app


def main():
    print("=" * 70)
    print("HOSPITAL EMERGENCY RESPONSE SYSTEM")
    print("=" * 70)
    print("\n[1/2] Starting Hospital Emergency Dispatch Server...")

    print("[OK] Server starting on http://127.0.0.1:5001")
    print("  🏥 Hospital Dashboard: http://127.0.0.1:5001")

    print("\n[2/2] Opening Hospital Emergency Dashboard in Web Browser...")
    try:
        import threading
        def open_browser():
            time.sleep(1.5)
            webbrowser.open("http://127.0.0.1:5001")
        threading.Thread(target=open_browser, daemon=True).start()
        print("[OK] Dashboard will open automatically.")
    except Exception as e:
        print(f"[WARNING] Could not auto-open browser: {e}")

    print("\n" + "=" * 70)
    print("Press Ctrl+C to stop the server.")
    print("=" * 70 + "\n")

    app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)


if __name__ == '__main__':
    main()
