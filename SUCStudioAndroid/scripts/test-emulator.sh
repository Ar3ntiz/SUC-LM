#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERIAL="${1:?Use an emulator serial, for example emulator-5580}"
PORT="${2:-5037}"
[[ "$SERIAL" == emulator-* ]] || { echo 'Questo script installa soltanto su emulatori, mai su dispositivi fisici.' >&2; exit 1; }
SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
ADB=("$SDK/platform-tools/adb" -P "$PORT" -s "$SERIAL")
"$ROOT/gradlew" -p "$ROOT" :app:assembleDebug :app:assembleDebugAndroidTest
python3 "$ROOT/tests/remote_fixture.py" 9876 > "$ROOT/docs/FIXTURE_LOG.txt" 2>&1 &
FIXTURE_PID=$!
trap 'kill "$FIXTURE_PID" 2>/dev/null || true; "${ADB[@]}" reverse --remove tcp:9876 >/dev/null 2>&1 || true' EXIT
sleep 1
kill -0 "$FIXTURE_PID" || { echo 'La porta 9876 è già occupata.' >&2; exit 1; }
"${ADB[@]}" reverse tcp:9876 tcp:9876
"${ADB[@]}" install -r "$ROOT/app/build/outputs/apk/debug/app-debug.apk"
"${ADB[@]}" install -r "$ROOT/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
"${ADB[@]}" shell am instrument -w local.sucstudio.android.test/local.sucstudio.android.SmokeInstrumentation | tee "$ROOT/docs/EMULATOR_TEST_RESULTS.txt"
! grep -q 'FAIL:' "$ROOT/docs/EMULATOR_TEST_RESULTS.txt"
grep -q 'PASS: persistent sessions' "$ROOT/docs/EMULATOR_TEST_RESULTS.txt"
"${ADB[@]}" exec-out run-as local.sucstudio.android cat files/qa-phone.png > "$ROOT/docs/QA_PHONE.png"

"${ADB[@]}" exec-out run-as local.sucstudio.android cat files/qa-captcha.png > "$ROOT/docs/QA_CAPTCHA.png"
