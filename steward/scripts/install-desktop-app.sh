#!/usr/bin/env bash
# Install ~/Desktop/Steward.app that launches Steward + Ego Harness and opens the control panel.
# The app is an AppleScript applet: macOS shows the ~/Documents consent prompt for applets, while a bundle whose
# executable is a shell script is denied that access without any prompt.
set -euo pipefail

STEWARD="$(cd "$(dirname "$0")/.." && pwd)"
ASSET="$STEWARD/scripts/desktop"
APP="${STEWARD_DESKTOP_APP:-$HOME/Desktop/Steward.app}"
BUNDLE_ID="com.deepseek.steward.desktop-launcher"
LAUNCH_SCRIPT="$STEWARD/scripts/launch-desktop.sh"
PLIST_BUDDY=/usr/libexec/PlistBuddy

[[ -f "$ASSET/AppIcon.icns" ]] || { echo "missing $ASSET/AppIcon.icns" >&2; exit 1; }
[[ -x "$LAUNCH_SCRIPT" ]] || chmod +x "$LAUNCH_SCRIPT"

APPLESCRIPT_SRC="$(mktemp -t steward-launcher).applescript"
trap 'rm -f "$APPLESCRIPT_SRC"' EXIT
cat > "$APPLESCRIPT_SRC" <<APPLESCRIPT
on run
	set launchScript to "$LAUNCH_SCRIPT"
	set logDir to (POSIX path of (path to home folder)) & "Library/Logs/Steward"
	try
		do shell script "mkdir -p " & quoted form of logDir & " && head -c 1 " & quoted form of launchScript & " >/dev/null"
	on error errMsg
		do shell script "open 'x-apple.systempreferences:com.apple.preference.security?Privacy_FilesAndFolders'"
		display dialog "Steward 无法读取启动脚本：" & return & errMsg & return & return & "请在 系统设置 → 隐私与安全性 → 文件和文件夹 中为 Steward 打开「文稿」文件夹，然后重新双击图标。" with title "Steward" buttons {"好"} default button "好" with icon caution
		return
	end try
	do shell script quoted form of launchScript & " >> " & quoted form of (logDir & "/app-wrapper.log") & " 2>&1 &"
end run
APPLESCRIPT

rm -rf "$APP"
/usr/bin/osacompile -o "$APP" "$APPLESCRIPT_SRC"
cp "$ASSET/AppIcon.icns" "$APP/Contents/Resources/applet.icns"
rm -f "$APP/Contents/Resources/Assets.car"
"$PLIST_BUDDY" -c "Delete :CFBundleIconName" "$APP/Contents/Info.plist" 2>/dev/null || true
"$PLIST_BUDDY" -c "Set :CFBundleName Steward" "$APP/Contents/Info.plist"
"$PLIST_BUDDY" -c "Add :CFBundleDisplayName string Steward" "$APP/Contents/Info.plist"
"$PLIST_BUDDY" -c "Add :CFBundleIdentifier string $BUNDLE_ID" "$APP/Contents/Info.plist" 2>/dev/null \
  || "$PLIST_BUDDY" -c "Set :CFBundleIdentifier $BUNDLE_ID" "$APP/Contents/Info.plist"
"$PLIST_BUDDY" -c "Add :NSAppleEventsUsageDescription string 'Steward 用系统对话框选择场景并显示启动通知。'" "$APP/Contents/Info.plist" 2>/dev/null || true
xattr -cr "$APP" 2>/dev/null || true
codesign --force --deep --sign - "$APP"
touch "$APP"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP" 2>/dev/null || true
echo "Installed $APP"
