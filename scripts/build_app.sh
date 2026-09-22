#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."

app_bundle="文档问答.app"
app_contents="$app_bundle/Contents"
mkdir -p "$app_contents/MacOS" "$app_contents/Resources"

scripts/build_app_icon.sh >/dev/null
cp assets/AppIcon.icns "$app_contents/Resources/AppIcon.icns"
cp assets/Info.plist "$app_contents/Info.plist"

clang -fobjc-arc -framework Cocoa -framework Foundation -framework AppKit \
  scripts/docqa_launcher.m -o "$app_contents/MacOS/启动文档问答"
chmod +x "$app_contents/MacOS/启动文档问答"

plutil -lint "$app_contents/Info.plist" >/dev/null
echo "$app_bundle"
