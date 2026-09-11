#!/bin/bash
# Build, sign, notarize and package bacteriaapp as a distributable DMG.
#
# Prerequisites, both one-time:
#   1. A "Developer ID Application" certificate in the login keychain.
#      Xcode > Settings > Accounts > Manage Certificates > + > Developer ID Application
#   2. Notary credentials stored under the profile name below:
#      xcrun notarytool store-credentials AgarScope \
#        --apple-id <your-apple-id> --team-id 2R786KVV88
#
# Usage: ./release.sh [version]     (default 1.0)

set -euo pipefail

VERSION="${1:-1.0}"
TEAM_ID="2R786KVV88"
NOTARY_PROFILE="AgarScope"
VOLNAME="AgarScope $VERSION"
DEST="$HOME/Desktop/AgarScope-$VERSION.dmg"

cd "$(dirname "$0")"
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

IDENTITY=$(security find-identity -v -p codesigning \
    | grep "Developer ID Application" | head -1 | sed 's/.*"\(.*\)"/\1/')
if [ -z "$IDENTITY" ]; then
    echo "No 'Developer ID Application' certificate found in the keychain." >&2
    echo "Create one: Xcode > Settings > Accounts > Manage Certificates > + " >&2
    exit 1
fi
echo "==> Signing identity: $IDENTITY"

echo "==> Building Release (universal)"
xcodebuild -project bacteriaapp.xcodeproj -scheme bacteriaapp -configuration Release \
    -derivedDataPath "$WORK/dd" \
    ARCHS="arm64 x86_64" ONLY_ACTIVE_ARCH=NO \
    MARKETING_VERSION="$VERSION" \
    CODE_SIGN_STYLE=Manual \
    CODE_SIGN_IDENTITY="$IDENTITY" \
    DEVELOPMENT_TEAM="$TEAM_ID" \
    CODE_SIGN_INJECT_BASE_ENTITLEMENTS=NO \
    OTHER_CODE_SIGN_FLAGS="--timestamp --options=runtime" \
    build > "$WORK/build.log" 2>&1 || { tail -40 "$WORK/build.log"; exit 1; }

APP="$WORK/dd/Build/Products/Release/bacteriaapp.app"
codesign --verify --deep --strict --verbose=2 "$APP"

# These two checks must not pipe into `grep -q`: it closes the pipe on the first
# match, the producer dies of SIGPIPE, and pipefail then reports the pipeline as
# failed exactly when the pattern was found -- inverting both tests.
ENTS=$(codesign -d --entitlements - --xml "$APP" 2>/dev/null || true)
case "$ENTS" in
    *get-task-allow*)
        echo "get-task-allow is still present; notarization would be rejected." >&2
        exit 1 ;;
esac

# Notarization requires a hardened runtime.
SIGINFO=$(codesign -d --verbose=2 "$APP" 2>&1 || true)
case "$SIGINFO" in
    *runtime*) ;;
    *) echo "hardened runtime missing:" >&2; echo "$SIGINFO" >&2; exit 1 ;;
esac

# The app is notarized and stapled BEFORE it goes into the disk image. A ticket
# stapled to the .dmg alone stops protecting the app the moment it is dragged
# out to /Applications: Gatekeeper then has to ask Apple over the network, and
# a machine that is offline on first launch fails. Stapling the bundle itself
# makes it self-contained.
echo "==> Notarizing the app (this takes a few minutes)"
ditto -c -k --keepParent "$APP" "$WORK/app.zip"
xcrun notarytool submit "$WORK/app.zip" --keychain-profile "$NOTARY_PROFILE" --wait
xcrun stapler staple "$APP"
xcrun stapler validate "$APP"

echo "==> Staging disk image"
STAGE="$WORK/stage"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/bacteriaapp.app"
ln -s /Applications "$STAGE/Applications"
cp "dist/BACA SAYA - Cara Instal.txt" "$STAGE/" 2>/dev/null || true
xattr -cr "$STAGE/bacteriaapp.app"

rm -f "$DEST"
hdiutil create -volname "$VOLNAME" -srcfolder "$STAGE" -ov -format UDZO -fs HFS+ "$DEST" > /dev/null

echo "==> Notarizing the disk image"
xcrun notarytool submit "$DEST" --keychain-profile "$NOTARY_PROFILE" --wait

echo "==> Stapling"
xcrun stapler staple "$DEST"

echo "==> Verifying as Gatekeeper would"
MP=$(hdiutil attach "$DEST" -nobrowse -readonly | tail -1 | cut -f3-)
spctl -a -vvv -t install "$MP/bacteriaapp.app"
hdiutil detach "$MP" > /dev/null

echo
echo "Done: $DEST"
