#!/bin/bash
# Install llm-tool-capability from GitHub
INSTALL_DIR="$HOME/.local/share/llm-tool-proxy"
mkdir -p "$INSTALL_DIR"
git clone https://github.com/xtmic/tool-cal.git "$INSTALL_DIR" 2>/dev/null || true
cd "$INSTALL_DIR"
npm install && npm run build

# Copy our files
cp "$(dirname "$0")/custom-proxy.js" "$INSTALL_DIR/"
cp "$(dirname "$0")/preload.cjs" "$HOME/.config/llm-tool-proxy/"

# Copy env if not exists
[ -f "$HOME/.config/llm-tool-proxy/env" ] || cp "$(dirname "$0")/env.example" "$HOME/.config/llm-tool-proxy/env"

# Install systemd service
mkdir -p "$HOME/.config/systemd/user"
cp "$(dirname "$0")/llm-tool-proxy.service" "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now llm-tool-proxy

echo "Done. Edit ~/.config/llm-tool-proxy/env with your API key."
