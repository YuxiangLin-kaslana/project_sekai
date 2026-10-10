#!/usr/bin/env bash
# 部署 Project Sekai：systemd 服务 + Caddy 反向代理。需要 root：
#   sudo bash deploy/install.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CADDYFILE=/etc/caddy/Caddyfile

echo "== 1/3 安装并启动 sekai.service"
install -m 644 deploy/sekai.service /etc/systemd/system/sekai.service
systemctl daemon-reload
systemctl enable --now sekai
systemctl restart sekai

echo "== 2/3 Caddy 配置"
if grep -q "sekai.yuxianglin.xyz" "$CADDYFILE"; then
  echo "Caddyfile 里已有 sekai.yuxianglin.xyz，跳过追加"
else
  backup="$CADDYFILE.bak.$(date +%Y%m%d-%H%M%S)"
  cp "$CADDYFILE" "$backup"
  cat deploy/Caddyfile.sekai >> "$CADDYFILE"
  if ! caddy validate --config "$CADDYFILE" --adapter caddyfile; then
    echo "Caddyfile 校验失败，已恢复备份 $backup" >&2
    cp "$backup" "$CADDYFILE"
    exit 1
  fi
  echo "已追加（备份：$backup）"
fi
systemctl reload caddy

echo "== 3/3 检查"
for _ in 1 2 3 4 5 6 7 8 9 10; do
  curl -fsS http://127.0.0.1:8095/healthz >/dev/null 2>&1 && break
  sleep 1
done
echo -n "后端：  "; curl -fsS http://127.0.0.1:8095/healthz; echo
echo -n "Caddy： "; curl -fsS -H "Host: sekai.yuxianglin.xyz" http://127.0.0.1/healthz; echo
echo "完成：https://sekai.yuxianglin.xyz"
