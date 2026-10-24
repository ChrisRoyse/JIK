#!/usr/bin/env bash
# Final verification that aiwonder is free of kernelweights work. Patterns live in variables so this script's own
# command line never matches them.
P1="kw-venv"; P2="gpuq_"; P3="board_proxy"; P4="kw_chain"; P5="gm_aiw"; P6="kernelweights-m3"
echo "== $(date -u +%FT%TZ) verify_free =="
echo "containers: [$(docker ps --format '{{.Names}}' | tr '\n' ' ')]"
nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,persistence_mode --format=csv,noheader
echo "compute apps: [$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')]"
for pat in "$P1" "$P2" "$P3" "$P4" "$P5"; do
  for p in $(pgrep -f -- "$pat"); do
    [ "$p" = "$$" ] && continue
    echo "LEFTOVER $pat pid $p: $(ps -o user=,cmd= -p "$p" | cut -c1-120)"
    sudo -n kill -KILL "$p" && echo "  killed $p"
  done
done
echo "enabled unit files matching gm-/kernelweights:"; systemctl list-unit-files --no-pager 2>/dev/null | grep -iE "^(gm-|kernelw)" | grep -E "enabled" || echo "  none enabled"
for u in $(systemctl list-unit-files --no-pager 2>/dev/null | grep -iE "^(gm-|kernelw)" | grep -E "enabled" | awk '{print $1}'); do sudo -n systemctl disable --now "$u" && echo "  disabled $u"; done
echo "running gm-/kernelweights services: [$(systemctl list-units --type=service --state=running --no-pager 2>/dev/null | grep -iE 'gm-|kernelw' | awk '{print $1}' | tr '\n' ' ')]"
free -g | head -2; uptime
ps -eo pid,user,pcpu,rss,etime,cmd --sort=-pcpu | head -4 | cut -c1-140
echo "== done =="
