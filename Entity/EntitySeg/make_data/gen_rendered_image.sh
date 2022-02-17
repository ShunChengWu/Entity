for d in /mnt/d/dataset/3RScan/data/3RScan/*/; do
 #echo "$d"
 f="$(basename -- $d)"
 b="$(dirname $d)"
 ./rio_renderer_render_all $b $f "sequence" "1"
done
