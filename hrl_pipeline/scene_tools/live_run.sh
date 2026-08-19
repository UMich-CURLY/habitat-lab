#!/bin/bash
# Cold single-run for the live click tool: build a 1-ep dataset from a click
# request, run the rule_yield expert with video, print a RESULT:<json> line the
# host server parses. ~1-2 min per call.
#   $1 = request json path (in-container)   $2 = tag (unique per request)
cd /habitat-lab && . activate habitat
export LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib MPLCONFIGDIR=/tmp/mpl
REQ="$1"; TAG="$2"; M=/habitat-lab/hrl_pipeline/manual
GZ=$M/_live_${TAG}.json.gz
STATUS=$M/_live_${TAG}.build.json

python hrl_pipeline/scene_tools/live_build.py "$REQ" "$GZ" "$STATUS" > $M/_live_${TAG}.build.log 2>&1
if ! python -c "import json,sys; sys.exit(0 if json.load(open('$STATUS')).get('ok') else 1)" 2>/dev/null; then
  python -c "import json; print('RESULT:'+json.dumps(json.load(open('$STATUS'))))" 2>/dev/null \
    || echo 'RESULT:{"ok":false,"errors":["build 失败，见 build.log"]}'
  exit 0
fi

rm -f hrl_pipeline/video_rule_yield/*.mp4 2>/dev/null
RULE_MODE=rule_yield RULE_VIDEO=1 RULE_CONFIG=social_nav/social_nav_hierarchical_v2.yaml \
  RULE_DATASET=$GZ RULE_EVALS=1 STEP_CAP=1200 EVAL_STATS=$M/_live_${TAG}.stats.json \
  python hrl_pipeline/scripted_hl_eval.py > $M/_live_${TAG}.eval.log 2>&1
mp4=$(ls -t hrl_pipeline/video_rule_yield/*.mp4 2>/dev/null | head -1)
[ -n "$mp4" ] && mv "$mp4" $M/vid_live_${TAG}.mp4

python -c "
import json
w=json.load(open('$STATUS')).get('warnings',[])
try:
    v=list(json.load(open('$M/_live_${TAG}.stats.json')).values())[0]
    su=v['social_nav_to_pos_success']; st=int(v['num_steps']); col=int(v['did_collide'])
    sc=int(v.get('robot_collisions.robot_scene_colls',0))
    verdict=('专家通过 OK' if su>=0.5 else ('撞停 (scene_col %d)'%sc if col else '超时卡住'))
    print('RESULT:'+json.dumps({'ok':True,'verdict':verdict,'success':su,'steps':st,
          'collide':col,'scene_col':sc,'warnings':w,'video':'vid_live_${TAG}.mp4'}))
except Exception as e:
    print('RESULT:'+json.dumps({'ok':False,'errors':['eval 失败: '+str(e)[:120]]}))
"
