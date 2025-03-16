locust -f src/hotelreservation_constant.py \
  --host http://127.0.0.1:5000 \
  --master \
  --headless \
  --users 3700 \
  -r 20 \
  -t 400s \
  --csv=locust_log \
  --expect-workers=8 \
  --master-bind-host=0.0.0.0  # 允许其他机器连接