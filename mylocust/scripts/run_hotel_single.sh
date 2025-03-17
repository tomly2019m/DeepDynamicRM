locust -f src/hotelreservation_constant.py \
  --host http://127.0.0.1:5000 \
  --headless \
  --users 300 \
  -r 50 \
  -t 400s \
  --csv=locust_log