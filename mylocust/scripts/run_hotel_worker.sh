locust -f src/hotelreservation_constant.py \
  --host http://127.0.0.1:5000 \
  --worker \
  --master-host=127.0.0.1
