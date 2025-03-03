# 获取所有 docker 服务的 ID
SERVICE_IDS=$(docker service ls -q)
# 获取所有服务的 CPU 限制（NanoCPUs）并求和，如果服务没有设置，则使用 0
TOTAL_NANO=$(docker service inspect $SERVICE_IDS | jq '[.[] | .Spec.TaskTemplate.Resources.Limits.NanoCPUs // 0] | add')

# 将 nanocpus 转换为 CPU 数量（1 CPU = 1000000000 nanoCPUs）
TOTAL_CPUS=$(echo "scale=2; $TOTAL_NANO/1000000000" | bc)

echo "所有服务的 CPU 限制总和："
echo "NanoCPUs: $TOTAL_NANO"
echo "CPUs: $TOTAL_CPUS"