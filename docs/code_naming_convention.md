# 代码与结果字段命名约定

## 1. 通用规则

- Python/JSON/CSV 字段统一使用小写 `snake_case`。
- 下界后缀 `_lower`，上界 `_upper`，中心 `_center`，半径 `_radius`。
- 点响应后缀 `_point`；不得用含义模糊的 `_min/_max` 代替物理上下界字段。
- 用户级字段包含 `user_id`；系统聚合字段使用 `system_` 前缀。
- 内部数值均为 SI；需要对外展示工程单位时创建单独展示字段，不改变物理字段值。

## 2. 核心时延与完整时延

| 语义 | 统一字段示例 | 是否含缓存未命中项 | 用途 |
|---|---|---:|---|
| 边缘核心点时延 | `edge_core_latency_point_s` | 否 | 真实物理点响应 |
| 边缘核心下/上界 | `edge_core_latency_lower_s`、`edge_core_latency_upper_s` | 否 | 内层求界结果、时延 BP 标签 |
| 边缘核心中心/半径 | `edge_core_latency_center_s`、`edge_core_latency_radius_s` | 否 | 时延 BP 直接监督量 |
| 缓存未命中附加时延 | `cache_miss_latency_s` | 不适用 | `T_dep=s/R_bh` |
| 边缘完整下/上界 | `edge_total_latency_lower_s`、`edge_total_latency_upper_s` | 是 | 用户响应、系统聚合、C7 |
| 边缘完整中心/半径 | `edge_total_latency_center_s`、`edge_total_latency_radius_s` | 是 | 诊断与结果输出 |

必须满足：

```text
edge_total_latency_*_s
= edge_core_latency_*_s + (1-cache_decision)*cache_miss_latency_s
```

不得把 `edge_total_*` 用作时延 BP 标签，也不得用 `edge_core_*` 直接计算 C7。

## 3. 本地响应与能耗

```text
local_latency_point_s
local_latency_lower_s / local_latency_upper_s
local_energy_point_j
local_energy_lower_j / local_energy_upper_j
edge_comm_energy_point_j
edge_comm_energy_lower_j / edge_comm_energy_upper_j
edge_comm_energy_center_j / edge_comm_energy_radius_j
```

当前能耗口径是移动终端通信能耗，因此使用 `edge_comm_energy_*`，不使用容易误解为边缘服务器能耗的 `edge_energy_*`。

## 4. 离线参考、原始预测与修正预测字段

离线高保真近似标签以 `ga_ref_` 开头；冻结BP原始结果以 `raw_pred_` 开头；alpha修正后的正式在线结果以 `corr_pred_` 开头。例如：

```text
ga_ref_edge_core_latency_lower_s
raw_pred_edge_core_latency_lower_s
corr_pred_edge_core_latency_lower_s
corr_pred_edge_total_latency_lower_s
corr_pred_edge_comm_energy_lower_j
corr_pred_system_latency_center_s
corr_pred_robust_latency_objective
corr_pred_possibility
corr_pred_constraint_violation
corr_pred_feasible
corr_pred_ideal_distance
```

`corr_pred_edge_total_*` 必须由修正后的core加解析缓存项得到，不是网络直接输出。

## 5. 最终结果字段

正式在线优化不执行搜索后真实认证，也不生成 `certified_*`。最终代表方案从 `corr_pred` 可行Rank-1前沿按冻结规则选择，并写入 `final_*`；离线 `ga_ref_*` 不得改变最终方案。

## 6. 决策与单位后缀

```text
offload_decision_m
cache_decision_c
edge_cpu_allocation_cycle_per_s
input_data_bit / output_data_bit
cpu_cycles
uplink_interference_w / downlink_interference_w
local_cpu_frequency_cycle_per_s
uplink_rate_bit_per_s / downlink_rate_bit_per_s
```

文件级 schema 的数组维度、用户排序、浮点序列化与版本号仍需在实现阶段单独冻结。
