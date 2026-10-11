# 已视觉确认的物理公式合同

> 本合同只包含 UQ-03 第一门已视觉确认的公式与已批准实现语义。内部单位遵循 `unit_convention.md`；工程单位不得进入以下函数。

## 1. 区间参数

```text
p_center=(p_lower+p_upper)/2
p_radius=(p_upper-p_lower)/2
p=p_center+p_radius*z,  -1≤z≤1
```

- 输入：有限实数 `p_lower,p_upper,z`，且 `p_lower≤p_upper`、`z∈[-1,1]`。
- 输出单位：与 `p` 相同；半径非负。
- 单调性：固定中心和非负半径时，恢复值关于 `z` 单调不减。
- 推荐函数：`interval_center_radius`、`recover_interval_value`。
- 异常处理：端点颠倒、非有限值、负半径或越界 `z` 必须拒绝，不得静默交换端点。

## 2. 用户级预算集合

```text
𝒰_{Γ,n}={z_n∈[-1,1]^6:Σ_{k=1}^6|z_{n,k}|≤Γ}, 0≤Γ≤6
𝒰_Γ=∏_{n∈𝒩}𝒰_{Γ,n}
```

- 输入：每用户六维扰动、`Γ`。
- 输出单位：无量纲可行性结果。
- 语义：每个用户拥有独立预算；禁止把所有用户扰动合并后共用一个全局 `Σ|z|≤Γ`。
- 推荐函数：`is_user_budget_feasible`、`validate_user_disturbance`。
- 异常处理：维度不是6、`Γ∉[0,6]`、非有限扰动或分量越界时拒绝。

## 3. 上行速率

```text
R_n^u(I_n^u)=B_n^u log2(1+P_n^u h_n^u/(σ²+I_n^u))
```

- 输入：`bandwidth_hz,uplink_tx_power_w,uplink_channel_gain_linear,noise_power_w,uplink_interference_w`。
- 输出：`bit/s`。
- 单调性：关于带宽、发射功率和线性信道功率增益不减，关于噪声和干扰递减；速率下界取干扰上界，速率上界取干扰下界。
- 推荐函数：`uplink_rate_bit_per_s`。
- 异常处理：要求 `B>0,P≥0,h≥0,σ²≥0,I≥0,σ²+I>0`；结果必须有限且严格为正，否则依赖速率作除数的调用应拒绝。

## 4. 下行速率

```text
R_n^d(I_n^d)=B_n^d log2(1+P_n^d h_n^d/(σ²+I_n^d))
```

- 输入：`bandwidth_hz,downlink_bs_tx_power_w,downlink_channel_gain_linear,noise_power_w,downlink_interference_w`。
- 输出：`bit/s`。
- 单调性与异常要求：同上行；速率下界取干扰上界，速率上界取干扰下界。
- 推荐函数：`downlink_rate_bit_per_s`。

## 5. 本地点响应

```text
T_n^L(z_n)=C_n(z_n)/F_n^L(z_n)
E_n^L(z_n)=κ_n C_n(z_n)[F_n^L(z_n)]²
```

- 输入：`cpu_cycles`、`local_cpu_frequency_cycle_per_s`、`switching_capacitance_coefficient`。
- 输出：时延 `s`、能耗 `J`。
- 单调性：时延随 `C` 增大、随 `F_L` 减小；能耗随 `κ,C,F_L` 增大。
- 推荐函数：`local_latency_point_s`、`local_energy_point_j`。
- 异常处理：`C≥0,F_L>0,κ≥0` 且全部有限；输出必须有限非负。

## 6. 统一本地求界

推荐入口：`local_response_bounds_si`。

- `Γ=0`：以 `C^c,F_L^c` 调用本地点响应，lower=upper=point。
- `0<Γ<2`：在 `|z_C|≤1,|z_F|≤1,|z_C|+|z_F|≤Γ` 上，对上述两个已验证点响应执行确定性二维求界；不得调用 BP、GA 或随机采样代替求界。
- `Γ≥2`：

```text
T_lower=C_lower/F_upper
T_upper=C_upper/F_lower
E_lower=κ C_lower(F_lower)²
E_upper=κ C_upper(F_upper)²
```

- 输出：`local_latency_lower_s/local_latency_upper_s`、`local_energy_lower_j/local_energy_upper_j`。
- 异常处理：除点响应要求外，还需 `0≤Γ≤6`、端点有序；结果检查 `0≤lower≤upper<∞`。禁止对 `0<Γ<2` 静默套用闭式端点组合。

## 7. 缓存未命中时延

```text
T_n^dep=s_n/R_bh
```

- 输入：`dependency_size_bit,backhaul_rate_bit_per_s`。
- 输出：`s`。
- 单调性：随依赖数据量增大、随回传速率减小。
- 推荐函数：`cache_miss_latency_s`。
- 异常处理：`s_n≥0,R_bh>0` 且有限。

## 8. 边缘核心时延

```text
T̃_n^E(x,z_n)=D_n^in(z_n)/R_n^u(z_n)+C_n(z_n)/f_n+D_n^out(z_n)/R_n^d(z_n)
```

- 输入：输入/输出数据量 `bit`、上下行速率 `bit/s`、CPU量 `cycle`、边缘算力 `cycle/s`。
- 输出：`s`。
- 单调性：随 `D_in,D_out,C` 增大，随 `R_u,R_d,f` 减小；干扰通过降低速率增大核心时延。
- 推荐函数：`edge_core_latency_point_s`。
- 异常处理：数据量和CPU量非负，`R_u,R_d,f` 严格为正，全部有限；不接受缓存变量，不加入回源项。

## 9. 完整边缘时延

```text
T_n^E(x,z_n)=T̃_n^E(x,z_n)+(1-c_n)T_n^dep
```

- 输入：`edge_core_latency_s,cache_decision_c,cache_miss_latency_s`。
- 输出：`s`。
- 语义：`c_n=1` 时 total=core；`c_n=0` 时加回源时延。total用于用户响应、系统聚合和C7，不作时延BP标签。
- 推荐函数：`edge_total_latency_point_s`。
- 异常处理：`c_n` 必须严格为0或1；两个时延有限非负。

## 10. 移动终端通信能耗

```text
E_n^E(x,z_n)=P_n^u D_n^in(z_n)/R_n^u(z_n)+P_n^r D_n^out(z_n)/R_n^d(z_n)
```

- 输入：终端上行发射功率 `W`、终端接收功率 `W`、数据量 `bit`、速率 `bit/s`。
- 输出：`J`。
- 单调性：在功率和速率固定时随数据量增大；随干扰导致的速率下降而增大。
- 推荐函数：`edge_comm_energy_point_j`。
- 异常处理：功率和数据量非负，速率严格为正，全部有限；下行能耗必须使用 `terminal_rx_power_w (P_n^r)`，禁止传入 `downlink_bs_tx_power_w (P_n^d)` 代替。

## 11. 命名与结果隔离

- 时延BP标签：`edge_core_latency_*`。
- 聚合和C7：`edge_total_latency_*`。
- 离线高保真近似标签：`ga_ref_*`。
- 冻结BP原始输出：`raw_pred_*`。
- alpha修正后的正式在线输出：`corr_pred_*`；最终代表方案写入`final_*`。
- 不生成搜索后认证字段。
- 所有返回值先做有限性、非负性和边界顺序检查，再进入下游模块。
