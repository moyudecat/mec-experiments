# 实现口径决策日志

## 2026-08-03：第一批实现口径冻结

状态：已批准作为代码实现合同；三个冻结 Word 尚未同步修改。

| 决策 ID | 决策内容 | 影响文档 | 后续需要修改的论文位置 |
|---|---|---|---|
| DF-001 | 保留第五章 `Γ=1`、S1 和 Exp-5 的 `Γ∈{1,2,3,4,5}`。 | `frozen_model_spec.md`、`frozen_algorithm_spec.md`、`experiment_protocol.md`、`unresolved_questions.md` | 第三章 3.5.2；第四章 4.5.1；第五章 5.1.2、Exp-5。需把统一分段求界协议写入论文，并消除“正式实验均 Γ≥2”的表述。 |
| DF-002 | 本地响应统一分段求界：`Γ=0` 计算中心点；`0<Γ<2` 在仅含 `C,F_L` 的二维预算域内确定性求界；`Γ≥2` 使用第三章闭式边界。所有分支均不调用 BP 或 GA。 | `frozen_model_spec.md`、`frozen_algorithm_spec.md`、`experiment_protocol.md` | 第三章式(3-4)—(3-5)附近增加 `Γ=0` 与 `0<Γ<2` 的确定性求解定义；第四章 4.5.1 更新在线本地评价。 |
| DF-003 | 代码严格区分 `edge_core_*` 与 `edge_total_*`。前者不含缓存未命中时延并作为时延 BP 标签；后者为 core 边界加 `(1-c)T_dep`，用于系统聚合和 C7。 | `frozen_model_spec.md`、`frozen_algorithm_spec.md`、`code_naming_convention.md` | 第三章 3.4.2、3.5.3—3.5.4；第四章 4.2、4.3、4.5、4.7。建议论文符号或表格显式加入 core/total 语义。 |
| DF-004 | 代码内部统一 SI：bit、Hz、bit/s、cycle/s、cycle、W、线性信道增益、s、J；工程单位仅允许在配置加载边界显式转换。 | `frozen_model_spec.md`、`unit_convention.md`、`source_traceability.md` | 第三章主要符号表与物理模型各节；第五章表5-1—5-4补全单位、原始工程单位及转换说明。 |
| DF-005 | 网络结构、GA预算、BP损失权重、`F0/S0`、外层预算、认证容差和统计参数继续由相应 Preview 阶段冻结，本轮不赋正式数值。 | `frozen_algorithm_spec.md`、`experiment_protocol.md`、`unresolved_questions.md` | 第四章所有“第五章给出”的超参数位置；第五章表5-1—5-4及各阶段冻结表。 |
| DF-006 | Word 视觉公式必须人工复核；本轮只建立检查清单，不宣称复核完成。 | `formula_visual_review_checklist.md`、`unresolved_questions.md` | 第三章与第四章清单所列公式；复核后再决定是否需要修改 Word 排版或符号。 |

## 决策效力与边界

- 本日志优先约束后续代码命名、单位和分支行为，但不代表 `reference_docs/` 中 Word 已更新。
- 未在本日志决定的数值仍为 Preview 待冻结项。
- 后续若修改本批决策，必须新增日期条目，不覆盖历史决策。

## 2026-08-04：Exp-1 Preview预注册

- UQ-11形成Preview预注册版本：B1—B4、Reference-GA、Dense-LHS及唯一条件B5已登记，但最终GA预算仍须由完整Preview按预注册规则选择。
- UQ-12形成Preview预注册版本：科学阈值、经验参考包络、失败处理和最多2次重标规则已写入配置哈希；Dry-run结果不得用于判断配置是否通过。

## 2026-08-04：Exp-1 完整 Preview 选型结论

- 冻结配置哈希 `c465ed63bbf916d420eee8f74469bc7ab3d5476c47272dcfaf71b445ca7319d5` 下的 240 个任务、5760 次 GA 单种子运行和 240 次 Dense-LHS 均完成，主运行失败为 0。
- B1、B2、B3 未通过全部科学阈值；B4 通过全部科学阈值，并按预注册的效率优先规则成为唯一选中候选。
- Preview 阶段重标触发数为 0，B5 未执行且 `B5_required=false`。
- UQ-11 更新为“Preview 选型已完成，Formal 预算冻结仍开放”；本次不将 `ga_budget_frozen` 改为 true。
- UQ-12 更新为“Preview 阈值已执行确认，待同步 Word 正文”；禁止依据本次排名反向修改阈值。
- 影响文档：`docs/unresolved_questions.md`、本决策日志及后续 Exp-1 Formal 设计文档。后续需同步论文第四章标签生成预算、第五章 Exp-1 通过规则和结果口径；本轮未修改 Word 文档。
- 完整Preview尚未启动；不得依据后续BP误差反向修改本次选择规则或阈值。

## 2026-08-05：Exp-1 Formal 协议设计冻结与工程门禁

- `B4_formal_candidate_v1` 已按人口 64、50 代、5 个固定种子及既有算子参数冻结为待验证候选；它仍不是最终标签 GA 预算。
- Formal 独立样本固定为 90 个、四类任务共 360 条；样本与 Smoke、Preview 的完全重复和尺度归一近重复均为 0。
- Formal 原样继承 Preview 前预注册的科学阈值；本轮未调整阈值，也未产生 B4 的科学通过/失败判定。
- UQ-11 更新为“B4 Formal 候选已冻结，最终标签 GA 预算仍开放”，仅在完整 Formal 通过后关闭。
- UQ-12 更新为“Formal 继承阈值已冻结，最终状态待 Formal”，Word 正文同步暂缓到 Formal 结论后。
- 本轮只完成批量物理一致性、安全并行、分片 checkpoint 和 2 样本 Dry-run；完整 Formal、BP 训练和 BP 数据生成均未启动。
- 影响文档：`docs/unresolved_questions.md`、本决策日志及 `outputs/01_formal/exp1_label_budget_formal_design_20260805_210449/`。后续需在 Formal 结论后同步第四章标签生成预算和第五章 Exp-1 判定口径。

## 2026-08-06：Exp-1 完整 Formal 独立验证结论

- 唯一输入为 `exp1_label_budget_formal_design_20260805_210449`，使用串行 `actual_workers=1` 完成90/90样本、360/360任务和5040/5040正式执行单元；0 failed、0 stale、0恢复单元。
- Formal配置、科学阈值、样本、任务、B4、Reference-GA、Dense-LHS、代码树、物理合同和Preview决策哈希均保持冻结；三份Word未修改。
- B4的中位/Q95/最大相对漏界、Gamma×任务最大Q95及Q95相对响应标准差均通过；`max_relative_response_std=0.021744027472745037` 超过冻结上限 `0.02`。
- 最终三态判定严格为 `failed`，不得使用“接近通过”“基本通过”或人工通过等表述。
- UQ-11保持开放，B4不冻结为标签GA预算；UQ-12记录阈值已完整执行且B4未通过，Word同步仍待后续统一修改。
- 禁止进入BP训练和标签数据生成，禁止放宽阈值、删除困难Formal样本、自动增加预算或用Preview结论替代Formal结论。后续若继续，必须另立协议版本并重新审计。
- B4不是数学精确求界器；Reference-GA与Dense-LHS组成的包络仍仅为经验近似参考。Formal样本永久排除在后续BP训练、验证和测试之外。

## 2026-08-10：Adaptive-B4-Reference-v1 独立 Confirmation 结论

- 旧 B4 Formal 的 `failed` 结论及全部产物保持不变；本次另立 `Adaptive-B4-Reference-v1` 协议版本，没有重跑或改写旧 Formal。
- Confirmation 使用60个独立样本、240个任务，以 `actual_workers=1` 串行完成3360/3360单元：B4 1200、Reference-GA 1920、Dense-LHS 240；0 failed、0 stale、0恢复单元。
- 预注册六项科学指标全部通过：中位漏界0、Q95漏界0.002299282608415397、最大漏界0.01297367303027114、Gamma×任务最大Q95为0.01273182224721543、Q95有效相对响应标准差0.002933249715130606、最大有效相对响应标准差0.007037180451836547。
- 仅2/240任务触发升级，升级率0.008333333333333333；数值、缓存、重现性、样本隔离、种子命名空间和检查点完整性审计均通过。
- 最终 `confirmation_decision=passed`，`Adaptive-B4-Reference-v1` 验证并冻结。UQ-11关闭；UQ-12更新为“Adaptive阈值已确认，待同步 Word 正文”。
- 允许进入 BP 数据集设计；在该设计及门禁完成前，不允许运行 BP 标签生成或训练 BP。本轮未修改三份 Word，也未启动五候选认证。
- 证据：`outputs/01_formal/adaptive_b4_reference_confirmation_20260810_000412/`。

## 2026-08-10：DualBP数据集协议与标签管线Smoke冻结

- 只读使用最新版`第四章_双BP误差修正_校订三线表版.docx`和`第五章_实验设计参考稿_误差修正_校订三线表版.docx`；共享只读OOXML提取成功，科学冲突为0。Word COM导出因外部占用超时，记录`word_sync_status=temporarily_locked`，未修改或关闭Word。
- 冻结`DualBP-Dataset-v1`：600个唯一输入，Train/Validation/Calibration/Test为420/60/60/60；每角色在5个Gamma×6类样本的30层内完全平衡，角色、历史数据和Smoke之间exact/near重复及种子冲突均为0。
- Time-BP输入冻结为19维，Energy-BP输入冻结为17维，Gamma进入两个网络；时延标签只含edge core，能耗标签只含终端通信能耗。
- 60样本标签Smoke以workers=1完成240/240任务：B4 1200次，1个任务自然升级，Reference-GA 8次、Dense-LHS 1次；1209/1209单元完成，failed/stale/recovery均为0，全部工程审计通过。
- `label_smoke_decision=passed`，数据协议状态为`frozen`，600 manifest状态为`frozen_inputs_only`。只开放下一轮600样本正式GA标签生成；BP训练、alpha计算、Exp-2和Exp-3仍禁止。
- UQ-11保持关闭；UQ-12更新为科学阈值与最新版Word规则已只读确认、Word正文修改待后续统一完成。

## 2026-08-03：UQ-03 第一门视觉复核结论

- 以 Word 只读导出的 XPS/PDF 和页面PNG为视觉证据，完成物理点响应与统一本地求界所需12项公式复核。
- 结论：没有未解决的物理公式冲突；本地上下界未颠倒；终端下行接收能耗使用 `P_n^r`。
- 处置：将核心时延Markdown符号由误写的 `T_n^{E,x}` 修正为Word中的 `T̃_n^E(x,z_n)`；Word不修改。
- 效力：UQ-03第一门关闭，可进入“物理点响应＋统一本地求界”代码阶段；Softplus、C7、鲁棒目标等后续门禁仍保留。
- 证据：`docs/formula_visual_review_report.md` 与 `outputs/05_audit/formula_visual_review_gate1_20260803_214226/render_evidence/`。

## 2026-08-10：600样本正式DualBP标签生成结论

- 严格读取 `bp_dataset_protocol_20260810_122228` 冻结的600样本manifest，未新增、删除、替换或调整任何样本及角色。
- 以 `workers=1` 串行执行 `Adaptive-B4-Reference-v1`：600/600样本、2400/2400任务和12000次B4种子运行全部完成；17个自然升级任务完整执行136次Reference-GA种子运行和17次Dense-LHS。
- 数值、Schema、特征合同、角色、checkpoint和复现审计全部通过；最终回归测试为217项通过。
- `formal_label_generation_decision=passed`，`formal_bp_dataset_status=labeled_and_frozen`，从下一轮起允许进入DualBP训练协议冻结及Train/Validation训练阶段。
- `alpha_calibration_allowed=false`，Calibration与Test继续封存；Exp-2、Exp-3仍未开放。本轮未训练BP、未计算alpha、未修改Word。
- 证据：`outputs/01_formal/bp_dataset_labels_formal_20260810_132254/`。

## 2026-08-10：Edge CPU参数与Time-BP Dataset V2输入冻结

- 作者在输入域审计后正式决定：`f_min^E=1.5e9 cycle/s`、`f_max^E=5.0e9 cycle/s`、`F0=default_F_max=20.0e9 cycle/s`；来源类型明确为`author_decision`，不是历史配置或Word既有数值。
- 既有600条5GHz Time标签正式定义为`f_max`上界切片并按hash复用；Energy继续绑定v1的600条标签，不复制、不重算。
- `time_bp_dataset_v2`按每个base三条记录冻结：5GHz上界锚点、role内low确定性分层点、role内mid-high确定性分层点；拆分单位为base group。
- V2 manifest为1800行，角色1260/180/180/180；1200条新增记录排队，对应下一轮2400个Time边界任务。全部输入域、角色隔离、复用和hash审计通过。
- 当前只开放V2新增Time标签生成；Time/Energy训练、alpha、Exp-2与Exp-3仍未开放。Word正文待后续统一同步。
- 证据：`outputs/05_audit/time_bp_dataset_v2_design_20260810_191130/`。
