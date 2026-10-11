# 规范来源追踪表

| 规范条目 | 来源文件 | 来源章节 | 是否明确 | 是否待确认 |
|---|---|---|---|---|
| 端—边—云场景、二元卸载、正交接入、独立用户预算 | 第三章_冻结模型.docx | 3.1、3.3、3.4.3 | 是 | 否 |
| 决策变量 `m,c,f` | 第三章_冻结模型.docx | 3.2 | 是 | 否 |
| 六类区间参数 | 第三章_冻结模型.docx；decision_log.md | 3.1、表3-2；DF-004 | 是 | 数值与来源待 Preview |
| 中心—半径与标准化扰动 | 第三章_冻结模型.docx；formula_visual_review_report.md | 3.1；第一门报告 | 是 | 第一门已视觉确认 |
| `Γ` 预算集合 | 第三章_冻结模型.docx；decision_log.md | 式(3-1)；DF-001/DF-002 | 是 | Word 同步待完成 |
| 上下行香农速率 | 第三章_冻结模型.docx；formula_visual_review_report.md；unit_convention.md | 3.3；第一门报告；SI规则 | 是 | 已视觉确认 |
| 本地时延与能耗点响应 | 第三章_冻结模型.docx；formula_visual_review_report.md；unit_convention.md | 3.4.1；第一门报告 | 是 | 已视觉确认 |
| 本地分段确定性求界协议 | 第三章_冻结模型.docx；decision_log.md；formula_visual_review_report.md | 3.5.2，式(3-4)—(3-5)；DF-001/DF-002；第一门报告 | 是 | `Γ≥2`公式已视觉确认；新增分段文字待Word同步 |
| 边缘核心/完整时延及回源项 | 第三章_冻结模型.docx；decision_log.md；code_naming_convention.md | 3.4.2；DF-003 | 是 | Word 同步待完成 |
| 终端通信能耗口径 | 第三章_冻结模型.docx；unit_convention.md | 式(3-2)、3.4.2；SI规则 | 是 | 参数来源待 Preview |
| 用户与系统区间聚合 | 第三章_冻结模型.docx | 式(3-10)—(3-12) | 是 | 否 |
| 两个鲁棒目标 | 第三章_冻结模型.docx | 式(3-13) | 是 | 参数值待确认 |
| C1—C7 | 第三章_冻结模型.docx | 式(3-17) | 是 | 数值容差待确认 |
| C7 可能度与 `CV` | 第三章_冻结模型.docx | 式(3-14)—(3-16) | 是 | 退化/边界容差待确认 |
| 离线 LHS 与四数据角色隔离 | 第四章_双BP误差修正_校订三线表版.docx；第五章_实验设计参考稿_误差修正_校订三线表版.docx | 4.2.1、4.2.5；5.1.2、Exp-2 | 是 | 600规模、420/60/60/60和种子空间已冻结 |
| 四类内层 GA 边界任务 | 第四章_冻结算法.docx | 4.2.2—4.2.3 | 是 | GA 超参数待冻结 |
| 时延 BP 输入输出和标签 | 第四章_双BP误差修正_校订三线表版.docx；feature_schema.json | 4.2、4.3.1 | 是 | 19维字段已冻结；网络结构待Preview |
| 能耗 BP 输入输出和标签 | 第四章_双BP误差修正_校订三线表版.docx；feature_schema.json | 4.2、4.3.2 | 是 | 17维字段已冻结；网络结构待Preview |
| Softplus 中心—半径机制 | 第四章_冻结算法.docx | 4.3.3—4.3.4 | 是 | `s_q` 具体规则待冻结 |
| 时延上界低估损失 | 第四章_冻结算法.docx | 4.3.5 | 是 | 权重/容差待冻结 |
| C1—C6 修复 | 第四章_冻结算法.docx | 4.4.4、表4-3 | 部分 | 阈值、并列、哈希待确认 |
| `ga_ref_*`、`raw_pred_*`、`corr_pred_*`与`final_*` | 第四章_双BP误差修正_校订三线表版.docx；code_naming_convention.md | 4.3、4.5、4.7 | 是 | 标签Schema v1已冻结；预测数组Schema待训练阶段 |
| 外层约束 NSGA-II | 第四章_冻结算法.docx | 4.6 | 是 | 数值预算待冻结 |
| 修正预测Rank-1前沿与最终代表方案 | 第四章_双BP误差修正_校订三线表版.docx | 4.7.1—4.7.2 | 是 | 具体外层容差待Preview |
| 取消搜索后真实认证 | 第四章_双BP误差修正_校订三线表版.docx；第五章_实验设计参考稿_误差修正_校订三线表版.docx | 4.5—4.8；Exp-6—Exp-8 | 是 | 否 |
| Exp-1—Exp-8 证据链与顺序 | 第五章_实验设计参考稿.docx | 总体表、第五章各实验、实施顺序 | 是 | 各通过阈值待冻结 |
| Smoke/Preview/Formal | 第五章_实验设计参考稿.docx | 5.1.3 | 是 | 阶段产物 schema 待确认 |
| S1—S5 | 第五章_实验设计参考稿.docx；decision_log.md | 5.1.2；DF-001/DF-002 | 部分 | `Γ=1` 已决定保留；`F0/S0` 待 Preview |
| 统计种子与显著性检验 | 第五章_实验设计参考稿.docx | 5.1.3 | 部分 | α、效应量与种子表待冻结 |
| 禁止按正式排名反向调参 | 第五章_实验设计参考稿.docx | 5.1.3、实施顺序 | 是 | 否 |
| outputs 结果隔离要求 | 第四章_冻结算法.docx；第五章_实验设计参考稿.docx | 4.7.6、4.8.5；5.1.3 | 部分 | 目录编号映射待确认 |
| 内部 SI 单位与工程单位转换 | decision_log.md；unit_convention.md | DF-004；全部章节 | 是 | 论文参数表同步待完成 |
| 实现字段命名 | decision_log.md；code_naming_convention.md | DF-003；全部章节 | 是 | 论文接口表同步待完成 |
| Word 公式人工视觉复核门禁 | decision_log.md；formula_visual_review_checklist.md；formula_visual_review_report.md | DF-006；第一门12项 | 是 | 第一门已关闭；后续未勾选公式待复核 |
