# 现有实验与协作阶段复核（2026-09-05）

本报告读取当前实验、评价、保存的工作流与图片，不启动服务，不重新运行实验，不修改用户评价。它是范围明确的审计，不是全模型排名。

## 当前存量

| 实验 ID | 总项目 | 完成 | 失败 | 待运行 |
| --- | ---: | ---: | ---: | ---: |
| 782d1062a7b6 | 360 | 360 | 0 | 0 |
| 7afc2592e560 | 738 | 488 | 7 | 243 |
| 11f3c7372e59 | 168 | 79 | 18 | 71 |
| c6def8c2ec49 | 168 | 21 | 0 | 147 |
| e01c126e052b | 8 | 8 | 0 | 0 |

完成项目共 956 项，记录中各有一张图片；其列出的图片文件均存在。这不是说已人工检查了 956 张图片。当前持久化自然语言评价只有 4 条，涉及 4 张图片，其中夜间矩阵 2 条、日常创作 2 条；对话中的偏好是另一种证据，不能虚构成库里的评价。

旧审计记录夜间矩阵完成 487 项，已不是最新快照。旧审计把固定 0.18 重绘、像素桥接和旧版本标签放进 contamination，这些本身不是错误，只是限制归因范围的实验条件。应保留旧产物，不因版本标签判废，也不因某个漂亮样本推广到所有场景。

## 亲自查看的样本与实际阶段

### [f17e29aa90fe](D:/Models/ai_comic/data/jobs/experiments/7afc2592e560/幽灵鲨/specter_underwater_memory/illustrious_2d_to_2d__幽灵鲨__specter_underwater_memory__compiled_native_v2__seed-42__f17e29aa90fe__01.png)

对应 Illustrious → NewBie，前后均为 2D 配方，交接重绘 0.18。源图仍在 [源阶段原图](D:/Models/ai_comic/runtime/ComfyUI/output/newbie_lab/f17e29aa90fe_source_base_00001_.png)。

观察：白底近景、头巾、手靠头部、平涂分块与十字架构图在 Illustrious 阶段已经存在。最终图改变了眼睛、脸部线条和一些纹理，但没有补回水下长廊，也没有改成闭眼触摸胸前十字架。保存的源阶段正向文本确实包含长廊和动作要求，不是完全没有收到描述。

结论范围：此配方值得保留作为用户喜欢的视觉方向，但不能把这一张当成环境或动作遵从成功。独特性至少部分来自源阶段，不应预先断言人物优势都由 NewBie 创造。仅查看源图与最终图，不能分离 NewBie 整图重采样和脸部细化各自的贡献。

### [c1f84e64ac02](D:/Models/ai_comic/data/jobs/experiments/7afc2592e560/W/w_minimal/illustrious_2d_to_2d__W__w_minimal__compiled_native_v2__seed-3407__c1f84e64ac02__01.png)

同一协作配方，源图在 [源阶段原图](D:/Models/ai_comic/runtime/ComfyUI/output/newbie_lab/c1f84e64ac02_source_base_00001_.png)。

观察：强烈的平涂、粗轮廓、暖色背景和近景构图已出现在源图；后续主要改变脸部表达。用户描述的“站在红色暮光里”不能仅凭近景头像证明完整实现，角色服装细节也不能由“canonical outfit”这类笼统标签保证。

### [6134483cec18](D:/Models/ai_comic/data/jobs/experiments/7afc2592e560/W/w_minimal/newbie_screenshot__W__w_minimal__compiled_native_v2__seed-42__6134483cec18__01.png)

这是 NewBie 单模型加 screenshot 配方，不是 Illustrious 协作。红灰大色块、细线条和克制构图本身已具有用户认可的表现力。它与 c1f84e64ac02 的 Seed 不同，不能把两者差异全归因于模型组合。

### [1b94b98eb2c4](D:/Models/ai_comic/data/jobs/experiments/7afc2592e560/超天酱/kangel_stream/z_2d_to_2d__超天酱__kangel_stream__compiled_native_v2__seed-42__1b94b98eb2c4__01.png)

用户评价原文：“由于缺失超天酱的信息，导致出现了‘类miku’的画，这是知识库问题。”原文以持久化记录为准。

[Z-Image 源图](D:/Models/ai_comic/runtime/ComfyUI/output/newbie_lab/1b94b98eb2c4_source_base_00001_.png) 已经出现青色双马尾、耳机式配件与双手握手机。最终图增加细节，没有改回用户要求的左手手机、右手 V 手势。

更早的输入端已有两项风险：无来源的蓝色双马尾角色模板；源阶段文本同时包含 “holding a phone with both hands” 和右手 V 手势。不能只归咎于最后的模型或 LoRA。本轮移除了当前 fallback 中这套无来源的超天酱外观/服装默认值，保留身份标签；没有凭记忆编造替代事实。这尚不证明角色准确性已经解决。

### [b701a07f4731](D:/Models/ai_comic/data/jobs/experiments/7afc2592e560/幽灵鲨/specter_underwater_memory/newbie_2d__幽灵鲨__specter_underwater_memory__compiled_native_v2__seed-3407__b701a07f4731__01.png)

用户评价：“人物比例失衡，场景单一没有风格”。实际图的主体在画面中很小，背景主要由大面积蓝色和重复立柱构成。它与 f17e29aa90fe 使用同一原始句子，但 Seed 不同：本图为 3407，f17e29aa90fe 的请求与两个已保存 KSampler 均为 42。先前写成两图同 Seed 是审计错误，本轮已逐项核对改正。两图还同时改变了路线与风格条件，只能作为不同配置的个例比较，不能隔离模型单一能力。

另查看了 [fafdecdfe9b6](D:/Models/ai_comic/data/jobs/one_sentence/w_(arknights)__standard_anime__seed-42__fafdecdfe9b6__01.png)，并读取了 674413d263dc 的评价。旧规则遗漏了“服装 OOC、画风违和”的问题摘录；这不是用户没有指出问题，而是索引漏报。

## 已据此实施的修正

- 分析页显示评价原文和原图入口，自动分类只作检索索引；旧评价现场重新索引但不改写原记录。
- 修正 OOC、违和、比例失衡、不匹配及明确否定问题的常见分类遗漏；未识别的句子保留，不虚构裁决。
- 评价覆盖按图片而非按任务计数；区分实验配方、执行路线与请求主模型。
- 取消“覆盖 60% 即可成为生产结论”的逻辑。目前生产发布入口拒绝仅依赖聚合统计的请求；带范围和同条件对照的结论发布仍待实现。当前没有已发布路由指导文件，现有路由规则和实验没有被改动。

## 对下一步管线工作的约束

1. 先检查原始意图到各阶段文本的角色、对象、左右手关系与背景保留；不能用一个模型重复处理错误输入来替代这一步。
2. 每个阶段的原图必须可比较、可保留。源图本身可能是更好的候选，不能强制经过 NewBie。
3. 检查要区分局部线条缺陷和整幅场景/动作不满足。0.18 的整图重绘不应承担补回整座长廊的默认职责。
4. 多模型迭代应允许停留、局部修改、退回文本/源阶段和回滚，而不是固定给每个模型永久分配人物或背景职责。
5. 新实验需要保留“源图单独交付”的对照，记录每一步的增益和退化；不得为增加文本数量而缩减已有路线。新增测试文本/形式仍按用户要求确认后再运行。

这些是后续实现与受控验证的要求，尚未宣称已完成。可复核数据由 [只读审计脚本](D:/Models/ai_comic/scripts/research/audit_existing_evidence.py) 生成在 [JSON 审计产物](D:/Models/ai_comic/work/contact_sheets/evidence-audit.json)。
