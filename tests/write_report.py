"""Write reports from measured artifacts; no hand-entered performance claims."""
import hashlib
import json
import platform
import subprocess
import sys

import cv2
import matplotlib
import numpy as np

from p1.io import ROOT, common, dump


def text(path, value):
    path.write_text(value, encoding="utf-8")


def interval_text(segments):
    return "; ".join(f"[{s['start']:.3f},{s['end']:.3f}) {s['kind']}/{s['motion_type']}" for s in segments)


def main():
    d = json.loads((ROOT / "artifacts/comparison.json").read_text(encoding="utf-8"))
    params = d["parameters"]
    dump(ROOT / "parameters.json", params)
    source_docs = [
        "C:/Users/Administrator/Desktop/视觉插件/video-keyframes-project-baseline-2026-11-15.md",
        "C:/Users/Administrator/Desktop/video-keyframes-breakthrough-execution-v0.2.md"]
    provenance = {"date_local": "2026-11-16", "timezone": "Asia/Taipei",
                  "python": sys.version, "platform": platform.platform(),
                  "numpy": np.__version__, "opencv": cv2.__version__, "matplotlib": matplotlib.__version__,
                  "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0],
                  "design_documents_read": [{"path": p, "sha256": common.sha256_file(p)} for p in source_docs],
                  "frozen_components": []}
    for name, original in [("common.py", "scripts/common.py"), ("select_v.py", "scripts/select_v.py"), ("experiment.json", "config/experiment.json")]:
        src = ROOT.parent / "real-observation-v2" / original
        frozen = ROOT / "vendor" / name
        provenance["frozen_components"].append({"original": str(src), "frozen": str(frozen),
                                                "sha256": common.sha256_file(frozen),
                                                "matches_original_now": common.sha256_file(src) == common.sha256_file(frozen)})
    dump(ROOT / "artifacts/provenance.json", provenance)
    analyses = {
        "hard_cut": "首轮 hard_cut_mad=0.14 高于实测 raw MAD=0.126338，候选已定位但返回 unknown。开发校准为0.10后变成 likely_edit；baseline 无改动。该用例参与调参，不能算留出集验证。区间为切换的采样括界，右支持帧 PTS=3.0在半开区间右端，不表示3.0这一帧未保留。",
        "dissolve": "静态端点的线性混合拟合误差低，锚点趋势成立。此例的混合机制与检测假设吻合，属于容易的合成正例；曝光反例表明该证据不足以可靠判定实际溶解。",
        "pan": "相邻配准后有效重叠残差低，先于编辑判断输出 continuous_motion。仅支持单平面平移连续性；没有语义场景结论，也没有虚拟全景累计。",
        "door_parallax": "两个深度平面的运动不服从同一平移，残差及混合误差不满足可信编辑条件，返回 uncertain/unknown。只是带遮挡柱的两层几何代理，未验证真实手持穿门视频。",
        "local_object": "输出一个stable区间，保留10条局部变化候选。这里未自动识别人、搬运或对象关系；这些名称来自生成器真值。更大前景遮挡仍可能触发全局候选。",
        "ppt": "36秒低变化内容被分为[0,30)、[30,36)两个stable覆盖块，30秒不是镜头边界。16秒新增小圆点在局部候选阈值下漏检（local_event_count=0），因此只通过稳定性验收，不代表PPT信息采样完整。",
        "flash_failure": "单帧闪光被5fps粗扫命中并返回uncertain，无likely_edit。反例名称保留以标识挑战用途，此轮该检查通过；错开采样相位的一帧闪光仍可能完全漏掉。",
        "exposure_failure": "仍然失败：曝光渐变被误判为likely_edit。曝光变化同样近似L/R线性插值，配准高响应但位移接近0，没有非零位移的连续运动支持。不能把低配准解释率+锚点趋势当溶解证明。",
        "blur_unknown": "首轮全图Laplacian被底部清晰水印抬高，模糊过程产生两处误报。改为中心80%区域的清晰度，并保留全图值作诊断，此轮返回一个uncertain/unknown区间；该修改受此开发用例启发，字幕位于中心等情况仍未知。",
        "edge_dissolve": "整段缺少稳定端点，返回unknown而非强行补造锚点。最后源帧及其持续时间仍留在区间与帧表中。",
        "pts_offset_vfr": "36张粗扫帧逐张与独立全帧解码像素比较，source_frame与ffprobe pts_time对应均通过。起点3秒、非均匀采样间隔、末帧持续时间被保留。该检查只验证此视频的时间对应，不覆盖所有容器/编解码器；原始VFR baseline的名义标签未当作已校正PTS。"}
    case_table, baseline_table = [], []
    for row in d["cases"]:
        case, evaluation = row["case"], row["evaluation"]
        cid = case["id"]
        out = ROOT / "artifacts/results" / cid
        result = json.loads((out / "intervals.json").read_text(encoding="utf-8"))
        native = json.loads((out / "baseline_v_native_rate.json").read_text(encoding="utf-8"))
        mapping = native["source_mapping_audit"]
        if isinstance(mapping, list):
            corrected = [mapping[i]["source_t"] for i in native["budgets"]["5"]["sample_indices"]]
            native_desc = str(corrected)
        else:
            corrected = None
            native_desc = "source PTS未解算；nominal=" + str(native["budgets"]["5"]["times"])
        active = [s for s in result["segments"] if s["kind"] != "stable"]
        status = "通过" if evaluation["pass"] else "失败"
        case_table.append(f"| {cid} | {status} | {interval_text(evaluation['intervals'])} | {evaluation['local_event_count']} |")
        baseline_table.append(f"| {cid} | {len(active)} | {row['baseline_same_pool_peaks']} | {row['baseline_same_pool_budget5']} | {native_desc} |")
        links = [m for m in result["pair_metrics"][1:] if m["raw_mad"] >= params["activity_mad"]]
        stats = {"active_link_count": len(links),
                 "median_raw_mad": float(np.median([m["raw_mad"] for m in links])) if links else None,
                 "median_overlap_residual_mad": float(np.median([m["residual_mad"] for m in links if m["residual_mad"] is not None])) if links else None,
                 "motion_supported_links": sum(m["motion_supported"] for m in links)}
        dump(out / "parameters.json", params)
        text(out / "CASE_REPORT.md", f"""# {cid}：{status}

输入：`{case['path']}`。SHA256：`{case['sha256']}`。

生成条件：{case['description']}。来源：{case['provenance']}。

验收条件：{evaluation['rule']}。

分段：{interval_text(evaluation['intervals'])}。

参数完整记录于 `parameters.json`；本例没有单独参数覆盖。输入读取频率5fps，尺寸320像素宽，最后源帧额外保留。

失败／限制／错误例子：{analyses[cid]}

指标摘要（仅描述本例）：`{json.dumps(stats, ensure_ascii=False)}`。

局部事件候选数：{evaluation['local_event_count']}。这些候选没有语义标签。

对照：同池原V变化峰 `{row['baseline_same_pool_peaks']}`；同池原V预算5的实际选帧 `{row['baseline_same_pool_budget5']}`；原配置原抽帧V预算5的实际选帧（可审计时显示源时间）`{native_desc}`。不足5张时保留原函数行为。

核查文件：`intervals.json`（源帧、锚点、质量、趋势、全部邻帧指标）、`evaluation.json`（生成真值与检查结果）、`diagnostics.png`、`metrics.csv`、`anchors.csv`、`contact_sheet.png`、`baseline_v.json`、`baseline_v_native_rate.json`、`native_v_pool/`。真值只由测试器读取，检测器不读取。
""")
    report = """# P1_REPORT

日期：2026-11-16（Asia/Taipei）。交付：独立、离线的视觉连续性切分原型。工作目录 `D:/codex/p1-transition`。v0.1 与 v0.2 设计文档均已审读；用户补充明确“先管算法”，因此没有等待不存在的正式插件仓库，也未改生产接口。

## 已验证

六类最小开发用例的验收检查均通过，另有5个挑战/时间核查用例；总计10/11项检查通过，曝光渐变检查失败。这个计数是下表明确条件的通过数，不是识别精度。所有样本都是合成开发材料，硬切和清晰度参数曾用其中样本校准，尚无留出集结果。

实现了视频与时间有序帧清单两种输入；输出stable/transition/uncertain、左右锚点、continuous_motion/likely_edit/unknown、质量信号、证据索引与时间。没有标定confidence概率。低纹理、模糊、稳定锚点不足、采样大缺口允许unknown。

运动补偿采用Hanning窗相位相关平移估计，将左帧与全1支持掩膜一同warp到右帧，排除越界及插值边缘，仅在有效重叠区求残差。motion_supported同时要求响应、重叠、残差比及非零位移，低匹配不会单独推出溶解。双锚点趋势与线性混合误差是启发式证据，曝光反例说明仍有混淆。

11项代码契约检查通过，包括平移方向/掩膜、低重叠拒绝、采样缺口、单帧unknown、帧清单原时间和额外元数据、非法时间拒绝、参数校验、采样上限显式失败、全部产物区间覆盖/证据可追溯、冻结文件hash、与原V独立CLI的一致性。见 `artifacts/contract_tests.log`。

非零起点VFR用例的36张帧与独立全帧解码及ffprobe时间表逐张核对通过，未用i/fps替代源PTS。全部视频输出区间无空洞/重叠，逐帧保留source_frame、pts、pts_time、时间基与原视频引用；模糊帧未删除。

## 组件审查与接口

| 组件 | 冻结来源与接口 | 本轮使用/边界 |
|---|---|---|
| 视频元信息 | real-observation-v2/scripts/common.py: `probe_video(path)->dict` | 直接复用；duration仅作信息，P1结束使用末帧PTS+持续时间 |
| PGM读取 | `read_pgm(path)->(w,h,bytes)` | 帧清单PGM输入、原始V缩略图复用 |
| 差分 | `mean_abs_diff(a:bytes,b:bytes)->float` | V使用原有步长抽样MAD；未替换成新评分 |
| 视觉baseline | select_v.py: `diffs_for(frames)`、`nms_top(frames,diffs,n,radius)` | 字节相同副本，预算3/5/8、NMS=4s、90百分位峰诊断规则全保留 |
| 原有批量抽帧 | `extract_candidates(video,out_dir,fps,width,height,thumb_width,thumb_height)` | 用原配置直接运行作为原始baseline；其t=i/fps为名义标签，不能用于P1源时间 |
| 单帧读取 | `extract_frame(video,t,out_path,width=640)` | 已审查，返回文件不返回实际PTS，未作为P1时间来源 |
| 时间校验经验 | apply_pts_correction.py / verify_candidate_pts.py | 已审查；借用“核对帧号、像素与PTS”的验证方法，不硬编码其项目特定时间公式到P1 |
| 镜头/配准/区域模块 | 当前可用源码未发现独立实现；用户确认正式插件尚不存在 | P1新写候选检测、4x6网格区域MAD、相位配准；不声称复用了不存在的接口 |

旧模块原文件没有修改。`vendor/`为完整字节副本，`artifacts/provenance.json`保存来源与SHA256。没有把文件内的其他实验指令扩展为此次任务。

## 输入输出与复现

运行方式、帧清单格式见 `README.md`。`parameters.json`为全部生效参数。视频解析要求严格递增、非重复的PTS与有效末帧持续时间；缺失时明确报错，不猜FPS。帧清单要求显式end及源时间t，不排序、不合成时间。

输出区间为半开[start,end)，单位为源时间秒；候选区间通常由变化前后两张支持帧的PTS括界，右支持帧可能恰处于end。帧列表和evidence仍保留这张帧及其源时间，锚点也允许位于区间之外。稳定段仅表示当前采样尺度未产生全局变化候选。30秒强制分块是覆盖策略，不是编辑边界。

每条用例在 `artifacts/results/<id>/CASE_REPORT.md` 提供输入、参数、完整分段、错误例子与失败分析；`intervals.json`保存全部机器可读数据；`diagnostics.png`与CSV提供原始MAD、补偿残差、相位响应、有效重叠、区域变化比例、位移、左右锚点差异/混合误差和清晰度。`contact_sheet.png`可人工核对。测试视频及hash清单为 `artifacts/videos/manifest.json`。

## 逐用例结果

| 用例 | 检查 | 完整分段（秒） | 局部候选数 |
|---|---|---|---|
""" + "\n".join(case_table) + """

## 基线比较

V原本输出选帧与变化峰，不输出stable/transition或运动类别，因此没有捏造V的分类F1，也不能据本表宣布新算法优于V。

对照分两种输入：同池V复用P1的5fps源帧，仅适配为96x54缩略图；原配置V则真实调用冻结extract_candidates，保留0.5fps、560x315展示图、96x54 PGM、NMS=4秒、预算3/5/8、原先低变化平顶峰与预算不足时的行为，未加P1末帧。原配置的名义时间保留；CFR用例额外用独立全帧缩略图像素核对19+40*i映射并增加source_t，不覆盖旧标签。VFR原配置的源映射未解决，明确保留未知，不拿其名义时间评定位误差。

| 用例 | P1非稳定区间数 | 同池V诊断峰（源秒） | 同池V预算5选帧（源秒） | 原配置V预算5选帧（源秒，VFR除外） |
|---|---|---|---|---|
""" + "\n".join(baseline_table) + """

同池V在低变化材料上也可能按原规则输出平顶峰；这些是原诊断行为，没有删去以美化P1。原配置与同池比较的采样频率和缩放流程不同，因此只做透明的输出比较，未做速度、预算收益或精度排名。独立原V CLI一致性测试核对选帧时间、帧号、分数及诊断峰。

## 失败

1. 曝光渐变仍被误判为likely_edit，见exposure_failure区间[2,4)。线性混合趋势不能排除光照，配准无可信非零位移也不能推出编辑。这是已执行反例，必须保留。
2. PPT的极小新增圆点没有进入局部事件候选（虽未误切镜头）；不具备信息完整性保证。
3. 初版硬切仅输出unknown；初版清晰度被锐利水印抬高，模糊过程误报编辑。首轮结果、日志和core.py完整留在 `artifacts/audits/initial/`。修改记录：hard_cut_mad 0.14→0.10；清晰度由全图改为中心80%区域并同时保留全图值。末尾缺锚点的候选延至源end，避免孤立尾帧被无依据称作稳定。没有修改baseline、生成真值或验收条件。

## 未知与限制

尚未验证真实自然视频、真实穿门、复杂视差、旋转/变焦、滚动快门、低照、字幕覆盖中心、压缩噪声和纯颜色变化。输入指标为灰度，等亮异色变化可能漏检；未实现特征/单应性补偿。稳定锚点可能在动态内容中找不到，保留unknown；语义上是否同一房间/事件不可据这些指标确认。

5fps未命中的短时事件不能由锚点/合并恢复。清晰度只是纹理代理，不是可读性保证。所有样本由同一开发流程构造，不是独立盲测；11项检查不是样本总体准确率。没有性能压测、实时延迟测量、识别精度评估或因果效果实验。

读取器先扫全部帧时间元数据，再顺序解码全视频，只保留选中的缩略帧，默认最多5000张，超过则报错。因此低粗扫FPS不等于低解码成本；此实现不能宣称实时或适合长视频内存开销。当前FFmpeg 9.0.2已测试，滤镜选帧表达式使用平衡树与文件参数，兼容性尚未覆盖旧版本。原始VFR baseline名义标签的精确源映射仍未知。

## 下一步

先冻结当前参数并采集独立自然视频小集，人工标出视图连续性区间，尤其曝光/穿门/动态前景/短时事件；报告定位误差与unknown覆盖，禁止据开发检查数宣称精度。优先加入光照变化证据或收紧likely_edit的弃权条件，再检验是否损害真实溶解召回。需要时补充特征几何一致性，不把低匹配单独解释为溶解。随后再评估按需局部原视频重读及P2；对象、关系、身份索引、VLM、完整场景图、重要性模型和因果解释仍在规划中，不是本次交付。
"""
    text(ROOT / "P1_REPORT.md", report)
    # Concise README keeps executable entry points separate from the report.
    text(ROOT / "README.md", """# P1 双锚点 + 运动补偿原型

独立离线算法。先读 `P1_REPORT.md`：六类开发验收通过，曝光渐变仍误报，不能据此宣布泛化精度或实时性。

已测试环境：Python 3.12.14、FFmpeg/ffprobe 9.0.2（PATH）、NumPy/OpenCV/Matplotlib/Pillow；完整版本见environment-lock.txt。本目录已有可运行.venv，不包含在交付zip中。

## 运行（PowerShell）

```powershell
Set-Location D:\\codex\\p1-transition
& .\\.venv\\Scripts\\python.exe -m p1 --video artifacts/videos/pan.mkv --out artifacts/demo_pan
& .\\.venv\\Scripts\\python.exe -m p1 --frames examples/frames.json --out artifacts/demo_frames
& .\\.venv\\Scripts\\python.exe -m tests.run_suite
& .\\.venv\\Scripts\\python.exe -m unittest tests.test_contracts -v
& .\\.venv\\Scripts\\python.exe -m tests.write_report
```

新环境用已安装Python创建 `.venv`，再运行 `.venv\\Scripts\\python.exe -m pip install -r requirements.txt`。默认测试运行生成报告并打印所有失败；`python -m tests.run_suite --strict`在任一挑战失败时返回非零（目前曝光反例使其退出1），不会隐去失败。

## 输入与接口

CLI二选一：`--video PATH`或`--frames MANIFEST.json`；另需`--out DIR`。可用`--coarse-fps 5`、`--max-frames 5000`或`--params parameters.json`配置。完整参数在parameters.json。

已解码帧清单：

```json
{"end": 8.5, "frames": [
  {"path": "f0.png", "t": 7.125, "source_frame": 142, "pts": 7125},
  {"path": "f1.png", "t": 7.381, "source_frame": 147, "pts": 7381}
]}
```

路径相对清单，t必须为严格递增源时间，不会自动排序；end必须大于最后t。额外元数据保留。大于1秒的采样间隔默认标为unknown。示例使用不规则源时间和3张实际PNG，验证此入口；这不是视频精度评估。

Python：`read_video(path,fps,width,max_frames)`或`read_manifest(path,width)`返回`frames,source`，`analyze(frames,source,Parameters())`返回结果。内存帧约定gray为同尺寸uint8二维NumPy图，t为源秒。

## 输出

`intervals.json`包含源hash、time_base、帧号与原PTS、全部生效参数、segments、左右锚点、运动类型、质量、时间证据、局部变化候选、逐帧质量、逐相邻对指标、双锚点曲线。

`diagnostics.png`、`metrics.csv`、`anchors.csv`为曲线；`contact_sheet.png`为带源时间的拼图；`baseline_v.json`为同采样池的冻结原V。测试集另外保存原配置原抽帧的`baseline_v_native_rate.json`与native_v_pool。完整baseline比较见P1_REPORT及artifacts/comparison.json。

所有区间为源时间的半开[start,end)，候选右支持帧可能恰在end；证据表仍完整保存。stable只指该采样尺度未发现全局变化，continuous_motion只表示平移模型有支持，likely_edit并非编辑事实，unknown为正常结果。

测试素材均为合成。`artifacts/videos/manifest.json`为输入清单；`artifacts/results/<id>/CASE_REPORT.md`逐条记录分段、错误例子、参数、失败分析；`artifacts/provenance.json`与vendor/记录未改动的对照来源。没有接入VLM、场景图、对象重要性或因果解释。
""")


if __name__ == "__main__":
    main()
