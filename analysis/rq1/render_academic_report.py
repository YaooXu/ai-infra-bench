#!/usr/bin/env python3
"""Render the cutoff-consistent RQ1 findings as a Chinese academic PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    LongTable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs/assets/rq1"
SUMMARY = ROOT / "analysis/rq1/summary.json"
DEFAULT_OUTPUT = ROOT / "output/pdf/vllm_rq1_workload_report_2026-07-31_zh.pdf"

NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#2E86AB")
CYAN = colors.HexColor("#68C3D4")
ORANGE = colors.HexColor("#F18F01")
GREEN = colors.HexColor("#3A7D44")
RED = colors.HexColor("#C14953")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5E6C7B")
PALE = colors.HexColor("#EEF3F7")
RULE = colors.HexColor("#CBD5DF")
WHITE = colors.white


def register_fonts() -> None:
    regular = "/System/Library/Fonts/Supplemental/Songti.ttc"
    bold = "/System/Library/Fonts/STHeiti Medium.ttc"
    sans = "/System/Library/Fonts/STHeiti Light.ttc"
    # The TTC defaults are Songti Black and traditional-Chinese Heiti. Pin the
    # intended Simplified Chinese faces explicitly for long-form readability.
    pdfmetrics.registerFont(TTFont("RQSong", regular, subfontIndex=6))
    pdfmetrics.registerFont(TTFont("RQHei", bold, subfontIndex=1))
    pdfmetrics.registerFont(TTFont("RQSans", sans, subfontIndex=1))
    pdfmetrics.registerFontFamily("RQSong", normal="RQSong", bold="RQHei", italic="RQSong", boldItalic="RQHei")
    addMapping("RQSong", 0, 0, "RQSong")
    addMapping("RQSong", 1, 0, "RQHei")


class AcademicDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(id="Cover", frames=frame, onPage=draw_cover_page),
                PageTemplate(id="Body", frames=frame, onPage=draw_body_page),
            ]
        )

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            level = getattr(flowable, "toc_level", None)
            if level is not None:
                text = flowable.getPlainText()
                key = f"section-{self.seq.nextf('section')}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=level, closed=False)
                self.notify("TOCEntry", (level, text, self.page, key))


def draw_cover_page(canvas, doc) -> None:
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(BLUE)
    canvas.rect(0, 0, 20 * mm, height, fill=1, stroke=0)
    canvas.setFillColor(ORANGE)
    canvas.rect(20 * mm, 0, 2.5 * mm, height, fill=1, stroke=0)
    canvas.setFillColor(colors.Color(1, 1, 1, alpha=0.08))
    canvas.circle(width - 22 * mm, height - 22 * mm, 45 * mm, fill=1, stroke=0)
    canvas.circle(width - 4 * mm, 22 * mm, 36 * mm, fill=1, stroke=0)
    canvas.restoreState()


def draw_body_page(canvas, doc) -> None:
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.45)
    canvas.line(doc.leftMargin, height - 15 * mm, width - doc.rightMargin, height - 15 * mm)
    canvas.setFont("RQSans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, height - 11.5 * mm, "AI Infra Bench - vLLM RQ1 实证研究")
    canvas.drawRightString(width - doc.rightMargin, height - 11.5 * mm, "数据截止 2026-07-31 UTC")
    canvas.line(doc.leftMargin, 14 * mm, width - doc.rightMargin, 14 * mm)
    canvas.drawString(doc.leftMargin, 9.5 * mm, "Research Report / 2026-08-13")
    canvas.drawRightString(width - doc.rightMargin, 9.5 * mm, str(doc.page))
    canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    result = {}
    result["Title"] = ParagraphStyle(
        "Title",
        parent=base["Title"],
        fontName="RQHei",
        fontSize=28,
        leading=37,
        textColor=WHITE,
        alignment=TA_LEFT,
        spaceAfter=7 * mm,
    )
    result["Subtitle"] = ParagraphStyle(
        "Subtitle",
        fontName="RQSans",
        fontSize=13,
        leading=20,
        textColor=colors.HexColor("#D7E8F3"),
        spaceAfter=4 * mm,
    )
    result["CoverMeta"] = ParagraphStyle(
        "CoverMeta",
        fontName="RQSans",
        fontSize=9.5,
        leading=16,
        textColor=colors.HexColor("#D7E8F3"),
    )
    result["AbstractTitle"] = ParagraphStyle(
        "AbstractTitle",
        fontName="RQHei",
        fontSize=11,
        leading=16,
        textColor=NAVY,
        spaceAfter=2 * mm,
    )
    result["Abstract"] = ParagraphStyle(
        "Abstract",
        fontName="RQSong",
        fontSize=9.5,
        leading=16,
        textColor=INK,
        alignment=TA_JUSTIFY,
        firstLineIndent=2 * 9.5,
    )
    result["H1"] = ParagraphStyle(
        "H1",
        fontName="RQHei",
        fontSize=17,
        leading=24,
        textColor=NAVY,
        spaceBefore=5 * mm,
        spaceAfter=3.2 * mm,
        keepWithNext=True,
    )
    result["H2"] = ParagraphStyle(
        "H2",
        fontName="RQHei",
        fontSize=12.5,
        leading=18,
        textColor=BLUE,
        spaceBefore=3.5 * mm,
        spaceAfter=2 * mm,
        keepWithNext=True,
    )
    result["Body"] = ParagraphStyle(
        "Body",
        fontName="RQSong",
        fontSize=9.4,
        leading=16.2,
        textColor=INK,
        alignment=TA_JUSTIFY,
        firstLineIndent=2 * 9.4,
        spaceAfter=2.1 * mm,
        wordWrap="CJK",
    )
    result["BodyNoIndent"] = ParagraphStyle(
        "BodyNoIndent",
        parent=result["Body"],
        firstLineIndent=0,
    )
    result["Bullet"] = ParagraphStyle(
        "Bullet",
        parent=result["BodyNoIndent"],
        leftIndent=5 * mm,
        firstLineIndent=-3.5 * mm,
        bulletIndent=1 * mm,
        spaceAfter=1.3 * mm,
    )
    result["Callout"] = ParagraphStyle(
        "Callout",
        fontName="RQHei",
        fontSize=10.2,
        leading=17,
        textColor=NAVY,
        alignment=TA_LEFT,
    )
    result["Caption"] = ParagraphStyle(
        "Caption",
        fontName="RQSong",
        fontSize=8.2,
        leading=12.5,
        textColor=MUTED,
        alignment=TA_JUSTIFY,
        spaceBefore=1.5 * mm,
        spaceAfter=3 * mm,
    )
    result["TableCaption"] = ParagraphStyle(
        "TableCaption",
        fontName="RQHei",
        fontSize=8.8,
        leading=13,
        textColor=NAVY,
        spaceBefore=2.5 * mm,
        spaceAfter=1.5 * mm,
    )
    result["Small"] = ParagraphStyle(
        "Small",
        fontName="RQSong",
        fontSize=7.4,
        leading=11.5,
        textColor=MUTED,
        wordWrap="CJK",
    )
    result["Reference"] = ParagraphStyle(
        "Reference",
        fontName="RQSong",
        fontSize=8.1,
        leading=13.2,
        textColor=INK,
        leftIndent=5 * mm,
        firstLineIndent=-5 * mm,
        spaceAfter=1.5 * mm,
        wordWrap="CJK",
    )
    return result


def heading(text: str, style, level: int):
    p = Paragraph(text, style)
    p.toc_level = level
    return p


def P(text: str, style):
    return Paragraph(text, style)


def bullet(text: str, st):
    return Paragraph(f"• {text}", st["Bullet"])


def callout(text: str, st):
    table = Table([[Paragraph(text, st["Callout"])]], colWidths=[164 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.6, BLUE),
                ("LINEBEFORE", (0, 0), (0, -1), 4, ORANGE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def styled_table(data, widths, st, header=True, font_size=7.7, alignments=None):
    cells = []
    for r, row in enumerate(data):
        style = st["Small"]
        cells.append([Paragraph(str(value), style) for value in row])
    table = LongTable(cells, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    rules = [
        ("FONTNAME", (0, 0), (-1, -1), "RQSong"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [WHITE, colors.HexColor("#F7F9FB")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        rules.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "RQHei"),
            ]
        )
    if alignments:
        for col, alignment in alignments.items():
            rules.append(("ALIGN", (col, 1 if header else 0), (col, -1), alignment))
    table.setStyle(TableStyle(rules))
    return table


def figure(path: str, caption: str, number: int, st, max_width=164 * mm, max_height=150 * mm):
    file = FIGURES / path
    if not file.exists():
        raise FileNotFoundError(file)
    with PILImage.open(file) as im:
        width, height = im.size
    scale = min(max_width / width, max_height / height)
    image = Image(str(file), width=width * scale, height=height * scale)
    image.hAlign = "CENTER"
    return KeepTogether(
        [
            image,
            Paragraph(f"<b>图 {number}.</b> {caption}", st["Caption"]),
        ]
    )


def add_figure_page(story, path, caption, number, st, note=None, max_height=178 * mm):
    story.append(PageBreak())
    story.append(figure(path, caption, number, st, max_height=max_height))
    if note:
        story.append(P(note, st["Small"]))


def build_story(st, summary):
    story = []

    story.extend(
        [
            Spacer(1, 42 * mm),
            P("vLLM 的真实维护 workload", st["Title"]),
            P("需求增长、维护 capacity 与 AI inference benchmark 的实证基础", st["Subtitle"]),
            Spacer(1, 18 * mm),
            P("AI Infra Bench", st["CoverMeta"]),
            P("RQ1 Empirical Study Report", st["CoverMeta"]),
            Spacer(1, 7 * mm),
            P("观察截止：2026-07-31 23:59:59 UTC", st["CoverMeta"]),
            P("报告版本：2026-08-13", st["CoverMeta"]),
            Spacer(1, 55 * mm),
            P("基于 49,925 个 GitHub artifact、205,998 条 conversation comment、131,473 个 submitted review 和 122,470 条可映射 inline review comment 的仓库级 census。", st["CoverMeta"]),
            NextPageTemplate("Body"),
            PageBreak(),
        ]
    )

    story.append(heading("摘要", st["H1"], 0))
    abstract = (
        "本研究回答 AI Infra Bench 的第一个研究问题：vLLM 的公开维护 workload 如何演化、包含什么工作，以及新增需求与可观察维护 capacity 是否匹配。"
        "研究以 Simon Mo 提供的 vLLM Fivetran 快照为基础，通过 GitHub API 将数据补充至 2026-07-31，并构建 cutoff-consistent 合并数据库。"
        "结果显示，2026 年 1-7 月相对 2025 年，月均新增 PR 增长 96.8%，但月均 merge 仅增长 35.3%，活跃 roster reviewer 增长 6.9%，每个新增 PR 的 submitted review 下降 37.7%。"
        "截至 cutoff，open PR 已达 4,194 个，其中 81.2% 没有 submitted roster review。外部贡献者提交了 75.1% 的 human-authored PR，而可观察 merge gatekeeping 仍主要由具有写权限的 actor 完成。"
        "技术 workload 以 distributed execution、attention/kernel、V1 runtime、KV cache、quantization、MoE、speculative decoding 与异构硬件集成为核心。"
        "这些结果表明，一个能够回答“LLM 可解决多少真实 AI inference engineering workload”的 benchmark，不能只从 merged feature PR 采样；它必须覆盖 bug diagnosis、review-heavy integration、open/closed-unmerged work、性能验证、verifier construction 和 specialist hardware tasks。"
    )
    story.append(P(abstract, st["Abstract"]))
    story.append(Spacer(1, 3 * mm))
    story.append(P("<b>关键词：</b>AI inference；vLLM；repository mining；maintenance workload；code agents；benchmark", st["BodyNoIndent"]))
    story.append(Spacer(1, 5 * mm))
    story.append(callout("核心发现：vLLM 当前最显著的维护压力不是 issue 数量失控，而是 PR 输入速度与可见 review/merge capacity 之间持续扩大的缺口。", st))
    story.append(Spacer(1, 6 * mm))
    story.append(heading("目录", st["H1"], 0))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC1", fontName="RQHei", fontSize=9.5, leading=16, leftIndent=0, textColor=NAVY),
        ParagraphStyle("TOC2", fontName="RQSong", fontSize=8.5, leading=14, leftIndent=6 * mm, textColor=INK),
    ]
    story.append(toc)
    story.append(PageBreak())

    story.append(heading("1. 研究问题与贡献", st["H1"], 0))
    story.append(P("本研究的目标不是挑选 200 个 PR，而是为 benchmark 建立一个可辩护的 workload denominator。核心研究问题为：", st["Body"]))
    story.append(callout("RQ1：vLLM 的可观察公开维护 workload 如何随时间变化，它由哪些工作构成，新增需求与可见 maintainer capacity 之间是什么关系？", st))
    story.append(Spacer(1, 3 * mm))
    story.append(P("RQ1 分解为五个互补维度：", st["BodyNoIndent"]))
    for text in [
        "Intake 与 throughput：issue、PR、closure 和 merge 的月度变化；",
        "Responsiveness 与 backlog：固定窗口 response、time-to-event 与 cutoff queue；",
        "Review burden 与 capacity：活跃 reviewer、reviewer-days、review density 与集中度；",
        "Work composition：issue intent、PR work type、subsystem、topic 与 hardware；",
        "Ownership 与 contributor lifecycle：谁实现、谁 review、谁 merge，以及 external contributor 的进入与返回。",
    ]:
        story.append(bullet(text, st))
    story.append(P("贡献有三点。第一，使用完整月窗口和 cutoff-consistent 状态重建，避免把累计 GitHub 总量误当趋势。第二，明确区分 any-human response 与 May-18 roster response，并避免把当前权限倒推为历史 maintainer 身份。第三，将 workload 结构直接映射到 benchmark 的 task contracts、sampling strata、environment 和 verifier 需求。", st["Body"]))

    story.append(heading("2. 数据、总体与可复现性", st["H1"], 0))
    story.append(heading("2.1 数据来源与观察窗口", st["H2"], 1))
    story.append(P("合并数据库以 Simon Mo 的 <i>vLLM GitHub Gym: vLLM GitHub Snapshot (Fivetran)</i> [1] 为基础，并补充 GitHub GraphQL/REST 数据至 2026-07-31 23:59:59 UTC。数据库通过 release 发布；未压缩文件 SHA-256 为 <font name='RQSans'>2ac86507a95f9b8785e6ce0bbf2745e3fbba67c747e37b54020a7e57ce80f8b5</font>。", st["Body"]))
    table1 = [
        ["数据对象", "规模", "分析角色"],
        ["Canonical artifact", "49,925", "issue 与 PR 总体"],
        ["Issue", "16,990", "用户需求、bug、design 与处置"],
        ["Pull request", "32,935", "实现、review、merge 与 task source"],
        ["Conversation comment", "205,998", "non-author textual response"],
        ["Submitted review", "131,473", "formal review burden"],
        ["可映射 inline review comment", "122,470", "code-level review burden"],
        ["PR-commit association", "77,682", "patch 与 source-frame reconstruction"],
        ["Default-branch commit", "19,416", "delivery 与 direct-commit audit"],
    ]
    story.append(P("表 1. 合并数据库的核心分析总体", st["TableCaption"]))
    story.append(styled_table(table1, [57 * mm, 31 * mm, 76 * mm], st, alignments={1: "RIGHT"}))
    story.append(P("报告比较三个窗口：launch-2024、2025，以及 2026 Jan-Jul 七个完整月份。所有当前状态与 queue 指标定义在 cutoff。固定 7/14/30/90/180 天 outcome 只纳入已经获得完整 follow-up 的 cohort，未解决 artifact 作为 right-censored observation 保留。", st["Body"]))

    story.append(heading("2.2 关键 operational definitions", st["H2"], 1))
    for text in [
        "Any-human response：artifact author 之外的 human conversation comment、submitted review 或 inline review comment。",
        "Roster response：同一事件限制在 2026-05-18 collaborator permission roster；这是 capacity sensitivity，而非历史 maintainer 身份。",
        "PR response clock：ready PR 从 creation 开始，draft PR 从首次 ready-for-review 开始；从未 ready 的 draft 不进入 risk set。",
        "Merged PR：observed merge event 与 materialized cutoff merged_at 的并集。279 个 PR 有 merge event 但通用 artifact state 为 CLOSED；168 个只有 merged_at、缺少 retained merge event，其 actor 保持 unknown。",
        "Review-intensive：至少 3 个 review-head rounds、至少 10 次 roster review submission，或 review span 至少 14 天。该指标是 sampling proxy，不是工时。",
    ]:
        story.append(bullet(text, st))

    story.append(heading("2.3 数据质量与复现验证", st["H2"], 1))
    story.append(P("数据库的 19 个 release validation 全部通过。分析器额外检查 canonical table 数量、ID 唯一性、PR database ID 到 artifact ID 的映射、cutoff 后事件、merged reconstruction 与跨表总体一致性。两次 clean run 得到逐字节一致的 summary.json、19 张图和 57 张 aggregate CSV。", st["Body"]))
    story.append(P("保留并公开说明的异常包括：443 个 closed artifact 缺少 close-history row、2 个 current-state/history disagreement、2 个 review 缺少 event time、21 条 inline comment 无法映射至 canonical PR、88 个 default-branch commit 无关联 PR。对 cutoff 后 representation 不确定性的排除敏感性显示，分类 share 最大变化为 issue intent 0.75、PR work type 0.28、hardware 0.28、topic 1.00 个百分点。", st["Body"]))

    story.append(heading("3. Demand、throughput 与 backlog", st["H1"], 0))
    compare = [
        ["月均指标", "2025", "2026 Jan-Jul", "变化"],
        ["新增 issue", "578.2", "609.9", "+5.5%"],
        ["新增 PR", "1,068.6", "2,102.6", "+96.8%"],
        ["Merged PR", "720.1", "974.4", "+35.3%"],
        ["活跃 roster reviewer", "54.3", "58.0", "+6.9%"],
        ["Reviewer-days", "560.6", "697.6", "+24.4%"],
        ["Submitted review", "2,316.7", "2,727.3", "+17.7%"],
        ["Inline review comment", "1,949.3", "2,003.4", "+2.8%"],
        ["Submitted review / 新增 PR", "2.17", "1.35", "-37.7%"],
        ["Inline comment / 新增 PR", "1.83", "1.01", "-44.8%"],
    ]
    story.append(P("表 2. 需求与可见 review capacity 的月均比较", st["TableCaption"]))
    story.append(styled_table(compare, [62 * mm, 31 * mm, 39 * mm, 31 * mm], st, alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}))
    story.append(P("新增 PR 几乎翻倍，但 merge、active reviewer、review submissions 和 reviewer-days 均未同比例增长。2026 年 7 月单月新增 2,722 个 PR、merge 1,134 个，只有 52 名 roster reviewer 有 review 活动，相当于每名活跃 reviewer 对应 52.3 个新增 PR。", st["Body"]))
    story.append(P("结果不是单一月份的异常：2026 年 3 月新增 2,342 个 PR，6 月 2,545 个，7 月 2,722 个。open PR 从 2025 年末的 1,320 增至 4,194，增长 217.7%；open issue 同期从 1,791 增至 2,055，增长 14.7%。因此最突出的 queue accumulation 发生在 code integration surface，而非 issue intake。", st["Body"]))
    story.append(figure("activity_and_backlog.png", "月度 issue/PR intake、merge throughput 与月末 backlog。PR intake 在 2026 年快速上升，而 PR backlog 明显脱离 issue backlog。", 1, st, max_height=104 * mm))

    story.append(heading("4. Cutoff queue 与 responsiveness", st["H1"], 0))
    story.append(heading("4.1 维护者在 cutoff 面对的公开队列", st["H2"], 1))
    queue = [
        ["Queue signal", "Open issue", "Open PR"],
        ["总量", "2,055", "4,194"],
        ["无 any-human response", "682 (33.2%)", "-"],
        ["无 roster response", "1,570 (76.4%)", "3,139 (74.8%)"],
        ["无 submitted roster review", "-", "3,405 (81.2%)"],
        ["Outstanding review request", "-", "3,013 (71.8%)"],
        ["超过 90 天", "798 (38.8%)", "-"],
        ["Rebase/conflict signal", "-", "1,783 (42.5%)"],
        ["当前有 assignee / 仍是 draft", "165 (8.0%)", "757 (18.0%)"],
    ]
    story.append(P("表 3. 2026-07-31 的 open queue", st["TableCaption"]))
    story.append(styled_table(queue, [72 * mm, 44 * mm, 47 * mm], st, alignments={1: "RIGHT", 2: "RIGHT"}))
    story.append(P("这些数不表示每个 open item 都应被接受或合并；它们表示项目必须 triage、review、redirect、close 或 integrate 的公开 work surface。特别是 81.2% 的 open PR 没有 submitted roster review，说明 queue 不只是“等待 merge”，而是大量 work 尚未进入 formal review。", st["Body"]))
    story.append(figure("current_queues.png", "cutoff 时 open issue 与 open PR 的类型分布，以及缺少 roster response/review 的数量。", 2, st, max_height=92 * mm))

    story.append(heading("4.2 固定窗口 response", st["H2"], 1))
    response = [
        ["Artifact / responder", "Launch-2024", "2025", "2026 Jan-Jul"],
        ["Issue: any non-author human", "65.0%", "59.1%", "53.3%"],
        ["Issue: May-18 roster", "46.1%", "40.0%", "23.0%"],
        ["PR: any non-author human", "82.8%", "82.6%", "63.5%"],
        ["PR: May-18 roster", "79.9%", "80.1%", "57.6%"],
        ["PR: submitted roster review", "72.1%", "73.8%", "50.8%"],
    ]
    story.append(P("表 4. 创建后 7 天内的 response rate", st["TableCaption"]))
    story.append(styled_table(response, [69 * mm, 31 * mm, 30 * mm, 34 * mm], st, alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}))
    story.append(P("2026 cohort 的 30-day rate 为：issue any-human 61.0%、issue roster 27.5%、PR any-human 72.0%、PR roster 66.6%、submitted roster review 59.4%。延长观察窗口能恢复一部分 response，但不会消除 early-review gap。External-human PR 的 7-day any-human/roster response 为 57.8%/50.8%，而 roster-authored PR 为 81.7%/79.2%，说明混合总体会掩盖外部贡献者体验。", st["Body"]))
    story.append(figure("response_within_7_days.png", "不同创建 cohort 的 7-day response。any-human 与 roster response 均在 2026 下降。", 3, st, max_height=75 * mm))

    add_figure_page(
        story,
        "response_survival.png",
        "Issue 与 PR 的 roster response cumulative incidence。曲线保留 unresolved artifact，而非只分析最终获得 response 的样本。",
        4,
        st,
        "注：PR 从 ready-for-review 时点开始计时；May-18 roster 是 sensitivity definition，不是 event-time maintainer roster。",
        155 * mm,
    )

    story.append(PageBreak())
    story.append(heading("5. Workload composition", st["H1"], 0))
    story.append(heading("5.1 Issue intent 与 PR work type", st["H2"], 1))
    story.append(P("2026 Jan-Jul 的 4,269 个 issue 中，2,475 个（58.0%）为 bug/correctness，534 个（12.5%）为 feature/model/backend request，302 个（7.1%）为 other/tracking，291 个（6.8%）为 CI/infrastructure，282 个（6.6%）为 design/RFC。bug share 从 2025 的 55.2% 升至 58.0%。", st["Body"]))
    work = [
        ["PR work type", "数量", "2026 share", "2025 share"],
        ["Bug/correctness", "4,982", "33.8%", "22.8%"],
        ["Other/unclear", "2,194", "14.9%", "16.7%"],
        ["CI/build/release", "2,100", "14.3%", "16.6%"],
        ["Documentation/API/UX", "1,984", "13.5%", "20.1%"],
        ["Feature/capability", "1,917", "13.0%", "13.7%"],
        ["Performance/efficiency", "895", "6.1%", "5.5%"],
        ["Refactor/maintainability", "486", "3.3%", "3.6%"],
        ["Test/evaluation", "125", "0.8%", "0.5%"],
    ]
    story.append(P("表 5. 2026 Jan-Jul 的 PR work type", st["TableCaption"]))
    story.append(styled_table(work, [69 * mm, 27 * mm, 34 * mm, 34 * mm], st, alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}))
    story.append(P("最清晰的结构变化是 bug/correctness share 上升约 11 个百分点。分类中 60.1% 来自 title tag/current label，25.0% 来自 deterministic lexical heuristic，14.9% 保持 unresolved。该 taxonomy 适合 source-frame stratification，但不能替代最终 task 的人工编码。", st["Body"]))
    story.append(figure("workload_mix.png", "Issue intent 与 PR work type 在三个时期的构成。分类为 single-label；比例各自加总为 100%。", 5, st, max_height=88 * mm))

    story.append(heading("5.2 Inference engineering topics 与 hardware", st["H2"], 1))
    technical = [
        ["Topic signal", "PR", "2026 share", "2025 share"],
        ["Distributed and parallelism", "5,418", "36.8%", "29.1%"],
        ["Attention and kernels", "4,025", "27.3%", "19.8%"],
        ["V1 engine and model runner", "3,692", "25.1%", "17.3%"],
        ["Model support", "2,704", "18.4%", "17.5%"],
        ["Frontend, serving, APIs", "2,493", "16.9%", "18.9%"],
        ["KV cache/connectors/offload", "2,148", "14.6%", "8.8%"],
        ["Quantization/low precision", "2,133", "14.5%", "10.9%"],
        ["MoE/expert parallelism", "1,761", "12.0%", "9.0%"],
        ["Speculative decoding", "1,505", "10.2%", "7.8%"],
    ]
    story.append(P("表 6. 2026 Jan-Jul 的主要 inference topic signals", st["TableCaption"]))
    story.append(styled_table(technical, [74 * mm, 25 * mm, 32 * mm, 32 * mm], st, alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}))
    story.append(P("这些 multi-label signals 表明，真实 AI inference engineering 并非主要是增加模型配置，而是 distributed execution、kernel、runtime、memory、quantization、MoE、speculative decoding 与 serving integration 的组合。", st["Body"]))
    story.append(figure("engineering_topics.png", "vLLM inference engineering topic signals。一个 PR 可以属于多个 topic，因此 share 不加总为 100%。", 6, st, max_height=104 * mm))

    add_figure_page(
        story,
        "subsystems_and_hardware.png",
        "PR subsystem 与 hardware signals。2026 中 CUDA 2,505 个、ROCm 2,432 个、CPU 1,099 个、cross-backend 1,000 个、XPU 879 个。",
        7,
        st,
        "Ascend/NPU 仅 42 个、MLU 为 0；若 benchmark 纳入这些平台，应作为 maintainer-nominated heterogeneous stress track，而不是声称来自 vLLM 代表性采样。",
        153 * mm,
    )

    story.append(PageBreak())
    story.append(heading("6. 谁在实现、review 与 merge", st["H1"], 0))
    story.append(heading("6.1 Implementation intake 与 integration gatekeeping", st["H2"], 1))
    story.append(P("2026 Jan-Jul 的 14,643 个 human-authored PR 中，external humans 提交 10,993 个（75.1%），May-18 snapshot write+ actors 提交 2,896 个（19.8%），triage-only actors 提交 754 个（5.1%）。外部社区提供绝大多数 implementation intake。", st["Body"]))
    story.append(P("在具有 90-day follow-up 的 PR 中，external-human PR 的 merge rate 为 42.5%，write+ author 为 83.8%，triage-only 为 80.8%。这些是 observational associations，不能解释为 patch quality 差异；task selection、specialization、author history、reviewer familiarity 和 priority 都与 author role 混杂。", st["Body"]))
    story.append(P("在 6,445 个 merge actor 可观察的 2026 human-authored merges 中，96.1% 由 May-18 write+ actors 完成，3.8% 由 triage-only actors 完成；168 个 materialized merge 的 actor 缺失且未被插补。因此 implementation supply 广泛分布，但 final integration 仍显著 permission-gated。", st["Body"]))
    story.append(figure("engineering_and_review_ownership.png", "Write+ engineering ownership 与 roster review ownership 的 concentration。不同 work type 的 specialist dependency 不同。", 8, st, max_height=102 * mm))
    story.append(figure("pr_authorship_by_type.png", "各 author-role 的 work-type composition，以及每类 PR 由何种 author role 提交。", 9, st, max_height=98 * mm))

    story.append(heading("6.2 Review capacity 与 concentration", st["H2"], 1))
    story.append(P("2026 Jan-Jul 有 77 名 roster member 提交 19,091 个 non-author submitted review。Top five 完成 35.0%，10 人完成一半，23 人完成 80%，Gini 为 0.664。相较早期 May 分析，active reviewer 从 75 增至 77，top-five share 从 39.7% 降至 35.0%。因此数据不支持“review 越来越集中”；更准确的结论是：参与稍有扩展，但 capacity 增长远慢于 demand。", st["Body"]))
    story.append(P("Merged PR 并不吸收全部 review。2026 creation cohort 中，merged PR 占 roster submitted review 的 81.4% 和 inline comment 的 74.1%；closed-unmerged PR 占 8.9%/11.5%；open PR 已占 9.7%/14.3%。只分析 merged PR 会漏掉 18.6% 的 submitted review 和 25.9% 的 inline review workload。", st["Body"]))
    story.append(figure("review_capacity.png", "Review demand、活跃 reviewer、每 reviewer PR intake，以及 review load 相对 PR volume 的分布。", 10, st, max_height=104 * mm))

    add_figure_page(
        story,
        "maintainer_workload.png",
        "不同公开维护 action 的月均事件量，以及各 PR 类型占 PR volume 与 submitted review 的份额。事件数是可观察 activity，不是 effort hours。",
        11,
        st,
        "Issue conversation comments 在 2026 的下降可能同时受 automation、templates、triage practice 与 unresolved demand 影响，不应单独解释为维护投入下降。",
        160 * mm,
    )

    story.append(PageBreak())
    story.append(heading("7. Contributor lifecycle 与 ownership breadth", st["H1"], 0))
    story.append(P("2026 Jan-Jul 有 3,401 名 external authors 提交 10,993 个 PR，其中 2,807 人是数据中首次观察到的 PR author。1,896 名作者只提交 1 个 PR，占 external PR 的 17.2%；984 名作者提交 2-4 个，占 23.2%；521 名作者提交 5 个以上，贡献 59.6% 的 external PR。广泛 onboarding 与高产 repeat production 同时存在。", st["Body"]))
    story.append(P("Roster review coverage 随观察到的 contributor experience 上升：first PR 为 34.0%，第 2-5 个为 43.0%，第 6 个及以后为 56.9%；90-day merge 分别为 27.9%、38.8%、53.8%。具有完整 90-day follow-up 的 2026 first-time external authors 中，42.0% 在 90 天内再次提交 PR。该指标仅衡量 public PR return，不等同于 retention、employment 或 expertise。", st["Body"]))
    story.append(figure("external_contributor_lifecycle.png", "External contributor frequency、experience-outcome association 与 first-author return。", 12, st, max_height=90 * mm))
    story.append(figure("contributor_pressure.png", "External contribution intake 与 reviewer capacity 随时间的相对演化。", 13, st, max_height=86 * mm))

    add_figure_page(
        story,
        "collaborator_portfolios.png",
        "May-18 snapshot roster 的公开 portfolio overlap 与 reviewer work-type specialization。No observed public action 不表示没有 private、security、release 或跨仓库工作。",
        14,
        st,
        None,
        153 * mm,
    )

    story.append(PageBreak())
    story.append(heading("8. 对 AI inference benchmark 的直接含义", st["H1"], 0))
    story.append(heading("8.1 Representative source frame", st["H2"], 1))
    story.append(P("截至 cutoff，2026 source frame 包含 6,613 个 merged、human-authored、具有 commit data 的 PR。其中 45.0% touch tests，39.0% 有 hardware signal，5.9% 为 performance intent，9.6% 为 large change，14.5% 为 review-intensive，2.9% 为 docs-only。", st["Body"]))
    story.append(P("Test-file presence 只是 visible verifier signal，不表示现有测试能够完整判定 benchmark query。Performance PR 中 37.4% touch tests，bug 40.2%，feature 44.9%，CI/build 52.4%。恰恰是 benchmark 最重要的 performance 和 bug work，现有 verifier coverage 较弱。", st["Body"]))
    story.append(figure("benchmark_task_signals.png", "2026 merged human code PR 的 test-file 与 review-intensive signals，按 work type 分解。", 15, st, max_height=82 * mm))
    story.append(figure("verifier_signals.png", "Eligible source frame 的 test-file signal，按 work type 与 hardware 分解。", 16, st, max_height=82 * mm))

    story.append(heading("8.2 Benchmark sampling 与 task contracts", st["H2"], 1))
    story.append(P("76 个 representative tasks 应由人工确认后的 eligible frame 分层抽样，而不能直接把 deterministic taxonomy share 当作最终配额。建议至少覆盖：", st["Body"]))
    for text in [
        "Bug/correctness：runtime、kernel、distributed、API 与 model support；",
        "Feature/capability 与 model/backend integration；",
        "Performance/efficiency，并保留 latency、throughput、memory 等连续 reward；",
        "CI/build/release 与跨平台 breakage；",
        "Refactor/architecture migration 与 test/evaluation；",
        "CUDA、ROCm、XPU、CPU 与 cross-backend；",
        "Distributed、attention/kernel、V1、KV cache、quantization、MoE 与 speculative decoding；",
        "Open/closed-unmerged、review-intensive 和 maintainer-nominated tasks，用于补足 merged-only frame 的盲点。",
    ]:
        story.append(bullet(text, st))
    story.append(P("24 个 memorable tasks 应作为独立的 expert-nominated stress track 报告，不与 probability-sampled representative track 合并成一个总体 pass rate。任务合同也应区分 implementation、diagnosis/reproduction 和 review；三者具有不同的 source population、environment 与 verifier。", st["Body"]))
    story.append(callout("Benchmark 的核心不是“agent 能不能改代码”，而是它能否在真实 inference 系统约束下完成 diagnosis、implementation、test construction、performance validation、hardware adaptation 与 review iteration。", st))

    story.append(heading("8.3 Environment 与 evaluation implications", st["H2"], 1))
    story.append(P("Hardware strata 意味着统一 CPU container 无法代表全部 workload。CUDA/ROCm/XPU/CPU/cross-backend tasks 需要固定 driver、runtime、model checkpoint 和 device topology；specialist tasks 还需要真实 accelerator。环境必须在断网条件下 ready，包括依赖、模型、测试数据与 compiler cache。", st["Body"]))
    story.append(P("Verifier 采用分层设计：优先使用 reference PR 的 unit/e2e tests；若 query 对应的原测试不足，则合成 query-specific tests，并经过 agent review 与 domain expert review；performance tasks 使用容差、重复运行和 continuous reward。每个 release 冻结 exact model ID、harness version、budget、sampling setting、environment digest 与 verifier commit。", st["Body"]))

    add_figure_page(
        story,
        "pr_complexity.png",
        "Patch-size strata 与 review-intensive/test signals。Large patch 更常 touch tests，也更可能进入 review-intensive stratum。",
        17,
        st,
        "Cumulative churn 是 commit-file changes 的累计值，可高于 final diff；由于 commit coverage 与 PR outcome 相关，不用它估计 merge probability。",
        150 * mm,
    )

    add_figure_page(
        story,
        "pr_competing_outcomes.png",
        "PR merge 与 closed-unmerged 的 competing cumulative incidence。2026 cohort 的 integration outcome 明显慢于早期 cohort。",
        18,
        st,
        "30-day incidence：merge 48.3%，closed-unmerged 18.1%；90-day incidence：merge 51.8%，closed-unmerged 23.8%。",
        143 * mm,
    )

    story.append(PageBreak())
    story.append(heading("9. 结论更新、稳健性与限制", st["H1"], 0))
    story.append(heading("9.1 从 May 分析到 July 完整月窗口", st["H2"], 1))
    audit = [
        ["Indicator", "Earlier May analysis", "July 31 analysis", "结论"],
        ["2026 月均新增 PR", "1,836.3", "2,102.6", "需求增长更强"],
        ["2026 月均 merged PR", "909.0", "974.4", "仍远慢于 intake"],
        ["月均 active reviewer", "60.3", "58.0", "无 capacity 扩张"],
        ["Submitted review / PR", "1.55", "1.35", "review density 更低"],
        ["Cutoff open PR", "3,037", "4,194", "integration queue 扩大"],
        ["Issue roster response 7d", "27.2%", "23.0%", "responsiveness 更低"],
        ["PR roster response 7d", "62.2%", "57.6%", "responsiveness 更低"],
        ["Active 2026 reviewers", "75", "77", "参与略扩展"],
        ["Top-five review share", "39.7%", "35.0%", "集中度下降"],
    ]
    story.append(P("表 7. 旧结论在完整 July cutoff 下的更新", st["TableCaption"]))
    story.append(styled_table(audit, [52 * mm, 36 * mm, 36 * mm, 40 * mm], st, font_size=7.1, alignments={1: "RIGHT", 2: "RIGHT"}))
    story.append(P("中心结论得到加强：demand-capacity gap、response slowdown、external implementation supply、permissioned integration 和 specialist workload 均保持稳健。需要修正的是 review concentration：参与者数量略增，top-five share 下降，因此不能声称 gatekeeping 变得更集中；可辩护的表述是 capacity 增长显著慢于 demand，导致每个 PR 的 visible review density 下降。", st["Body"]))

    story.append(heading("9.2 Validity boundaries", st["H2"], 1))
    limits = [
        ["Validity", "边界与报告约束"],
        ["Construct", "GitHub event 是公开 activity proxy，不是总劳动或工时；不包含 Slack、private security、vendor coordination、本地 debugging。"],
        ["Role", "May-18 roster 不是 July roster 或历史 membership table；any-human response 是 primary estimand。"],
        ["Classification", "Title/label/path taxonomy 是 exploratory deterministic classifier；paper release 仍需 time-stratified human gold sample。"],
        ["Temporal", "1,799 个 artifact 的 text representation 可能在 cutoff 后更新，1,227 个 open PR file list 的稳定性不能证明；stable-only sensitivity 最大 1.00 pp。"],
        ["Selection", "Merged implementation frame 排除 unanswered issues、abandoned PR 与无 stable solution 的工作；agent coverage 不可外推为全部维护劳动。"],
        ["Causal", "Author-role、hardware、work type 与 outcomes 的差异是 descriptive associations，不是质量、偏见、burnout 或 staffing effect。"],
    ]
    story.append(P("表 8. 主要 validity threats 与解释边界", st["TableCaption"]))
    story.append(styled_table(limits, [32 * mm, 132 * mm], st, font_size=7.4))

    story.append(heading("10. RQ1 最终回答", st["H1"], 0))
    story.append(callout("真实 vLLM 维护是一个 high-growth、community-driven、specialist-integration-constrained 的系统。2026 的 PR demand 接近翻倍，外部作者提供约四分之三的 human PR intake，但 review、merge 和 specialist gatekeeping 的可见 capacity 增长明显更慢。", st))
    story.append(Spacer(1, 4 * mm))
    story.append(P("因此，一个声称衡量 LLM agents 能解决多少真实 AI inference workload 的 benchmark，必须超越普通 repository editing。它应测试 agent 是否能够：", st["Body"]))
    for text in [
        "从不完整 issue 和 failing behavior 中诊断 root cause；",
        "在 distributed runtime、kernel、memory 与 serving architecture 中实现正确修改；",
        "构建足以验证 query 的测试，而非只让已有测试通过；",
        "在真实 accelerator 和固定环境中验证 performance 与 heterogeneous compatibility；",
        "理解 review feedback、修改方案并完成多轮 integration；",
        "识别何时任务需要 specialist judgement，避免以 superficially passing patch 替代正确解决方案。",
    ]:
        story.append(bullet(text, st))
    story.append(P("RQ1 提供的不是 benchmark pass rate，而是 pass rate 应针对什么 population 才有意义。Representative sampling、memorable stress tasks、hardware tracks 与 verifier quality 必须分别报告；否则一个看似精确的总体分数无法回答最初的研究问题。", st["Body"]))

    add_figure_page(
        story,
        "path_area_ownership.png",
        "附录图：2026 各 changed-path area 的 PR volume、author-role composition 与 ownership concentration。",
        19,
        st,
        "Path area 是 changed-file-level deterministic category，用于 environment、reviewer 和 task dependency planning，不用于评价个人。",
        165 * mm,
    )

    story.append(PageBreak())
    story.append(heading("参考文献与数据可用性", st["H1"], 0))
    refs = [
        "[1] Simon Mo. <i>vLLM GitHub Gym: vLLM GitHub Snapshot (Fivetran)</i>. GitHub Gist, 2026-05-18. https://gist.github.com/simon-mo/2b0f4e9f872d479a08ae53edac51ecb1",
        "[2] GitHub. <i>REST API endpoints for timeline events</i>. https://docs.github.com/en/rest/issues/timeline",
        "[3] GitHub. <i>REST API endpoints for pull request reviews</i>. https://docs.github.com/en/rest/pulls/reviews",
        "[4] Wessel et al. <i>Understanding the Time to First Response in GitHub Pull Requests</i>. 2023. https://arxiv.org/abs/2304.08426",
        "[5] Kalliamvakou et al. <i>The Promises and Perils of Mining GitHub</i>. MSR 2014.",
        "[6] Chatterjee, Sharma, and Ralph. <i>Empirical Standards for Repository Mining</i>. MSR 2022. https://arxiv.org/abs/2203.15950",
        "[7] Huang et al. <i>What Do Users Ask in Open-Source AI Repositories?</i>. 2023. https://arxiv.org/abs/2303.09795",
        "[8] Khatoonabadi et al. <i>On Wasted Contributions: Understanding the Dynamics of Contributor-Abandoned Pull Requests</i>. TOSEM 2023.",
    ]
    for ref in refs:
        story.append(P(ref, st["Reference"]))
    story.append(Spacer(1, 4 * mm))
    story.append(heading("数据与复现", st["H2"], 1))
    story.append(P("合并数据库发布于：<link href='https://github.com/ai-infra-bench/ai-infra-bench/releases/tag/vllm-github-data-2026-07-31' color='#2E86AB'>github.com/ai-infra-bench/ai-infra-bench/releases/tag/vllm-github-data-2026-07-31</link>。分析代码、operational codebook、中英文 findings、aggregate summary 与 figure assets 位于 AI Infra Bench repository 的 <font name='RQSans'>analysis/rq1/</font> 与 <font name='RQSans'>docs/</font>。", st["BodyNoIndent"]))
    story.append(P("数据库包含 public GitHub text、usernames、actor identifiers 与 commit metadata；不是 de-identified survey dataset。项目不对第三方 GitHub content 主张新的许可，使用者应遵守 source 与 GitHub terms。公开报告不发布 actor-level rankings、names、email 或 comment text。", st["BodyNoIndent"]))
    story.append(Spacer(1, 9 * mm))
    story.append(P("AI Infra Bench / RQ1 Empirical Study", ParagraphStyle("End", parent=st["Small"], alignment=TA_CENTER, textColor=NAVY)))
    return story


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    opt = parser.parse_args()
    register_fonts()
    with SUMMARY.open() as stream:
        summary = json.load(stream)
    if summary["snapshot"]["cutoff"] != "2026-07-31 23:59:59":
        raise ValueError("The report requires the July 31 RQ1 summary")
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    doc = AcademicDocTemplate(
        str(opt.output),
        pagesize=A4,
        leftMargin=23 * mm,
        rightMargin=23 * mm,
        topMargin=21 * mm,
        bottomMargin=20 * mm,
        title="vLLM 的真实维护 workload：需求增长、维护 capacity 与 AI inference benchmark 的实证基础",
        author="AI Infra Bench",
        subject="RQ1 empirical study through 2026-07-31",
    )
    doc.multiBuild(build_story(styles(), summary))
    print(opt.output)


if __name__ == "__main__":
    main()
