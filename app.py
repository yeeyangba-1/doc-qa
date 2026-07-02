"""
AI 学习与考试助手
"""
import streamlit as st
import time, re, threading
from rag import build_kb, KnowledgeBase
from features import smart_qa, smart_qa_with_trace, generate_outline, generate_exam, analyze_wrong_answer
from rag_logger import append_qa_log, get_log_path

st.set_page_config(page_title="StudyRAG", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

# ============================================
# CSS
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
:root {
    --blue: #4F46E5; --blue-light: #EEF2FF; --green: #059669; --green-light: #ECFDF5;
    --purple: #7C3AED; --purple-light: #F5F3FF; --amber: #D97706; --amber-light: #FFFBEB;
    --g50: #F9FAFB; --g100: #F3F4F6; --g200: #E5E7EB; --g300: #D1D5DB;
    --g400: #9CA3AF; --g500: #6B7280; --g600: #4B5563; --g700: #374151; --g800: #1F2937; --g900: #111827;
}
.stApp { background: #FAFBFC; }

@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
@keyframes slideIn { from { opacity: 0; transform: translateX(-16px); } to { opacity: 1; transform: translateX(0); } }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }

/* ===== 欢迎页 ===== */
.hero-wrap {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-height: 80vh; padding: 40px 20px; text-align: center;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--blue-light); color: var(--blue);
    font-size: 13px; font-weight: 600; padding: 6px 16px;
    border-radius: 100px; margin-bottom: 20px;
    animation: fadeInUp 0.5s ease-out;
}
.hero-title {
    font-size: 44px; font-weight: 800; color: var(--g900);
    letter-spacing: -2px; margin-bottom: 6px;
    animation: fadeInUp 0.5s ease-out 0.1s both;
}
.hero-desc {
    font-size: 16px; color: var(--g500); margin-bottom: 36px;
    animation: fadeInUp 0.5s ease-out 0.2s both;
}

/* ===== 上传区 ===== */
.upload-box {
    max-width: 500px; width: 100%; margin: 0 auto;
    animation: fadeInUp 0.5s ease-out 0.3s both;
}
.upload-box .stFileUploader > section {
    border: 2px dashed var(--g300) !important; border-radius: 16px !important;
    padding: 36px 28px !important; text-align: center !important;
    background: white !important; transition: all 0.3s ease !important;
}
.upload-box .stFileUploader > section:hover {
    border-color: var(--blue) !important; background: var(--blue-light) !important;
    box-shadow: 0 8px 30px rgba(79,70,229,0.1) !important;
}
.upload-box .stFileUploader button {
    background: var(--blue-light) !important; color: var(--blue) !important;
    border: none !important; border-radius: 50% !important;
    width: 48px !important; height: 48px !important; padding: 0 !important; font-size: 0 !important;
    transition: all 0.3s ease !important;
}
.upload-box .stFileUploader button::before {
    content: '+'; font-size: 24px; font-weight: 300;
    display: flex; align-items: center; justify-content: center; height: 100%;
}
.upload-box .stFileUploader section:hover button {
    background: var(--blue) !important; color: white !important; transform: scale(1.08);
}
.upload-box .stFileUploader small { font-size: 12px !important; color: var(--g400) !important; }

/* ===== 功能卡片 ===== */
.fgrid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px;
    max-width: 960px; margin: 40px auto 0;
    animation: fadeInUp 0.5s ease-out 0.4s both;
}
.fcard {
    background: white; border-radius: 16px; padding: 28px 22px 22px;
    border: 1px solid var(--g100); box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    transition: all 0.3s ease; position: relative; overflow: hidden;
}
.fcard::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 16px 16px 0 0; transition: height 0.3s ease;
}
.fcard:nth-child(1)::before { background: var(--blue); }
.fcard:nth-child(2)::before { background: var(--green); }
.fcard:nth-child(3)::before { background: var(--purple); }
.fcard:nth-child(4)::before { background: var(--amber); }
.fcard:hover { transform: translateY(-4px); box-shadow: 0 12px 36px rgba(0,0,0,0.07); }
.fcard:hover::before { height: 5px; }
.fc-icon {
    width: 40px; height: 40px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center; margin-bottom: 12px;
}
.fcard:nth-child(1) .fc-icon { background: var(--blue-light); }
.fcard:nth-child(2) .fc-icon { background: var(--green-light); }
.fcard:nth-child(3) .fc-icon { background: var(--purple-light); }
.fcard:nth-child(4) .fc-icon { background: var(--amber-light); }
.fc-title { font-size: 14px; font-weight: 700; color: var(--g800); margin-bottom: 4px; }
.fc-desc { font-size: 12px; color: var(--g400); line-height: 1.5; }

/* ===== 侧边栏 ===== */
.sb-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 18px 14px 14px; border-bottom: 1px solid var(--g100);
}
.sb-logo {
    width: 32px; height: 32px; border-radius: 8px;
    background: var(--blue); color: white;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 13px;
}
.sb-name { font-weight: 700; font-size: 14px; color: var(--g900); }
.nav-label {
    font-size: 10px; font-weight: 600; color: var(--g400);
    text-transform: uppercase; letter-spacing: 1.2px;
    padding: 14px 10px 2px;
}

/* ===== 结构树 ===== */
.tree-chapter {
    padding: 8px 10px; margin: 2px 4px; border-radius: 8px;
    cursor: pointer; transition: all 0.15s ease; font-size: 13px;
    color: var(--g700); font-weight: 500;
    display: flex; align-items: center; gap: 6px;
    border: 1px solid transparent;
}
.tree-chapter:hover { background: var(--g50); }
.tree-chapter.active { background: var(--blue-light); color: var(--blue); border-color: #C7D2FE; }
.tree-chapter .ch-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--g300); flex-shrink: 0; }
.tree-chapter.active .ch-dot { background: var(--blue); }
.tree-chapter .ch-arrow { font-size: 10px; color: var(--g400); margin-left: auto; transition: transform 0.2s ease; }
.tree-chapter.active .ch-arrow { transform: rotate(90deg); }

.tree-kp {
    padding: 5px 10px 5px 28px; margin: 1px 4px; border-radius: 6px;
    cursor: pointer; transition: all 0.15s ease; font-size: 12px;
    color: var(--g500); border: 1px solid transparent;
}
.tree-kp:hover { background: var(--g50); color: var(--g700); }
.tree-kp.active { background: var(--blue-light); color: var(--blue); font-weight: 500; border-color: #E0E7FF; }

/* ===== 主区域 ===== */
.main-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 14px; border-bottom: 1px solid var(--g100);
    margin-bottom: 16px; animation: fadeIn 0.4s ease;
}
.doc-title { font-size: 17px; font-weight: 700; color: var(--g900); display: flex; align-items: center; gap: 8px; }
.doc-meta { font-size: 11px; color: var(--g400); background: var(--g50); padding: 3px 10px; border-radius: 100px; }
.doc-status {
    font-size: 11px; padding: 3px 10px; border-radius: 100px; margin-left: 8px;
}

/* ===== 原文面板 ===== */
.text-panel-label {
    font-size: 13px; font-weight: 600; color: var(--g600); margin-bottom: 6px;
    display: flex; align-items: center; gap: 6px;
}
.stTextArea textarea {
    border: 1px solid var(--g200) !important; border-radius: 10px !important;
    padding: 14px 18px !important; font-size: 14px !important;
    box-shadow: none !important; background: white !important; line-height: 1.8 !important;
}
.stTextArea textarea:focus {
    border-color: var(--blue) !important; box-shadow: 0 0 0 3px rgba(79,70,229,0.08) !important;
}

/* ===== 聊天 ===== */
.chat-label {
    font-size: 13px; font-weight: 600; color: var(--g600); margin-bottom: 6px;
}
.stButton > button {
    background: var(--g900) !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    padding: 10px 24px !important; font-weight: 600 !important;
    font-size: 14px !important; transition: all 0.2s ease !important;
}
.stButton > button:hover { background: #1f2937 !important; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
.stButton > button:active { transform: translateY(0); }

/* ===== Radio 切换 ===== */
.stRadio [role="radiogroup"] { display: flex; gap: 4px; background: var(--g100); border-radius: 12px; padding: 4px; }
.stRadio label {
    background: transparent; border: none; border-radius: 9px;
    padding: 8px 16px !important; font-size: 13px; font-weight: 500; color: var(--g600);
    transition: all 0.2s ease; cursor: pointer;
}
.stRadio label:hover { color: var(--g900); }
.stRadio label:has(input:checked) { background: white; color: var(--g900); font-weight: 600; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }

@media (max-width: 900px) { .fgrid { grid-template-columns: repeat(2, 1fr); } .hero-title { font-size: 30px; } }
</style>
""", unsafe_allow_html=True)

# SVG
SVG = {
    "book": '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="1.5"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
    "sparkle": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="1.5"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>',
    "qa": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "outline": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    "exam": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>',
    "analysis": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
    "doc": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>',
}

# ============================================
# Session
# ============================================
if "docs" not in st.session_state: st.session_state.docs = {}
if "active_doc" not in st.session_state: st.session_state.active_doc = None
if "doc_list" not in st.session_state: st.session_state.doc_list = []
if "active_ch" not in st.session_state: st.session_state.active_ch = None   # 选中的章节索引
if "active_kp" not in st.session_state: st.session_state.active_kp = None
if "sel_text" not in st.session_state: st.session_state.sel_text = ""
if "answer" not in st.session_state: st.session_state.answer = ""
if "retrieved_docs" not in st.session_state: st.session_state.retrieved_docs = []
if "scores" not in st.session_state: st.session_state.scores = []
if "is_low_confidence" not in st.session_state: st.session_state.is_low_confidence = False
if "show_hero" not in st.session_state: st.session_state.show_hero = True

has_docs = len(st.session_state.doc_list) > 0

# ============================================
# 欢迎页
# ============================================
if not has_docs or st.session_state.show_hero:
    col_l, col_c, col_r = st.columns([1, 2.2, 1])
    with col_c:
        st.markdown(f"""
        <div class="hero-wrap">
            <div class="hero-badge">{SVG['sparkle']} 基于 RAG 检索增强生成</div>
            {SVG['book']}
            <div class="hero-title">StudyRAG</div>
            <p class="hero-desc">上传学习资料，AI 帮你智能复习、自动出题、攻克难点</p>
            <div class="upload-box">
                <p style="text-align:center;font-size:12px;color:#9ca3af;margin-bottom:2px;">
                    支持 PDF · TXT · 教师讲义 · 实验指导书
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("选择文件", type=["pdf","txt"], label_visibility="collapsed", key="hero_upload")

        if uploaded_file:
            # 阶段1：快速读原文
            status = st.status("正在处理文件...", expanded=True)
            with status:
                st.write("读取文件中...")
                file_bytes = uploaded_file.read()
                from rag import build_kb
                kb = build_kb(file_bytes, uploaded_file.name)
                st.write(f"已提取 {len(kb.raw_text)} 字")

                # 阶段2：构建搜索索引
                st.write("构建搜索索引...")
                kb.build_index()
                st.write(f"索引完成：{kb.n_chunks} 个文本块")
                status.update(label="处理完成！", state="complete")

            st.session_state.docs[uploaded_file.name] = kb
            st.session_state.doc_list.append(uploaded_file.name)
            st.session_state.active_doc = uploaded_file.name
            st.session_state.show_hero = False
            st.session_state.active_ch = None
            st.session_state.active_kp = None
            st.session_state.answer = ""
            st.session_state.retrieved_docs = []
            st.session_state.scores = []
            st.session_state.is_low_confidence = False
            time.sleep(0.3)
            st.rerun()

        # 功能卡片
        st.markdown(f"""
        <div class="fgrid">
            <div class="fcard">
                <div class="fc-icon">{SVG['qa']}</div>
                <div class="fc-title">智能问答</div>
                <div class="fc-desc">针对资料深入提问<br/>AI 基于原文精准回答</div>
            </div>
            <div class="fcard">
                <div class="fc-icon">{SVG['outline']}</div>
                <div class="fc-title">复习提纲</div>
                <div class="fc-desc">自动提取核心知识<br/>分章节整理重点难点</div>
            </div>
            <div class="fcard">
                <div class="fc-icon">{SVG['exam']}</div>
                <div class="fc-title">自动出题</div>
                <div class="fc-desc">根据资料生成试题<br/>附带答案与详细解析</div>
            </div>
            <div class="fcard">
                <div class="fc-icon">{SVG['analysis']}</div>
                <div class="fc-title">错题解析</div>
                <div class="fc-desc">分析错误深层原因<br/>关联对应知识点</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ============================================
# 有资料：侧边栏 + 主界面
# ============================================
kb = st.session_state.docs[st.session_state.active_doc]

# =====================
# 左侧边栏
# =====================
with st.sidebar:
    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-logo">AI</div>
        <span class="sb-name">StudyRAG</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-label">检索参数</div>', unsafe_allow_html=True)
    top_k = st.slider("top_k", min_value=1, max_value=10, value=5, step=1)
    score_threshold = st.slider("score_threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

    log_path = get_log_path()
    if log_path.exists():
        st.download_button(
            "下载 RAG 问答日志 CSV",
            data=log_path.read_bytes(),
            file_name="rag_qa_logs.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("暂无问答日志")

    # 添加资料
    new_file = st.file_uploader("添加资料", type=["pdf","txt"], label_visibility="collapsed", key="sidebar_upload")
    if new_file and new_file.name not in st.session_state.docs:
        with st.spinner("读取中..."):
            from rag import build_kb
            file_bytes = new_file.read()
            kb_new = build_kb(file_bytes, new_file.name)
            # 后台构建索引
            t = threading.Thread(target=kb_new.build_index)
            t.start()
            st.session_state.docs[new_file.name] = kb_new
            st.session_state.doc_list.append(new_file.name)
            st.session_state.active_doc = new_file.name
            st.session_state.active_ch = None
            st.session_state.active_kp = None
            st.session_state.answer = ""
            st.session_state.retrieved_docs = []
            st.session_state.scores = []
            st.session_state.is_low_confidence = False
        st.rerun()

    # 资料列表
    st.markdown('<div class="nav-label">历史资料</div>', unsafe_allow_html=True)
    for fname in st.session_state.doc_list:
        active = st.session_state.active_doc == fname
        icon = "📄" if fname.endswith(".pdf") else "📃"
        if st.button(
            f"{icon}  {fname[:22]}{'...' if len(fname)>22 else ''}",
            key=f"doc_{fname}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state.active_doc = fname
            st.session_state.active_ch = None
            st.session_state.active_kp = None
            st.session_state.answer = ""
            st.session_state.retrieved_docs = []
            st.session_state.scores = []
            st.session_state.is_low_confidence = False
            st.rerun()

    st.markdown('<div class="nav-label">文档结构</div>', unsafe_allow_html=True)

    # 索引状态
    if not kb.ready:
        st.caption("⏳ 正在构建搜索索引...")
        if st.button("等待构建完成"):
            kb.build_index()
            st.rerun()

    # 结构分析按钮
    if not kb.structure_ready:
        if kb.ready:
            if st.button("智能分析文档结构", use_container_width=True):
                with st.spinner("AI 正在分析文档结构..."):
                    kb.build_structure()
                st.rerun()
        else:
            st.caption("索引构建完成后可分析结构")

    # 章节树
    if kb.structure_ready and kb.structure:
        structure = kb.structure
        for i, sec in enumerate(structure):
            ch_active = st.session_state.active_ch == i
            arrow = "▾ " if ch_active else "▸ "
            label = f"{arrow}{sec['title'][:26]}"
            if st.button(label, key=f"ch_{i}", use_container_width=True,
                         type="primary" if ch_active else "secondary"):
                st.session_state.active_ch = i
                st.session_state.active_kp = None
                st.rerun()

            # 知识点（激活章节下展开）
            if ch_active and sec.get("knowledge_points"):
                for kp in sec["knowledge_points"]:
                    kp_active = st.session_state.active_kp == kp["id"]
                    kp_label = f"   · {kp['title'][:28]}" if not kp_active else f"   ● {kp['title'][:28]}"
                    if st.button(kp_label, key=f"kp_{kp['id']}", use_container_width=True,
                                 type="primary" if kp_active else "secondary"):
                        st.session_state.active_kp = kp["id"]
                        st.session_state.active_ch = i
                        st.session_state.sel_text = kp.get("title", "")
                        st.rerun()
    elif not kb.ready:
        pass  # 等索引
    else:
        st.caption("点击「智能分析文档结构」开始")

# =====================
# 主界面
# =====================

# 状态标签
status_html = ""
if kb.ready and kb.structure_ready:
    status_html = f'<span class="doc-status" style="background:#ECFDF5;color:#059669;">已分析</span>'
elif kb.ready:
    status_html = f'<span class="doc-status" style="background:#FFFBEB;color:#D97706;">可搜索</span>'
else:
    status_html = f'<span class="doc-status" style="background:#FEF2F2;color:#DC2626;animation:pulse 2s infinite;">构建中</span>'

st.markdown(f"""
<div class="main-topbar">
    <div class="doc-title">
        {SVG['doc']} {st.session_state.active_doc[:45]}
        {status_html}
    </div>
    <span class="doc-meta">{kb.n_chunks if kb.ready else '...'} 个文本块</span>
</div>
""", unsafe_allow_html=True)

# 确保索引就绪
if not kb.ready:
    st.info("正在后台构建搜索索引，请稍候...")
    if st.button("点击构建索引"):
        kb.build_index()
        st.rerun()
    st.stop()

# 功能切换
feature = st.radio(
    "功能", ["💬 智能问答", "📋 复习提纲", "📝 自动出题", "🔍 错题解析"],
    horizontal=True, label_visibility="collapsed",
)

# ===== 确定显示的原文 =====
if st.session_state.active_ch is not None and kb.structure_ready and kb.structure:
    if st.session_state.active_ch < len(kb.structure):
        display_text = kb.get_section_text(st.session_state.active_ch)
    else:
        display_text = kb.raw_text
else:
    display_text = kb.raw_text

# ===== 原文面板 =====
st.markdown('<p class="text-panel-label">原文</p>', unsafe_allow_html=True)
st.text_area("原文", value=display_text[:10000], height=320, label_visibility="collapsed", key="text_viewer")
st.caption("从原文中选中文字，Ctrl+C 复制，粘贴到下方引用框，再输入问题")

# ===== 分隔 =====
st.markdown("---")

# ===== 对话区域 =====
st.markdown('<p class="chat-label">对话</p>', unsafe_allow_html=True)

c1, c2 = st.columns([1, 1])
with c1:
    quoted = st.text_area("引用文本", value=st.session_state.sel_text,
                          placeholder="从原文粘贴你想问的段落", height=80, label_visibility="collapsed")
with c2:
    question = st.text_area("你的问题", placeholder="例如：这段话的重点是什么？",
                            height=80, label_visibility="collapsed")

btn_col1, btn_col2 = st.columns([1, 5])
with btn_col1:
    ask_btn = st.button("发送", type="primary", use_container_width=True)

if ask_btn and question.strip():
    st.session_state.sel_text = quoted
    with st.spinner("检索中..."):
        full_q = f"用户引用了以下文本：\n\n{quoted}\n\n用户的问题：{question}" if quoted.strip() else question
        if feature == "💬 智能问答":
            trace = smart_qa_with_trace(kb, full_q, top_k=top_k, score_threshold=score_threshold)
            answer = trace["answer"]
            st.session_state.retrieved_docs = trace["retrieved_docs"]
            st.session_state.scores = trace["scores"]
            st.session_state.is_low_confidence = trace["is_low_confidence"]
            append_qa_log(
                query=full_q,
                answer=answer,
                retrieved_docs=trace["retrieved_docs"],
                scores=trace["scores"],
            )
        elif feature == "📋 复习提纲":
            answer = generate_outline(kb, question.strip() or "全部内容")
            st.session_state.retrieved_docs = []
            st.session_state.scores = []
            st.session_state.is_low_confidence = False
        elif feature == "📝 自动出题":
            answer = generate_exam(kb, 10)
            st.session_state.retrieved_docs = []
            st.session_state.scores = []
            st.session_state.is_low_confidence = False
        elif feature == "🔍 错题解析":
            answer = analyze_wrong_answer(kb, question, "", "")
            st.session_state.retrieved_docs = []
            st.session_state.scores = []
            st.session_state.is_low_confidence = False
        st.session_state.answer = answer
    st.rerun()

# ===== 回答展示 =====
if st.session_state.answer:
    with st.container():
        st.markdown(st.session_state.answer)
        if st.session_state.retrieved_docs:
            st.markdown("### 参考片段 / 检索证据")
            for doc in st.session_state.retrieved_docs:
                preview = doc["content"][:300].replace("\n", " ")
                with st.expander(f"chunk_id={doc['chunk_id']} · score={doc['score']:.3f}"):
                    st.write(preview)
        elif st.session_state.is_low_confidence:
            st.markdown("### 参考片段 / 检索证据")
            st.caption("无")

st.markdown("""
<div style="text-align:center; padding:36px 0 12px; color:#d1d5db; font-size:11px;">
    StudyRAG · RAG 检索溯源与质量日志
</div>
""", unsafe_allow_html=True)
