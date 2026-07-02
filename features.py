"""
四大功能：智能问答 / 复习提纲 / 自动出题 / 错题解析
"""
from rag import KnowledgeBase, ask_llm, chat_with_trace

# ============================================
# 1. 智能问答
# ============================================

QA_PROMPT = """你是一位大学课程助教，专门帮助学生理解和复习课程内容。

你的回答规则：
1. 严格根据提供的资料回答，不要编造资料里没有的内容
2. 如果资料不足以回答问题，直接说"资料中没有涉及这部分内容"
3. 回答要简洁清晰，用分点列举的形式
4. 如果学生问"重点是什么"，请结合资料中的要求、考试占比、老师强调的内容来回答
5. 回答末尾标注参考了哪些内容"""


def smart_qa(kb: KnowledgeBase, question: str) -> str:
    context = kb.search(question, top_k=5)
    return ask_llm(QA_PROMPT, question, context)


def smart_qa_with_trace(
    kb: KnowledgeBase,
    question: str,
    top_k: int = 5,
    score_threshold: float = 0.25,
) -> dict:
    return chat_with_trace(kb, QA_PROMPT, question, top_k, score_threshold)


# ============================================
# 2. 复习提纲 / 重点总结
# ============================================

OUTLINE_PROMPT = """你是一位大学课程助教，学生要求你根据学习资料整理复习提纲。

你的任务：
1. 按章节或主题组织内容
2. 每个章节列出 3~5 个核心知识点
3. 标注"重点掌握"和"了解即可"两个层次
4. 如果有公式、定义、关键概念，要单独列出
5. 格式规范、层次分明，方便学生打印复习

输出格式示例：
## 第一章 xxx
### 重点掌握
- 知识点1：xxx
- 知识点2：xxx
### 了解即可
- 知识点3：xxx
### 关键概念
- xxx：xxx"""


def generate_outline(kb: KnowledgeBase, topic: str = "全部内容") -> str:
    question = f"请为以下内容生成复习提纲：{topic}。要求覆盖核心知识点，区分重点和了解。"
    context = kb.search(topic, top_k=8)
    return ask_llm(OUTLINE_PROMPT, question, context)


# ============================================
# 3. 自动出题
# ============================================

EXAM_PROMPT = """你是一位大学教师，需要根据教学资料为学生出题。

你的出题规则：
1. 题目必须基于资料内容，不能脱离资料
2. 题型为选择题，每题4个选项（A/B/C/D）
3. 至少出5道题，覆盖不同知识点
4. 每道题标注正确答案和解析（为什么选这个）
5. 难度适中，不要出太偏的题

输出格式：
## 第1题
**题目**：xxx
A. xxx
B. xxx
C. xxx
D. xxx
**答案**：x
**解析**：xxx"""


def generate_exam(kb: KnowledgeBase, num_questions: int = 10) -> str:
    context = kb.search("重点 考试 核心", top_k=10)
    question = f"请根据资料出{num_questions}道选择题，覆盖核心知识点，标注答案和解析。"
    return ask_llm(EXAM_PROMPT, question, context)


# ============================================
# 4. 错题解析
# ============================================

ANALYSIS_PROMPT = """你是一位大学课程助教，学生做错了一道题，请你帮他分析。

你的任务：
1. 结合参考资料，解释为什么正确答案是对的
2. 解释学生选错的选项为什么不对
3. 关联资料中的相关知识点，帮助学生举一反三
4. 语气鼓励、耐心，不要打击学生

输出格式：
### 正确答案：x
### 为什么选 x：（结合资料解释）
### 你选的 y 为什么不对：
### 相关知识点：
- xxx"""


def analyze_wrong_answer(kb: KnowledgeBase, question_text: str, student_answer: str, correct_answer: str) -> str:
    context = kb.search(question_text, top_k=5)
    prompt_text = f"""学生遇到的题目：{question_text}
学生选的答案：{student_answer}
正确答案：{correct_answer}
请分析为什么学生选错了，并结合资料给出正确解释。"""
    return ask_llm(ANALYSIS_PROMPT, prompt_text, context)
