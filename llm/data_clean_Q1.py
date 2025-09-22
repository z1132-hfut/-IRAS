"""
清洗处理微调简历匹配功能对应模型需要的数据
"""
import pdfplumber
import io
from typing import Dict, Any, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor


class DataCleanQ1:
    """
    PDF文件处理类：清洗出简历的各个部分信息
    """

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def extract_full_text(self, pdf_content: bytes) -> str:
        """
        提取PDF文件的全部文本内容
        :param pdf_content: PDF文件字节内容
        :return: 完整的文本内容字符串
        """
        try:
            # 异步执行PDF解析
            full_text = await self._extract_pdf_text(pdf_content)
            return full_text
        except Exception as e:
            raise Exception(f"提取PDF文本内容失败: {str(e)}")

    async def process_pdf(self, pdf_content: bytes, filename: str) -> Dict[str, str]:
        """
        处理单个PDF文件
        :param pdf_content: PDF文件字节内容
        :param filename: 文件名
        :return: 清洗出的pdf各部分
        """
        try:
            # 异步执行PDF解析
            resume_data = await self._extract_pdf_text(pdf_content)

            # 清洗和提取各个部分
            cleaned_data = self._clean_resume_data(resume_data, filename)

            return cleaned_data

        except Exception as e:
            return {
                "filename": filename,
                "basic_info": "暂无",
                "skills": "暂无",
                "education": "暂无",
                "internship": "暂无",
                "projects": "暂无",
                "campus_experience": "暂无",
                "certificates": "暂无",
                "self_evaluation": "暂无",
                "others": "暂无"
            }

    async def _extract_pdf_text(self, pdf_content: bytes) -> str:
        """
        异步提取PDF文本内容
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._sync_extract_pdf_text, pdf_content
        )

    def _sync_extract_pdf_text(self, pdf_content: bytes) -> str:
        """
        同步提取PDF文本内容
        """
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            raise Exception(f"PDF解析错误: {str(e)}")
        return text

    def _clean_resume_data(self, raw_text: str, filename: str) -> Dict[str, str]:
        """
        清洗和提取简历的各个部分信息
        :param raw_text: 原始文本
        :return: 结构化的简历数据
        """
        cleaned_data = {
            "filename": filename,
            "basic_info": self._extract_basic_info(raw_text),
            "skills": self._extract_section_improved(raw_text, ["技能", "技术能力", "掌握技术", "专业技能"]),
            "education": self._extract_section_improved(raw_text, ["教育背景", "学历", "教育经历"]),
            "internship": self._extract_section_improved(raw_text, ["实习经历", "实习经验"]),
            "projects": self._extract_section_improved(raw_text, ["项目经历", "项目经验"]),
            "campus_experience": self._extract_section_improved(raw_text, ["校园经历", "校内实践", "社会实践"]),
            "certificates": self._extract_section_improved(raw_text, ["证书", "获奖", "荣誉", "比赛"]),
            "self_evaluation": self._extract_section_improved(raw_text, ["自我评价", "个人评价"]),
            "others": self._extract_section_improved(raw_text, ["其他", "附加信息"])
        }

        return cleaned_data

    def _extract_basic_info(self, text: str) -> str:
        """
        专门提取基本信息（通常在简历开头）
        """
        lines = text.split('\n')
        basic_info_lines = []

        # 常见的基本信息关键词
        basic_keywords = ["姓名", "电话", "邮箱", "出生", "民族", "年龄", "毕业院校", "学历", "专业"]

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if any(keyword in line_stripped for keyword in basic_keywords):
                basic_info_lines.append(line_stripped)
            # 基本信息通常在简历开头，遇到第一个非基本信息行就停止
            elif basic_info_lines and i > 10:  # 假设基本信息在前10行内
                break

        return '\n'.join(basic_info_lines)

    def _extract_section_improved(self, text: str, keywords: List[str]) -> str:
        """
        改进的分段提取方法
        """
        lines = text.split('\n')
        section_lines = []
        in_target_section = False

        # 所有可能的部分标题
        all_section_keywords = [
            "个人信息", "基本信息", "技能", "技术能力", "掌握技术", "专业技能",
            "教育背景", "学历", "教育经历", "实习经历", "实习经验",
            "项目经历", "项目经验", "校园经历", "校内实践", "社会实践",
            "证书", "获奖", "荣誉", "比赛", "自我评价", "个人评价",
            "其他", "附加信息"
        ]

        for line in lines:
            line_stripped = line.strip()

            # 检查是否是任何部分的标题
            is_section_header = any(keyword in line_stripped for keyword in all_section_keywords)

            if is_section_header:
                # 检查是否是目标部分
                if any(keyword in line_stripped for keyword in keywords):
                    in_target_section = True
                    continue
                # 如果是其他部分的标题，且当前在目标部分中，则结束目标部分
                elif in_target_section:
                    break

            # 如果在目标部分中，收集内容
            if in_target_section and line_stripped and not is_section_header:
                section_lines.append(line_stripped)

        return '\n'.join(section_lines)

    async def close(self):
        """清理资源"""
        self.executor.shutdown()