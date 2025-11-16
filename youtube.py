import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError as exc:  # pragma: no cover
    raise ImportError("langchain-openai 패키지가 필요합니다. 설치 후 다시 실행해주세요.") from exc

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None  # type: ignore

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None  # type: ignore

MODEL_OPTIONS = ["gpt-4o", "gpt-5", "gemini-2.5-pro", "claude-4-sonnet"]

PAGE_TITLE = "Youtube Q&A Chatbot"


def init_session_state() -> None:
    defaults = {
        "chat_history": [],
        "conversation_memory": [],
        "vectorstore": None,
        "retriever": None,
        "processed_url": None,
        "summary": None,
        "chunk_summaries": [],
        "is_processing": False,
        "selected_model": MODEL_OPTIONS[0],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_state() -> None:
    keys_to_reset = [
        "chat_history",
        "conversation_memory",
        "vectorstore",
        "retriever",
        "processed_url",
        "summary",
        "chunk_summaries",
    ]
    for key in keys_to_reset:
        st.session_state[key] = [] if isinstance(st.session_state.get(key), list) else None
    st.session_state["is_processing"] = False


def get_openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY")


def get_llm(model_name: str):
    if model_name in {"gpt-4o", "gpt-5"}:
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")
        return ChatOpenAI(model=model_name, temperature=0.2, api_key=api_key)

    if model_name == "gemini-2.5-pro":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai 패키지를 설치해주세요.")
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 또는 GEMINI_API_KEY 환경 변수가 필요합니다.")
        return ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.2)

    if model_name == "claude-4-sonnet":
        if ChatAnthropic is None:
            raise ImportError("langchain-anthropic 패키지를 설치해주세요.")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY 환경 변수가 필요합니다.")
        return ChatAnthropic(model="claude-4-sonnet", temperature=0.2, anthropic_api_key=api_key)

    raise ValueError(f"지원하지 않는 모델입니다: {model_name}")


def summarize_chunk(chunk: str, model_name: str, chunk_idx: int) -> str:
    llm = get_llm(model_name)
    messages = [
        SystemMessage(
            content="당신은 유튜브 전사 내용을 정돈된 요약으로 재구성하는 전문 비서입니다. 핵심 메시지와 주요 사실을 중심으로 요약하세요."
        ),
        HumanMessage(
            content=f"다음은 전사 청크 #{chunk_idx + 1}입니다.\n\n{chunk}\n\n위 내용을 5문장 이하로 간결하게 요약하고, 핵심 키워드 3개를 bullet 없이 한 문장으로 제시하세요."
        ),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def aggregate_summary(chunk_summaries: List[str], model_name: str) -> str:
    llm = get_llm(model_name)
    summaries_text = "\n\n".join(f"[청크 {idx + 1}] {summary}" for idx, summary in enumerate(chunk_summaries))
    messages = [
        SystemMessage(
            content="당신은 유튜브 영상 내용을 전체적으로 정리하는 요약 전문가입니다. 청중에게 이해하기 쉬운 구조화된 요약을 제공합니다."
        ),
        HumanMessage(
            content=(
                "다음은 유튜브 영상 전사 내용을 청크 단위로 요약한 결과 입니다.\n\n"
                f"{summaries_text}\n\n"
                "위 정보를 바탕으로 영상 전체 요약을 작성하세요. 형식은 다음을 따릅니다:\n"
                "1. 한 문장으로 핵심 메시지를 제시하는 '핵심 요약'\n"
                "2. 주요 내용을 3~5개 항목으로 정리한 '주요 내용'\n"
                "3. 시청자가 얻을 수 있는 인사이트나 다음 행동을 제안하는 '활용 포인트'\n"
                "각 항목은 문장형으로 작성하고 존댓말을 사용하세요."
            )
        ),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def build_vector_store(chunks: List[str]) -> Tuple[FAISS, Any]:
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("임베딩 생성을 위해 OPENAI_API_KEY 환경 변수가 필요합니다.")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    return vectorstore, retriever


def process_youtube_video(url: str, model_name: str) -> None:
    st.session_state["is_processing"] = True

    progress_text = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)

    try:
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 필요합니다.")

        progress_text.write("1/5 ▶ 동영상 오디오 다운로드 중...")
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = GenericLoader(
                YoutubeAudioLoader([url], temp_dir),
                OpenAIWhisperParser(api_key=api_key),
            )
            docs = loader.load()
        progress_bar.progress(20)

        progress_text.write("2/5 ▶ 전사 텍스트 정리 중...")
        combined_docs = [doc.page_content for doc in docs]
        transcript_text = "\n".join(combined_docs)

        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = splitter.split_text(transcript_text)
        if not splits:
            raise ValueError("전사 텍스트를 가져오지 못했습니다.")
        progress_bar.progress(35)

        progress_text.write("3/5 ▶ 청크 요약 병렬 수행 중...")
        chunk_summaries: List[str] = [""] * len(splits)
        with ThreadPoolExecutor(max_workers=min(4, len(splits))) as executor:
            futures = {
                executor.submit(summarize_chunk, chunk, model_name, idx): idx
                for idx, chunk in enumerate(splits)
            }
            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    chunk_summaries[idx] = future.result()
                except Exception as err:  # pragma: no cover
                    chunk_summaries[idx] = f"요약 실패: {err}"
                completed += 1
                progress_bar.progress(35 + int(30 * completed / total))
                progress_text.write(f"3/5 ▶ 청크 요약 병렬 수행 중... ({completed}/{total})")

        progress_text.write("4/5 ▶ 전체 요약 통합 중...")
        overall_summary = aggregate_summary(chunk_summaries, model_name)
        progress_bar.progress(75)

        progress_text.write("5/5 ▶ QA 인덱스 구축 중...")
        vectorstore, retriever = build_vector_store(splits)
        progress_bar.progress(100)

        st.session_state["vectorstore"] = vectorstore
        st.session_state["retriever"] = retriever
        st.session_state["processed_url"] = url
        st.session_state["summary"] = overall_summary
        st.session_state["chunk_summaries"] = chunk_summaries
        st.session_state["chat_history"] = []
        st.session_state["conversation_memory"] = []

        progress_text.success("동영상 처리가 완료되었습니다.")
    except Exception as error:
        reset_state()
        progress_text.error(f"처리 중 오류가 발생했습니다: {error}")
    finally:
        st.session_state["is_processing"] = False


def answer_question(prompt: str, model_name: str) -> str:
    if st.session_state.get("retriever") is None:
        raise ValueError("먼저 동영상을 처리해주세요.")

    retrieved_docs = st.session_state["retriever"].invoke(prompt)
    if not retrieved_docs:
        return "관련 정보를 찾지 못했습니다. 질문을 다시 작성해주실 수 있을까요?"

    context_text = ""
    for idx, doc in enumerate(retrieved_docs[:5], start=1):
        context_text += f"[참고 {idx}]\n{doc.page_content}\n\n"

    llm = get_llm(model_name)
    messages = [
        SystemMessage(
            content=(
                "당신은 유튜브 영상 내용을 바탕으로 질의에 답변하는 전문 어시스턴트입니다. "
                "제공된 참고 문서를 중심으로 정확하고 근거 있는 답변을 생성하세요."
            )
        ),
        HumanMessage(
            content=(
                f"질문: {prompt}\n\n"
                f"참고 문서:\n{context_text}\n"
                "위 자료를 기반으로 질문에 답변하세요. 존댓말을 사용하고, 필요한 경우 요약된 참고 근거를 함께 설명하세요."
            )
        ),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def render_css() -> None:
    st.markdown(
        """
<style>
.main-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: #1f77b4;
}
.summary-box {
    background-color: #f8f9fa;
    border: 1px solid #e1e4e8;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.chunk-expander > div {
    border: 1px solid #e1e4e8 !important;
    border-radius: 6px !important;
    margin-bottom: 0.5rem !important;
}
.stChatMessage {
    font-size: 0.95rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🎬", layout="wide")
    render_css()
    init_session_state()

    with st.sidebar:
        st.markdown("#### 모델 선택")
        selected_model = st.selectbox(
            "사용할 모델을 선택하세요", MODEL_OPTIONS, index=MODEL_OPTIONS.index(st.session_state["selected_model"])
        )
        st.session_state["selected_model"] = selected_model

        st.markdown("#### YouTube URL")
        youtube_url = st.text_input("동영상 URL을 입력하세요", value=st.session_state.get("processed_url") or "")

        process_disabled = st.session_state["is_processing"] or not youtube_url

        if st.button("동영상 처리하기", disabled=process_disabled):
            reset_state()
            process_youtube_video(youtube_url, selected_model)

        st.button("다시 시작하기", on_click=reset_state, type="secondary")

    st.markdown(f"<div class='main-title'>{PAGE_TITLE}</div>", unsafe_allow_html=True)

    if st.session_state.get("summary"):
        st.markdown("### 영상 요약")
        st.markdown(f"<div class='summary-box'>{st.session_state['summary']}</div>", unsafe_allow_html=True)
        with st.expander("청크별 요약 보기"):
            for idx, chunk_summary in enumerate(st.session_state.get("chunk_summaries", []), start=1):
                st.markdown(f"**청크 {idx} 요약**")
                st.write(chunk_summary)

    if st.session_state["chat_history"]:
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    if user_prompt := st.chat_input("질문을 입력하세요. (영상 처리 후 사용 가능)"):
        st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        try:
            answer = answer_question(user_prompt, st.session_state["selected_model"])
        except Exception as error:
            answer = f"응답 생성 중 오류가 발생했습니다: {error}"

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.session_state["conversation_memory"].append(f"사용자: {user_prompt}")
        st.session_state["conversation_memory"].append(f"AI: {answer}")


if __name__ == "__main__":
    main()

