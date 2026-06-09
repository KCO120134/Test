import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from openai import OpenAI
import pdfplumber
import io
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
import os
import zipfile
import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(
    page_title="제지 업종 AI 분석 대시보드",
    page_icon="📄",
    layout="wide",
)

# ── 폰트 & 스타일 ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif !important; }

.stock-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 10px;
}
.card-name  { font-size: 13px; color: #666; margin-bottom: 8px; }
.card-price { font-size: 22px; font-weight: 700; color: #111; }
.card-up    { font-size: 15px; font-weight: 700; color: #FF4444; margin-top: 4px; }
.card-down  { font-size: 15px; font-weight: 700; color: #0066CC; margin-top: 4px; }
.card-flat  { font-size: 15px; font-weight: 700; color: #888;    margin-top: 4px; }
.card-vol   { font-size: 11px; color: #999; margin-top: 6px; }

.chat-wrap {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 14px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# ── 종목 정의 (제지 5개) ───────────────────────────────────
PAPER_STOCKS = {
    "한솔제지":  "213500.KS",
    "무림페이퍼": "009580.KS",
    "아세아제지": "002310.KS",
    "깨끗한나라": "004540.KS",
    "신풍제지":  "002870.KS",
}

PLOTLY_FONT = dict(family="Noto Sans KR, sans-serif", size=13)

# ── yfinance 세션 설정 (재시도 + User-Agent) ───────────────
def _make_yf_session():
    """재시도 로직과 User-Agent가 설정된 requests 세션 반환"""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })
    return session

# ── 데이터 로딩 함수 ───────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_quotes(tickers: dict) -> pd.DataFrame:
    rows = []
    sym_list = list(tickers.values())
    name_map = {v: k for k, v in tickers.items()}

    # yf.download()로 한 번에 가격 데이터 수집 (더 안정적)
    try:
        raw = yf.download(
            sym_list,
            period="5d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        raw = pd.DataFrame()

    for sym in sym_list:
        name = name_map[sym]
        try:
            # 단일 종목이면 컬럼이 단순, 복수면 MultiIndex
            if len(sym_list) == 1:
                close_col  = "Close"
                volume_col = "Volume"
                hist_close  = raw[close_col].dropna()
                hist_volume = raw[volume_col].dropna()
            else:
                hist_close  = raw["Close"][sym].dropna()
                hist_volume = raw["Volume"][sym].dropna()

            if len(hist_close) < 2:
                raise ValueError("insufficient data")

            cur  = float(hist_close.iloc[-1])
            prev = float(hist_close.iloc[-2])
            chg  = cur - prev
            chg_pct = chg / prev * 100
            vol  = int(hist_volume.iloc[-1])

            # info는 별도 try/except (실패해도 가격 데이터는 표시)
            info = {}
            try:
                time.sleep(0.3)
                t = yf.Ticker(sym, session=_make_yf_session())
                info = t.fast_info  # fast_info는 info보다 가볍고 안정적
                market_cap = getattr(info, "market_cap", 0) or 0
                week52_high = getattr(info, "year_high", 0) or 0
                week52_low  = getattr(info, "year_low",  0) or 0
                per = None
                pbr = None
            except Exception:
                market_cap = 0
                week52_high = 0
                week52_low  = 0
                per = None
                pbr = None

            rows.append({
                "종목명":    name,
                "티커":     sym,
                "현재가":   int(cur),
                "전일대비": int(chg),
                "등락률":   round(chg_pct, 2),
                "거래량":   vol,
                "시가총액": market_cap,
                "52주최고": week52_high,
                "52주최저": week52_low,
                "PER":      per,
                "PBR":      pbr,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

@st.cache_data(ttl=300)
def fetch_history(tickers: dict, period_days: int) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=period_days)
    sym_list  = list(tickers.values())
    name_map  = {v: k for k, v in tickers.items()}

    try:
        raw = yf.download(
            sym_list,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    frames = []
    for sym in sym_list:
        name = name_map[sym]
        try:
            if len(sym_list) == 1:
                df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                df = raw.xs(sym, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = df.dropna(subset=["Close"])
            if df.empty:
                continue
            df["종목"] = name
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames) if frames else pd.DataFrame()

# ── GPT 컨텍스트 빌더 ─────────────────────────────────────
def build_context(quotes: pd.DataFrame, hist: pd.DataFrame) -> str:
    lines = [
        f"[분석 대상] 한국 제지 업종 5개 종목 (조회일: {datetime.now().strftime('%Y-%m-%d')})",
        "",
        "[현재 시세]",
    ]
    for _, r in quotes.iterrows():
        sign = "▲" if r["등락률"] > 0 else ("▼" if r["등락률"] < 0 else "-")
        per  = f"{r['PER']:.1f}" if r["PER"] else "N/A"
        pbr  = f"{r['PBR']:.2f}" if r["PBR"] else "N/A"
        mcap = f"{r['시가총액']/1e8:.0f}억원" if r["시가총액"] > 0 else "N/A"
        lines.append(
            f"  • {r['종목명']}: {r['현재가']:,}원 ({sign}{abs(r['등락률']):.2f}%) | "
            f"거래량 {r['거래량']:,} | 시가총액 {mcap} | "
            f"PER {per} | PBR {pbr} | "
            f"52주최고 {r['52주최고']:,.0f}원 / 52주최저 {r['52주최저']:,.0f}원"
        )

    if not hist.empty:
        lines.append("")
        lines.append("[기간 내 수익률]")
        for name, grp in hist.groupby("종목"):
            grp = grp.sort_index()
            ret = (grp["Close"].iloc[-1] / grp["Close"].iloc[0] - 1) * 100
            vol = grp["Close"].pct_change().std() * 100
            lines.append(f"  • {name}: 수익률 {ret:+.2f}%, 일간 변동성 {vol:.2f}%")

    return "\n".join(lines)

# ── 기업 뉴스 수집 (Google News RSS) ──────────────────────
@st.cache_data(ttl=600)
def fetch_news(company: str, limit: int = 20) -> list:
    q = urllib.parse.quote(f"{company} 주식")
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    articles = []
    try:
        data = urllib.request.urlopen(req, timeout=15).read()
        root = ET.fromstring(data)
        for item in root.findall(".//item")[:limit]:
            title_raw = item.findtext("title", "")
            if " - " in title_raw:
                title, src_in_title = title_raw.rsplit(" - ", 1)
            else:
                title, src_in_title = title_raw, ""
            link = item.findtext("link", "")
            source_el = item.find("source")
            source = source_el.text if source_el is not None else src_in_title
            pub_raw = item.findtext("pubDate", "")
            try:
                pub_dt = parsedate_to_datetime(pub_raw)
                pub_str = pub_dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pub_str = pub_raw
            articles.append({
                "title":   title.strip(),
                "link":    link.strip(),
                "source":  (source or "").strip(),
                "pub":     pub_str,
            })
    except Exception:
        return []
    return articles

def build_news_context(all_news: dict) -> str:
    """모든 종목의 뉴스를 GPT 컨텍스트 문자열로 변환."""
    lines = [f"[수집된 뉴스 데이터 — 조회일: {datetime.now().strftime('%Y-%m-%d')}]"]
    for company, articles in all_news.items():
        if not articles:
            continue
        lines.append(f"\n## {company} ({len(articles)}건)")
        for i, a in enumerate(articles, 1):
            lines.append(f"  {i}. [{a['pub']}] {a['title']} ({a['source']})")
    return "\n".join(lines)

# ── OpenDART: 기업 고유번호(corp_code) 로딩 ───────────────
CORPCODE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpcode.csv")

@st.cache_data(show_spinner=False)
def load_corp_codes(dart_key: str) -> pd.DataFrame:
    """corpcode.csv가 있으면 읽고, 없으면 OpenDART API로 받아 생성한다.
    반환 컬럼: corp_code, corp_name, stock_code"""
    # 1) 로컬 CSV 우선
    if os.path.exists(CORPCODE_CSV):
        df = pd.read_csv(CORPCODE_CSV, dtype=str).fillna("")
        # 컬럼명 정규화
        cols = {c.lower().strip(): c for c in df.columns}
        rename = {}
        for want in ["corp_code", "corp_name", "stock_code"]:
            if want in cols:
                rename[cols[want]] = want
        df = df.rename(columns=rename)
        df["stock_code"] = df.get("stock_code", "").astype(str).str.zfill(6).str.strip()
        df["corp_code"]  = df["corp_code"].astype(str).str.zfill(8)
        return df

    # 2) API로 corpCode.xml(zip) 다운로드 후 파싱 → CSV 저장
    if not dart_key:
        return pd.DataFrame(columns=["corp_code", "corp_name", "stock_code"])
    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={dart_key}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=30).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        xml_name = zf.namelist()[0]
        xml_bytes = zf.read(xml_name)
    root = ET.fromstring(xml_bytes)
    rows = []
    for el in root.findall(".//list"):
        rows.append({
            "corp_code":  (el.findtext("corp_code") or "").strip(),
            "corp_name":  (el.findtext("corp_name") or "").strip(),
            "stock_code": (el.findtext("stock_code") or "").strip(),
        })
    df = pd.DataFrame(rows)
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    # 상장사만 CSV로 저장(파일 크기 절감) — 전체는 메모리 유지
    try:
        df[df["stock_code"] != ""].to_csv(CORPCODE_CSV, index=False, encoding="utf-8-sig")
    except Exception:
        pass
    df["stock_code"] = df["stock_code"].str.zfill(6).where(df["stock_code"] != "", "")
    return df

def find_corp_code(corp_df: pd.DataFrame, ticker: str, name: str) -> str:
    """yfinance 티커(예: '213500.KS')의 종목코드로 corp_code 매칭."""
    if corp_df.empty:
        return ""
    stock_no = ticker.split(".")[0].zfill(6)
    hit = corp_df[corp_df["stock_code"] == stock_no]
    if not hit.empty:
        return hit.iloc[0]["corp_code"]
    # 종목코드 매칭 실패 시 회사명 부분일치 시도
    hit = corp_df[corp_df["corp_name"].str.replace(" ", "") == name.replace(" ", "")]
    if not hit.empty:
        return hit.iloc[0]["corp_code"]
    return ""

@st.cache_data(ttl=600, show_spinner=False)
def fetch_disclosures(dart_key: str, corp_code: str, bgn_de: str, end_de: str, count: int = 20) -> dict:
    """OpenDART 공시검색 API. 반환: {'status':..., 'items':[...]}"""
    params = {
        "crtfc_key": dart_key,
        "corp_code": corp_code,
        "bgn_de": bgn_de,
        "end_de": end_de,
        "page_no": "1",
        "page_count": str(count),
    }
    url = "https://opendart.fss.or.kr/api/list.json?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=20).read().decode("utf-8"))
    except Exception as e:
        return {"status": "ERR", "message": str(e), "items": []}
    return {
        "status":  data.get("status"),
        "message": data.get("message"),
        "items":   data.get("list", []),
    }

# ── PDF 텍스트 추출 ────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    text_pages = []
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_pages.append(f"[{i+1}페이지]\n{page_text.strip()}")
    return "\n\n".join(text_pages)

# ── GPT 호출 ──────────────────────────────────────────────
def ask_gpt(api_key: str, messages: list) -> str:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=1200,
    )
    return resp.choices[0].message.content

def friendly_openai_error(e: Exception) -> str:
    """OpenAI 예외를 사용자 친화적 한국어 메시지로 변환."""
    msg = str(e)
    if "401" in msg or "invalid_api_key" in msg or "Incorrect API key" in msg:
        return ("❌ OpenAI API 키가 올바르지 않습니다 (401).\n\n"
                "- 키가 `sk-`로 시작하는 유효한 OpenAI 키인지 확인하세요.\n"
                "- **OpenDART 인증키를 OpenAI 키 칸에 잘못 입력**하지 않았는지 확인하세요.\n"
                "- 키는 https://platform.openai.com/api-keys 에서 확인/재발급할 수 있습니다.")
    if "429" in msg or "insufficient_quota" in msg or "quota" in msg:
        return ("❌ 사용 한도를 초과했거나 크레딧이 부족합니다 (429).\n\n"
                "OpenAI 계정의 결제/사용량 설정을 확인하세요.")
    if "rate_limit" in msg:
        return "❌ 요청이 너무 잦습니다. 잠시 후 다시 시도하세요."
    return f"❌ OpenAI 호출 중 오류가 발생했습니다:\n\n{msg}"

# ══════════════════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════════════════
st.sidebar.header("⚙️ 설정")

# ── [1] PDF 업로드 — 사이드바 최상단 ──────────────────────
st.sidebar.subheader("📎 PDF 문서 업로드")
uploaded_pdf = st.sidebar.file_uploader(
    "PDF 파일을 선택하세요 (.pdf)",
    type=["pdf"],
    key="pdf_uploader",
)
if uploaded_pdf is not None:
    if st.session_state.get("pdf_filename") != uploaded_pdf.name:
        with st.spinner("PDF 텍스트 추출 중..."):
            try:
                pdf_text = extract_pdf_text(uploaded_pdf)
                st.session_state["pdf_text"]     = pdf_text
                st.session_state["pdf_filename"] = uploaded_pdf.name
                st.session_state["chat_history"] = []
            except Exception as e:
                st.sidebar.error(f"PDF 파싱 오류: {e}")
                st.session_state.pop("pdf_text", None)
    if st.session_state.get("pdf_text"):
        char_cnt = len(st.session_state["pdf_text"])
        st.sidebar.success(f"✅ {uploaded_pdf.name} ({char_cnt:,}자)")
        with st.sidebar.expander("📄 미리보기"):
            st.sidebar.text(st.session_state["pdf_text"][:800] + ("..." if char_cnt > 800 else ""))
        if st.sidebar.button("🗑️ PDF 제거"):
            st.session_state.pop("pdf_text", None)
            st.session_state.pop("pdf_filename", None)
            st.rerun()
else:
    st.session_state.pop("pdf_text", None)
    st.session_state.pop("pdf_filename", None)

st.sidebar.markdown("---")

# ── [2] OpenAI API Key ─────────────────────────────────────
st.sidebar.subheader("🔑 OpenAI API Key")
api_key = st.sidebar.text_input(
    "API Key 입력", type="password", placeholder="sk-...",
    help="OpenAI 키는 'sk-'로 시작합니다. DART 인증키와 혼동하지 마세요.",
)
api_key = api_key.strip() if api_key else ""
if api_key:
    if not api_key.startswith("sk-"):
        st.sidebar.error("⚠️ OpenAI 키는 'sk-'로 시작해야 합니다. "
                         "DART 인증키를 잘못 입력하지 않았는지 확인하세요.")
        st.session_state.pop("api_key", None)
    else:
        st.session_state["api_key"] = api_key
        st.sidebar.success("API Key 설정 완료")
else:
    st.session_state.pop("api_key", None)

st.sidebar.markdown("---")

# ── [3] OpenDART API Key ───────────────────────────────────
st.sidebar.subheader("🏛️ OpenDART API Key")
dart_key = st.sidebar.text_input(
    "DART 인증키 입력", type="password", placeholder="40자리 인증키",
    help="공시 정보 조회에 사용됩니다. opendart.fss.or.kr 에서 발급",
)
if dart_key:
    st.session_state["dart_key"] = dart_key
    st.sidebar.success("DART 키 설정 완료")
else:
    st.session_state.pop("dart_key", None)

st.sidebar.markdown("---")

# ── [3] 기간 선택 ──────────────────────────────────────────
st.sidebar.subheader("📅 조회 기간")
period_map = {"7일": 7, "30일": 30, "90일": 90}
period_label = st.sidebar.radio("기간 선택", list(period_map.keys()), index=1)
period_days = period_map[period_label]

st.sidebar.markdown("---")
st.sidebar.caption("데이터 출처: Yahoo Finance\n모델: GPT-4o-mini")

# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════
st.title("📄 제지 업종 AI 분석 대시보드")
st.markdown("한솔제지 · 무림페이퍼 · 아세아제지 · 깨끗한나라 · 신풍제지")
st.markdown("---")

# 데이터 로딩
with st.spinner("주식 데이터 로딩 중..."):
    quotes = fetch_quotes(PAPER_STOCKS)
    hist_df = fetch_history(PAPER_STOCKS, period_days)

if quotes.empty:
    st.error("데이터를 불러오지 못했습니다. 네트워크 상태를 확인하세요.")
    st.stop()

# ── 섹션 1: 종목 카드 ─────────────────────────────────────
st.subheader("📊 종목별 현재 시세")
cols = st.columns(5)
for i, (_, r) in enumerate(quotes.iterrows()):
    pct  = r["등락률"]
    sign = "▲" if pct > 0 else ("▼" if pct < 0 else "")
    css  = "card-up" if pct > 0 else ("card-down" if pct < 0 else "card-flat")
    with cols[i]:
        st.markdown(f"""
        <div class="stock-card">
            <div class="card-name">{r['종목명']}</div>
            <div class="card-price">{r['현재가']:,}원</div>
            <div class="{css}">{sign} {abs(pct):.2f}%</div>
            <div class="card-vol">거래량 {r['거래량']:,}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── 섹션 2: 차트 (종가 추이 | 캔들 | 변동성) ─────────────
st.subheader(f"📈 주가 분석 ({period_label})")
tab1, tab2, tab3 = st.tabs(["종가 추이", "캔들 차트", "변동성 비교"])

with tab1:
    if not hist_df.empty:
        fig = go.Figure()
        for name, grp in hist_df.groupby("종목"):
            grp = grp.sort_index()
            fig.add_trace(go.Scatter(
                x=grp.index, y=grp["Close"],
                mode="lines", name=name,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}원<extra>%{fullData.name}</extra>",
            ))
        fig.update_layout(
            height=420, hovermode="x unified", font=PLOTLY_FONT,
            xaxis_title="날짜", yaxis_title="종가 (원)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#fafafa", paper_bgcolor="#fff",
            xaxis=dict(gridcolor="#eee"), yaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    stock_sel = st.selectbox("종목 선택", list(PAPER_STOCKS.keys()), key="candle_sel")
    grp = hist_df[hist_df["종목"] == stock_sel].sort_index()
    if not grp.empty:
        candle = go.Figure(go.Candlestick(
            x=grp.index,
            open=grp["Open"], high=grp["High"],
            low=grp["Low"],   close=grp["Close"],
            increasing_line_color="#FF4444",
            decreasing_line_color="#0066CC",
        ))
        candle.update_layout(
            height=450, font=PLOTLY_FONT,
            xaxis_title="날짜", yaxis_title="가격 (원)",
            xaxis_rangeslider_visible=False,
            plot_bgcolor="#fafafa", paper_bgcolor="#fff",
        )
        st.plotly_chart(candle, use_container_width=True)

        vol_bar = px.bar(grp, x=grp.index, y="Volume",
                         labels={"Volume": "거래량", "index": "날짜"},
                         title="거래량")
        vol_bar.update_layout(height=220, font=PLOTLY_FONT,
                               plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(vol_bar, use_container_width=True)

with tab3:
    if not hist_df.empty:
        vol_rows = []
        for name, grp in hist_df.groupby("종목"):
            grp = grp.sort_index()
            vol = grp["Close"].pct_change().std() * 100
            ret = (grp["Close"].iloc[-1] / grp["Close"].iloc[0] - 1) * 100
            vol_rows.append({"종목": name, "변동성(%)": round(vol, 3), "수익률(%)": round(ret, 2)})
        vol_df = pd.DataFrame(vol_rows).sort_values("변동성(%)", ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            bar_v = px.bar(vol_df, x="종목", y="변동성(%)",
                           text="변동성(%)", color="종목",
                           title="일간 변동성 (표준편차)")
            bar_v.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            bar_v.update_layout(height=380, showlegend=False, font=PLOTLY_FONT,
                                 plot_bgcolor="#fafafa", paper_bgcolor="#fff",
                                 yaxis=dict(gridcolor="#eee"))
            st.plotly_chart(bar_v, use_container_width=True)
        with c2:
            colors = ["#FF4444" if v >= 0 else "#0066CC" for v in vol_df["수익률(%)"]]
            bar_r = go.Figure(go.Bar(
                x=vol_df["종목"], y=vol_df["수익률(%)"],
                marker_color=colors,
                text=vol_df["수익률(%)"].apply(lambda v: f"{v:+.2f}%"),
                textposition="outside",
            ))
            bar_r.update_layout(
                title=f"기간 수익률 ({period_label})", height=380,
                showlegend=False, font=PLOTLY_FONT,
                plot_bgcolor="#fafafa", paper_bgcolor="#fff",
                xaxis_title="종목", yaxis_title="수익률 (%)",
                yaxis=dict(gridcolor="#eee"),
            )
            st.plotly_chart(bar_r, use_container_width=True)

st.markdown("---")

# ── 섹션 3: 상세 데이터 테이블 ───────────────────────────
with st.expander("📋 상세 시세 테이블 보기"):
    disp = quotes.copy()
    disp["현재가"] = disp["현재가"].apply(lambda x: f"{x:,}원")
    disp["전일대비"] = disp["전일대비"].apply(lambda x: f"{x:+,}원")
    disp["등락률"] = disp["등락률"].apply(lambda x: f"{x:+.2f}%")
    disp["거래량"] = disp["거래량"].apply(lambda x: f"{x:,}")
    disp["시가총액"] = disp["시가총액"].apply(lambda x: f"{x/1e8:.0f}억원" if x > 0 else "-")
    disp["52주최고"] = disp["52주최고"].apply(lambda x: f"{x:,.0f}원" if x > 0 else "-")
    disp["52주최저"] = disp["52주최저"].apply(lambda x: f"{x:,.0f}원" if x > 0 else "-")
    disp["PER"] = disp["PER"].apply(lambda x: f"{x:.1f}" if x else "-")
    disp["PBR"] = disp["PBR"].apply(lambda x: f"{x:.2f}" if x else "-")
    st.dataframe(disp.drop(columns=["티커"]), use_container_width=True, hide_index=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# 섹션 4: 기업 뉴스
# ══════════════════════════════════════════════════════════
st.subheader("📰 기업 관련 뉴스")

nc1, nc2, nc3 = st.columns([2, 1, 1])
news_company = nc1.selectbox("종목 선택", list(PAPER_STOCKS.keys()), key="news_company")
news_count   = nc2.selectbox("표시 개수", [5, 10, 15, 20], index=1, key="news_count")
fetch_all_btn = nc3.button("🔄 전체 종목 수집", help="5개 종목 뉴스를 한번에 수집하여 챗봇이 모두 참고할 수 있게 합니다.")

# 선택 종목 뉴스 표시
with st.spinner(f"{news_company} 뉴스를 불러오는 중..."):
    articles = fetch_news(news_company, limit=news_count)

if not articles:
    st.warning(f"'{news_company}' 관련 뉴스를 불러오지 못했습니다.")
else:
    st.caption(f"'{news_company}' 관련 최신 뉴스 {len(articles)}건 (출처: Google News)")
    for a in articles:
        meta = " · ".join([x for x in [a["source"], a["pub"]] if x])
        st.markdown(
            f"**[{a['title']}]({a['link']})**  \n"
            f"<span style='color:#888; font-size:12px;'>{meta}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

# 전체 종목 뉴스 수집
if fetch_all_btn:
    all_news = {}
    prog = st.progress(0, text="전체 종목 뉴스 수집 중...")
    for idx, company in enumerate(PAPER_STOCKS.keys()):
        prog.progress((idx + 1) / len(PAPER_STOCKS), text=f"{company} 뉴스 수집 중...")
        all_news[company] = fetch_news(company, limit=20)
    prog.empty()
    st.session_state["all_news"] = all_news
    total = sum(len(v) for v in all_news.values())
    st.success(f"✅ 5개 종목 총 {total}건 뉴스 수집 완료 — 챗봇이 전체 뉴스를 참고합니다.")

# 세션에 뉴스 저장 (선택 종목 + 이미 수집된 전체 종목)
all_news_session = st.session_state.get("all_news", {})
all_news_session[news_company] = articles   # 현재 선택 종목은 항상 최신 유지
st.session_state["all_news"] = all_news_session

st.markdown("---")

# ══════════════════════════════════════════════════════════
# 섹션 5: OpenDART 공시 정보
# ══════════════════════════════════════════════════════════
st.subheader("🏛️ 기업 공시 정보 (OpenDART)")

dart_key_now = st.session_state.get("dart_key", "")

if not dart_key_now:
    st.info("👈 사이드바에서 OpenDART API Key를 입력하면 공시 정보를 조회할 수 있습니다. "
            "(인증키 발급: https://opendart.fss.or.kr)")
else:
    dc1, dc2, dc3 = st.columns([2, 1, 1])
    disc_company = dc1.selectbox("종목 선택", list(PAPER_STOCKS.keys()), key="disc_company")
    disc_months  = dc2.selectbox("조회 기간", [3, 6, 12], index=1,
                                 format_func=lambda m: f"최근 {m}개월", key="disc_months")
    disc_count   = dc3.selectbox("표시 개수", [10, 20, 30, 50], index=1, key="disc_count")

    # corp_code 로딩
    corp_df = pd.DataFrame()
    try:
        with st.spinner("기업 고유번호(corp_code) 로딩 중..."):
            corp_df = load_corp_codes(dart_key_now)
    except Exception as e:
        st.error(f"corp_code 로딩 실패: {e}")

    if corp_df.empty:
        st.warning("corpcode.csv가 없고 자동 다운로드도 실패했습니다. DART 인증키를 확인해주세요.")
    else:
        ticker = PAPER_STOCKS[disc_company]
        corp_code = find_corp_code(corp_df, ticker, disc_company)

        if not corp_code:
            st.warning(f"'{disc_company}'(종목코드 {ticker.split('.')[0]})의 corp_code를 찾지 못했습니다.")
        else:
            end_de = datetime.today().strftime("%Y%m%d")
            bgn_de = (datetime.today() - timedelta(days=disc_months * 30)).strftime("%Y%m%d")
            with st.spinner(f"{disc_company} 공시 조회 중..."):
                result = fetch_disclosures(dart_key_now, corp_code, bgn_de, end_de, disc_count)

            status = result.get("status")
            if status == "000":
                items = result["items"]
                st.caption(f"'{disc_company}' (corp_code: {corp_code}) · "
                           f"{bgn_de}~{end_de} · 공시 {len(items)}건")
                table_rows = []
                for it in items:
                    rcept_no = it.get("rcept_no", "")
                    viewer = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"
                    rcept_dt = it.get("rcept_dt", "")
                    dt_fmt = (f"{rcept_dt[:4]}-{rcept_dt[4:6]}-{rcept_dt[6:]}"
                              if len(rcept_dt) == 8 else rcept_dt)
                    table_rows.append({
                        "접수일자": dt_fmt,
                        "보고서명": it.get("report_nm", ""),
                        "제출인":   it.get("flr_nm", ""),
                        "바로가기": viewer,
                    })
                disc_df = pd.DataFrame(table_rows)
                st.dataframe(
                    disc_df, use_container_width=True, hide_index=True,
                    column_config={
                        "바로가기": st.column_config.LinkColumn("바로가기", display_text="📄 열기"),
                        "보고서명": st.column_config.TextColumn("보고서명", width="large"),
                    },
                )
                # 챗봇 컨텍스트용 저장
                st.session_state["latest_disclosures"] = {
                    "company": disc_company,
                    "items": [f"{r['접수일자']} | {r['보고서명']} ({r['제출인']})" for r in table_rows],
                }
            elif status == "013":
                st.info(f"'{disc_company}'의 해당 기간 공시가 없습니다.")
                st.session_state.pop("latest_disclosures", None)
            else:
                st.error(f"DART API 오류 [{status}]: {result.get('message', '')}")

st.markdown("---")

# ══════════════════════════════════════════════════════════
# 섹션 6: GPT 챗봇 (주식 분석 탭 / 뉴스 Q&A 탭)
# ══════════════════════════════════════════════════════════
st.subheader("🤖 AI 분석 챗봇 (GPT-4o-mini)")

api_key_now = st.session_state.get("api_key", "")

if not api_key_now:
    st.info("👈 사이드바에서 OpenAI API Key를 입력하면 챗봇을 사용할 수 있습니다.")
else:
    # ── 공통 컨텍스트 빌드 ──────────────────────────────────
    stock_context = build_context(quotes, hist_df)
    pdf_text     = st.session_state.get("pdf_text", "")
    pdf_filename = st.session_state.get("pdf_filename", "")

    pdf_section = ""
    if pdf_text:
        clipped = pdf_text[:6000] + ("\n...(이하 생략)" if len(pdf_text) > 6000 else "")
        pdf_section = f"\n\n[업로드된 PDF 문서: {pdf_filename}]\n{clipped}"

    disc_section = ""
    latest_disc = st.session_state.get("latest_disclosures")
    if latest_disc and latest_disc.get("items"):
        joined = "\n".join(f"  - {d}" for d in latest_disc["items"])
        disc_section = f"\n\n[{latest_disc['company']} 최근 공시 목록 (OpenDART)]\n{joined}"

    all_news_ctx = st.session_state.get("all_news", {})
    news_context_str = build_news_context(all_news_ctx) if all_news_ctx else ""

    # ── 컨텍스트 상태 배지 ──────────────────────────────────
    b1, b2, b3, b4 = st.columns(4)
    b1.success("✅ 주식 데이터")
    news_count_all = sum(len(v) for v in all_news_ctx.values())
    if news_count_all > 0:
        b2.success(f"✅ 뉴스 {news_count_all}건")
    else:
        b2.info("📰 뉴스 1종목")
    if disc_section:
        b3.success("✅ 공시")
    else:
        b3.info("🏛️ 공시 미조회")
    if pdf_text:
        b4.success("✅ PDF")
    else:
        b4.info("📎 PDF 미업로드")

    st.markdown("")

    # ── 탭 구성 ─────────────────────────────────────────────
    chat_tab1, chat_tab2 = st.tabs(["📊 주식·공시·PDF 분석", "📰 뉴스 Q&A"])

    # ════════════════════════════════════
    # 탭 1: 주식 / 공시 / PDF 분석
    # ════════════════════════════════════
    with chat_tab1:
        system_prompt_stock = (
            "당신은 한국 제지 업종 전문 주식 분석 AI입니다. "
            "아래 실시간 주식 데이터"
            + (", PDF 문서" if pdf_text else "")
            + (", 기업 공시 목록" if disc_section else "")
            + "를 기반으로 사용자 질문에 한국어로 정확하게 답변하세요. "
            "제공된 데이터 범위 밖의 내용은 추측하지 말고 '데이터에 없는 정보입니다'라고 명시하세요.\n\n"
            + stock_context + disc_section + pdf_section
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.markdown("**💡 추천 질문**")
        pq_cols = st.columns(4)
        preset_qs = [
            "오늘 가장 많이 오른 종목은?",
            "변동성이 가장 높은 종목은?",
            "시가총액 1위 종목은?",
            "52주 최저가에 가장 근접한 종목은?",
        ]
        for i, q in enumerate(preset_qs):
            if pq_cols[i].button(q, key=f"stock_preset_{i}"):
                st.session_state["stock_preset_input"] = q

        st.markdown("")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        preset_val = st.session_state.pop("stock_preset_input", None)
        user_input = st.chat_input("주식·공시·PDF에 대해 질문하세요...", key="chat_stock") or preset_val

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            msgs = [{"role": "system", "content": system_prompt_stock}]
            msgs += [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]
            with st.chat_message("assistant"):
                with st.spinner("GPT-4o-mini 분석 중..."):
                    try:
                        answer = ask_gpt(api_key_now, msgs)
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(friendly_openai_error(e))
                        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                            st.session_state.chat_history.pop()

        if st.session_state.get("chat_history"):
            if st.button("🗑️ 대화 초기화", key="reset_stock"):
                st.session_state.chat_history = []
                st.rerun()

    # ════════════════════════════════════
    # 탭 2: 뉴스 Q&A
    # ════════════════════════════════════
    with chat_tab2:
        if not news_context_str and not all_news_ctx:
            st.info("📰 뉴스 섹션에서 **'🔄 전체 종목 수집'** 버튼을 누르거나 "
                    "종목을 선택하면 뉴스 데이터가 준비됩니다.")
        else:
            # 뉴스가 1종목만 있을 때도 자연스럽게 처리
            if not news_context_str:
                single = list(all_news_ctx.items())[0]
                news_context_str = build_news_context({single[0]: single[1]})

            system_prompt_news = (
                "당신은 한국 제지 업종 뉴스 분석 전문 AI입니다. "
                "아래에 수집된 뉴스 헤드라인 데이터만을 근거로 사용자 질문에 한국어로 답변하세요. "
                "뉴스에 없는 내용은 '수집된 뉴스에 없는 정보입니다'라고 명시하세요. "
                "뉴스 요약, 주요 이슈 파악, 종목별 이슈 비교, 투자 시사점 도출 등에 집중하세요.\n\n"
                + news_context_str
            )

            if "news_chat_history" not in st.session_state:
                st.session_state.news_chat_history = []

            # 뉴스 전용 추천 질문
            st.markdown("**💡 추천 질문**")
            nq_cols = st.columns(4)
            news_preset_qs = [
                "수집된 뉴스를 종목별로 요약해줘",
                "가장 많이 언급된 이슈는?",
                "긍정적인 뉴스가 많은 종목은?",
                "최근 주요 공시·실적 관련 뉴스는?",
            ]
            for i, q in enumerate(news_preset_qs):
                if nq_cols[i].button(q, key=f"news_preset_{i}"):
                    st.session_state["news_preset_input"] = q

            # 수집된 뉴스 종목 현황 표시
            if all_news_ctx:
                coverage = " · ".join(
                    f"{c}({len(v)}건)" for c, v in all_news_ctx.items() if v
                )
                st.caption(f"📊 수집 현황: {coverage}")

            st.markdown("")

            for msg in st.session_state.news_chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            news_preset_val = st.session_state.pop("news_preset_input", None)
            news_input = st.chat_input("수집된 뉴스에 대해 질문하세요...", key="chat_news") or news_preset_val

            if news_input:
                st.session_state.news_chat_history.append({"role": "user", "content": news_input})
                with st.chat_message("user"):
                    st.markdown(news_input)
                msgs = [{"role": "system", "content": system_prompt_news}]
                msgs += [{"role": m["role"], "content": m["content"]} for m in st.session_state.news_chat_history]
                with st.chat_message("assistant"):
                    with st.spinner("뉴스 분석 중..."):
                        try:
                            answer = ask_gpt(api_key_now, msgs)
                            st.markdown(answer)
                            st.session_state.news_chat_history.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.error(friendly_openai_error(e))
                            if st.session_state.news_chat_history and st.session_state.news_chat_history[-1]["role"] == "user":
                                st.session_state.news_chat_history.pop()

            if st.session_state.get("news_chat_history"):
                if st.button("🗑️ 대화 초기화", key="reset_news"):
                    st.session_state.news_chat_history = []
                    st.rerun()

st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 데이터: Yahoo Finance | 모델: GPT-4o-mini")
