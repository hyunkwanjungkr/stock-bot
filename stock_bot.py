import os
import re
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import anthropic

load_dotenv(override=True)

# ─── 환경 변수 ───────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = []
for _i in range(1, 10):
    _cid = os.environ.get(f"TELEGRAM_CHAT_ID_{_i}")
    if _cid:
        TELEGRAM_CHAT_ID.append(_cid.strip())
if not TELEGRAM_CHAT_ID and os.environ.get("TELEGRAM_CHAT_ID"):
    TELEGRAM_CHAT_ID = [os.environ["TELEGRAM_CHAT_ID"].strip()]

NAVER_CLIENT_ID = os.environ["NAVER_CLIENT_ID"]
NAVER_CLIENT_SECRET = os.environ["NAVER_CLIENT_SECRET"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── 설정 ─────────────────────────────────────────────────
FINANCIAL_WEIGHT = 0.70   # 재무점수 가중치
NEWS_WEIGHT = 0.30        # 뉴스감성 가중치
TOP_FOR_NEWS = 30         # 뉴스 분석 대상 (재무 상위 N개)
TOP_FINAL = 10            # 최종 전송 종목 수
YF_WORKERS = 10           # yfinance 병렬 워커 수
KR_DETAILED_LIMIT = 200   # 한국 종목 상세 조회 한도


# ─── 재무 점수 함수 (각 항목 20점 만점, 합산 100점) ──────────

def score_per(per) -> float:
    """PER: 낮을수록 유리 (저평가 판단). 만점 25점."""
    if per is None or per <= 0:
        return 0.0
    if per <= 5:   return 25.0
    if per <= 10:  return 25.0 - (per - 5) * 1.25
    if per <= 15:  return 18.75 - (per - 10) * 1.25
    if per <= 25:  return 12.5 - (per - 15) * 0.625
    if per <= 50:  return max(0.0, 6.25 - (per - 25) * 0.25)
    return 0.0


def score_pbr(pbr) -> float:
    """PBR: 낮을수록 자산 대비 저평가."""
    if pbr is None or pbr <= 0:
        return 0.0
    if pbr <= 0.5: return 20.0
    if pbr <= 1.0: return 20.0 - (pbr - 0.5) * 10.0
    if pbr <= 2.0: return 15.0 - (pbr - 1.0) * 5.0
    if pbr <= 5.0: return 10.0 - (pbr - 2.0) * 2.0
    return max(0.0, 4.0 - (pbr - 5.0) * 0.5)


def score_roe(roe) -> float:
    """ROE: 높을수록 자본 효율성 우수. roe는 소수 (0.15 = 15%). 만점 15점."""
    if roe is None:
        return 0.0
    pct = roe * 100
    if pct < 0:    return 0.0
    if pct >= 30:  return 15.0
    if pct >= 20:  return 11.25 + (pct - 20) * 0.375
    if pct >= 10:  return 7.5 + (pct - 10) * 0.375
    if pct >= 5:   return 3.75 + (pct - 5) * 0.75
    return pct * 0.75


def score_debt(de_ratio) -> float:
    """부채비율(D/E): 낮을수록 재무 안정성 우수. 만점 15점."""
    if de_ratio is None:
        return 7.0
    if de_ratio < 0:    return 3.0
    if de_ratio <= 0.3: return 15.0
    if de_ratio <= 0.7: return 15.0 - (de_ratio - 0.3) / 0.4 * 4.0
    if de_ratio <= 1.5: return 11.0 - (de_ratio - 0.7) / 0.8 * 4.0
    if de_ratio <= 3.0: return 7.0 - (de_ratio - 1.5) / 1.5 * 4.0
    return max(0.0, 3.0 - (de_ratio - 3.0))


def score_revenue_growth(growth) -> float:
    """매출성장률: 높을수록 성장성 우수. growth는 소수 (0.15 = 15%). 만점 10점."""
    if growth is None:
        return 2.0
    pct = growth * 100
    if pct >= 30:  return 10.0
    if pct >= 15:  return 7.5 + (pct - 15) / 15 * 2.5
    if pct >= 5:   return 5.0 + (pct - 5) / 10 * 2.5
    if pct >= 0:   return 2.5 + pct / 5 * 2.5
    if pct >= -10: return max(0.0, 2.5 + pct * 0.25)
    return 0.0


def score_profit_margin(margin) -> float:
    """순이익률: 높을수록 실제 남기는 이익 우수. margin은 소수 (0.15 = 15%). 만점 15점."""
    if margin is None:
        return 0.0
    pct = margin * 100
    if pct < 0:    return 0.0
    if pct >= 20:  return 15.0
    if pct >= 10:  return 9.0 + (pct - 10) * 0.6
    if pct >= 5:   return 5.0 + (pct - 5) * 0.8
    return pct * 1.0


def calc_financial_score(metrics: dict) -> tuple[float, dict]:
    """재무 종합 점수 계산. (총점 0~100, 항목별 점수) 반환."""
    s_per    = score_per(metrics.get("per"))
    s_pbr    = score_pbr(metrics.get("pbr"))
    s_roe    = score_roe(metrics.get("roe"))
    s_margin = score_profit_margin(metrics.get("profit_margin"))
    s_debt   = score_debt(metrics.get("de_ratio"))
    s_growth = score_revenue_growth(metrics.get("revenue_growth"))
    total = s_per + s_pbr + s_roe + s_margin + s_debt + s_growth
    breakdown = {
        "per_score":    round(s_per, 1),
        "pbr_score":    round(s_pbr, 1),
        "roe_score":    round(s_roe, 1),
        "margin_score": round(s_margin, 1),
        "debt_score":   round(s_debt, 1),
        "growth_score": round(s_growth, 1),
        "total":        round(total, 1),
    }
    return total, breakdown


# ─── 분석 대상 종목 (고정) ────────────────────────────────

WATCH_LIST: list[dict] = [
    {"ticker": "340570.KQ", "name": "티앤엘",           "market": "KOSDAQ", "per": None, "pbr": None},
    {"ticker": "005380.KS", "name": "현대차",           "market": "KOSPI",  "per": None, "pbr": None},
    {"ticker": "GOOGL",     "name": "구글 (Alphabet)",  "market": "NASDAQ", "per": None, "pbr": None},
    {"ticker": "OPEN",      "name": "오픈도어",         "market": "NASDAQ", "per": None, "pbr": None},
    {"ticker": "CPNG",      "name": "쿠팡",             "market": "NYSE",   "per": None, "pbr": None},
    {"ticker": "INMD",      "name": "인모드",           "market": "NASDAQ", "per": None, "pbr": None},
    {"ticker": "033100.KQ", "name": "제룡전기",         "market": "KOSDAQ", "per": None, "pbr": None},
    {"ticker": "020000.KS", "name": "한섬",             "market": "KOSPI",  "per": None, "pbr": None},
    {"ticker": "013520.KS", "name": "화승인더",         "market": "KOSPI",  "per": None, "pbr": None},
    {"ticker": "009540.KS", "name": "HD한국조선해양",   "market": "KOSPI",  "per": None, "pbr": None},
    {"ticker": "034220.KS", "name": "LG디스플레이",     "market": "KOSPI",  "per": None, "pbr": None},
    {"ticker": "042660.KS", "name": "한화오션",         "market": "KOSPI",  "per": None, "pbr": None},
    {"ticker": "100120.KQ", "name": "뷰웍스",           "market": "KOSDAQ", "per": None, "pbr": None},
    {"ticker": "329180.KS", "name": "HD현대중공업",     "market": "KOSPI",  "per": None, "pbr": None},
]


# ─── yfinance 상세 재무 수집 ──────────────────────────────

def fetch_naver_per_pbr(ticker: str) -> tuple:
    """네이버 증권에서 한국 종목 PER/PBR 스크래핑. ticker는 yfinance 형식 (예: 005380.KS)."""
    try:
        code = ticker.split(".")[0]
        resp = requests.get(
            f"https://finance.naver.com/item/main.naver?code={code}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        import re as _re
        per_match = _re.search(r"PER\(배\).*?<td[^>]*>\s*([\d.]+)\s*</td>", resp.text, _re.DOTALL)
        pbr_match = _re.search(r"PBR\(배\).*?<td[^>]*>\s*([\d.]+)\s*</td>", resp.text, _re.DOTALL)
        per = float(per_match.group(1)) if per_match else None
        pbr = float(pbr_match.group(1)) if pbr_match else None
        return per, pbr
    except Exception:
        return None, None


def fetch_yf_financials(stock_info: dict) -> dict:
    """yfinance로 상세 재무 지표 수집. 한국 종목 PER/PBR은 네이버 증권으로 보완."""
    ticker = stock_info["ticker"]
    is_kr = stock_info.get("market") in ("KOSPI", "KOSDAQ")
    try:
        info = yf.Ticker(ticker).info
        result = stock_info.copy()
        result["per"]            = result.get("per") or info.get("trailingPE") or info.get("forwardPE")
        result["pbr"]            = result.get("pbr") or info.get("priceToBook")
        result["roe"]            = info.get("returnOnEquity")
        de = info.get("debtToEquity")
        result["de_ratio"]       = de / 100 if de is not None else None  # yfinance는 % 단위
        result["revenue_growth"] = info.get("revenueGrowth")
        result["profit_margin"]  = info.get("profitMargins")
        result["current_price"]  = info.get("currentPrice") or info.get("regularMarketPrice")
        result["currency"]       = info.get("currency", "KRW")

        # 한국 종목 PER/PBR이 없으면 네이버 증권으로 보완
        if is_kr and (not result["per"] or not result["pbr"]):
            nv_per, nv_pbr = fetch_naver_per_pbr(ticker)
            if not result["per"]:
                result["per"] = nv_per
            if not result["pbr"]:
                result["pbr"] = nv_pbr

        return result
    except Exception:
        # yfinance 실패 시 한국 종목은 네이버 증권으로 PER/PBR 수집
        result = stock_info.copy()
        if is_kr:
            nv_per, nv_pbr = fetch_naver_per_pbr(ticker)
            result["per"] = nv_per
            result["pbr"] = nv_pbr
        return result


# ─── 뉴스 수집 ────────────────────────────────────────────

def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).replace("&quot;", '"').replace("&amp;", "&").replace("&#039;", "'")


def get_naver_news(company_name: str, count: int = 5) -> list[str]:
    """네이버 뉴스 API로 종목 뉴스 수집. 제목+요약 텍스트 리스트 반환."""
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": company_name, "display": count, "sort": "date"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [
            strip_html(item["title"]) + " " + strip_html(item.get("description", ""))
            for item in items
        ]
    except Exception:
        return []


def get_yf_news(ticker: str, count: int = 5) -> list[str]:
    """yfinance로 US 종목 뉴스 수집. 제목+요약 텍스트 리스트 반환."""
    try:
        news = yf.Ticker(ticker).news or []
        texts = []
        for n in news[:count]:
            # 신버전: {"id": ..., "content": {"title": ..., "summary": ...}}
            if "content" in n:
                c = n["content"]
                title = c.get("title", "")
                summary = c.get("summary", c.get("description", ""))
            # 구버전: {"title": ..., "summary": ...}
            else:
                title = n.get("title", "")
                summary = n.get("summary", n.get("description", ""))
            if title:
                texts.append(f"{title} {summary}".strip())
        return texts
    except Exception:
        return []


# ─── Claude 뉴스 감성 분석 ────────────────────────────────

def analyze_news_batch(stocks: list[dict]) -> dict[str, dict]:
    """Claude로 5개씩 묶어 뉴스 감성 분석.
    반환: {ticker: {"score": 0~100, "positive": str, "negative": str}}
    """
    # 뉴스 텍스트 수집
    for s in stocks:
        if s["market"] in ("KOSPI", "KOSDAQ"):
            s["news_texts"] = get_naver_news(s["name"], count=5)
        else:
            s["news_texts"] = get_yf_news(s["ticker"], count=5)

    results: dict[str, dict] = {}
    batch_size = 5

    for i in range(0, len(stocks), batch_size):
        batch = stocks[i : i + batch_size]
        articles_block = ""
        for j, s in enumerate(batch):
            news_lines = "\n".join(f"  - {t}" for t in s["news_texts"]) if s["news_texts"] else "  - (뉴스 없음)"
            articles_block += f"\n[{j+1}] {s['name']} ({s['ticker']})\n{news_lines}\n"

        prompt = (
            f"아래 {len(batch)}개 종목의 최근 뉴스를 분석해 투자 감성 점수를 매겨줘.\n"
            "점수 기준: 0~100점 (100=매우 긍정, 50=중립, 0=매우 부정)\n"
            "반드시 아래 형식으로만 출력해. 다른 설명 없이.\n"
            "형식: 번호|점수|긍정요인(없으면 없음)|부정요인(없으면 없음)\n"
            "예시: 1|75|AI 수주 증가, 실적 상향|글로벌 경쟁 심화\n\n"
            f"{articles_block}"
        )
        try:
            resp = claude.messages.create(
                model="claude-haiku-4-5",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            for line in resp.content[0].text.strip().splitlines():
                parts = line.strip().split("|")
                if len(parts) < 4:
                    continue
                try:
                    idx = int(parts[0]) - 1
                    score = float(parts[1])
                    if 0 <= idx < len(batch):
                        results[batch[idx]["ticker"]] = {
                            "score":    max(0.0, min(100.0, score)),
                            "positive": parts[2].strip(),
                            "negative": parts[3].strip(),
                        }
                except (ValueError, IndexError):
                    pass
        except Exception as e:
            print(f"  Claude 분석 실패 (배치 {i//batch_size + 1}): {e}")
        time.sleep(0.5)

    return results


# ─── 투자 의견 분석 (Claude Sonnet) ──────────────────────

def analyze_investment_opinion(final_results: list[dict]) -> str:
    """Claude Sonnet으로 각 종목 투자 의견 생성. 전체 결과를 하나의 분석 메시지로 반환."""
    stocks_block = ""
    for i, r in enumerate(final_results, 1):
        s      = r["stock"]
        fin    = r["fin_breakdown"]
        news   = r["news_result"]
        per    = s.get("per")
        pbr    = s.get("pbr")
        roe    = s.get("roe")
        margin = s.get("profit_margin")
        de     = s.get("de_ratio")
        growth = s.get("revenue_growth")
        stocks_block += (
            f"\n[{i}] {s['name']} ({s['ticker']} / {s['market']})\n"
            f"  종합점수: {r['final_score']:.1f} | 재무: {fin['total']:.1f} | 뉴스감성: {news.get('score', 50):.0f}\n"
            f"  PER: {f'{per:.1f}' if per else 'N/A'} | PBR: {f'{pbr:.2f}' if pbr else 'N/A'} | "
            f"ROE: {f'{roe*100:.1f}%' if roe else 'N/A'} | "
            f"순이익률: {f'{margin*100:.1f}%' if margin else 'N/A'} | "
            f"부채비율: {f'{de*100:.0f}%' if de else 'N/A'} | "
            f"매출성장: {f'{growth*100:+.1f}%' if growth else 'N/A'}\n"
            f"  뉴스 긍정: {news.get('positive', '없음')} | 뉴스 부정: {news.get('negative', '없음')}\n"
        )

    prompt = (
        "당신은 10년 경력의 주식 애널리스트입니다.\n"
        "아래 종목들의 재무 데이터와 뉴스 감성을 바탕으로 각 종목에 대한 투자 의견을 작성해주세요.\n\n"
        "각 종목마다 다음 형식으로 출력하세요:\n"
        "번호|의견(매수/중립/매도)|한줄요약(30자 이내)|근거(60자 이내)\n"
        "예시: 1|매수|저PER·고ROE 우량주, 저평가 매력|PER 5배 수준으로 동종업계 대비 현저히 저평가\n\n"
        "다른 설명 없이 형식에 맞게만 출력하세요.\n\n"
        f"{stocks_block}"
    )

    try:
        resp = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        opinion_map = {}
        for line in resp.content[0].text.strip().splitlines():
            parts = line.strip().split("|")
            if len(parts) < 4:
                continue
            try:
                idx = int(parts[0]) - 1
                if 0 <= idx < len(final_results):
                    opinion_map[idx] = {
                        "verdict": parts[1].strip(),
                        "summary": parts[2].strip(),
                        "reason":  parts[3].strip(),
                    }
            except (ValueError, IndexError):
                pass
    except Exception as e:
        print(f"  투자 의견 생성 실패: {e}")
        return ""

    verdict_icon = {"매수": "🟢", "중립": "🟡", "매도": "🔴"}
    lines = ["📋 *애널리스트 투자 의견*\n"]
    for i, r in enumerate(final_results):
        op = opinion_map.get(i)
        if not op:
            continue
        icon = verdict_icon.get(op["verdict"], "⚪")
        lines.append(
            f"{icon} *{i+1}. {r['stock']['name']}* — {op['verdict']}\n"
            f"  {op['summary']}\n"
            f"  _{op['reason']}_"
        )
    return "\n\n".join(lines)


# ─── 텔레그램 메시지 포맷 ─────────────────────────────────

def fmt_val(val, fmt, suffix="", fallback="N/A") -> str:
    return f"{val:{fmt}}{suffix}" if val is not None else fallback


def _w(s: str) -> int:
    """문자열 표시 너비 (한글 등 2바이트 문자는 2칸)."""
    return sum(2 if ord(c) > 127 else 1 for c in s)


def _pad(s: str, width: int) -> str:
    """표시 너비 기준으로 오른쪽 공백 패딩."""
    return s + " " * max(0, width - _w(s))


def format_full_message(final_results: list[dict], now_kst: datetime) -> str:
    """전체 종목을 표 형태 하나의 메시지로 포맷."""
    header = (
        f"📊 주식 상승여력 분석  {now_kst.strftime('%Y-%m-%d %H:%M')} KST\n"
        f"재무 {int(FINANCIAL_WEIGHT*100)}%  +  뉴스감성 {int(NEWS_WEIGHT*100)}%"
    )

    # 표 헤더
    col_h = ["#", "종목", "현재가", "종합", "재무", "뉴스", "PER", "PBR", "ROE", "순이익률", "부채비율", "매출성장"]
    rows = []
    for i, r in enumerate(final_results, 1):
        s          = r["stock"]
        fin        = r["fin_breakdown"]
        news_score = r["news_result"].get("score", 50.0)
        price      = s.get("current_price")
        per        = s.get("per")
        pbr        = s.get("pbr")
        roe        = s.get("roe")
        margin     = s.get("profit_margin")
        de         = s.get("de_ratio")
        growth     = s.get("revenue_growth")

        price_str = (f"{price:,.0f}원" if s.get("currency") == "KRW"
                     else f"${price:,.2f}") if price else "N/A"

        rows.append([
            str(i),
            s["name"],
            price_str,
            f"{r['final_score']:.1f}",
            f"{fin['total']:.1f}",
            f"{news_score:.0f}",
            f"{per:.1f}" if per else "N/A",
            f"{pbr:.2f}" if pbr else "N/A",
            f"{roe*100:.1f}%" if roe else "N/A",
            f"{margin*100:.1f}%" if margin else "N/A",
            f"{de*100:.0f}%" if de else "N/A",
            f"{growth*100:+.1f}%" if growth else "N/A",
        ])

    # 열 너비 계산 (헤더·데이터 중 최대)
    widths = [max(_w(col_h[j]), max(_w(row[j]) for row in rows)) for j in range(len(col_h))]

    sep  = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    def fmt_row(cells):
        return "|" + "|".join(f" {_pad(c, widths[j])} " for j, c in enumerate(cells)) + "|"

    table_lines = [sep, fmt_row(col_h), sep]
    for row in rows:
        table_lines.append(fmt_row(row))
    table_lines.append(sep)

    # 뉴스 감성 요약
    news_lines = []
    for i, r in enumerate(final_results, 1):
        positive = r["news_result"].get("positive", "")
        negative = r["news_result"].get("negative", "")
        s_name   = r["stock"]["name"]
        if positive and positive != "없음":
            news_lines.append(f"{i}.{s_name} ✅ {positive}")
        if negative and negative != "없음":
            news_lines.append(f"{i}.{s_name} ⚠️ {negative}")

    table_str = "\n".join(table_lines)
    msg = f"{header}\n\n```\n{table_str}\n```"
    if news_lines:
        msg += "\n\n📰 뉴스 감성\n" + "\n".join(news_lines)
    return msg


# ─── 텔레그램 전송 ────────────────────────────────────────

def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chat_id in TELEGRAM_CHAT_ID:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            print(f"  전송 완료 → {chat_id}")
        except Exception as e:
            print(f"  전송 실패 → {chat_id}: {e}")


# ─── 메인 ─────────────────────────────────────────────────

def main() -> None:
    now_kst = datetime.now(timezone.utc) + timedelta(hours=9)
    print(f"[{now_kst.strftime('%Y-%m-%d %H:%M')} KST] 주식 상승여력 분석 시작")

    # 1. 고정 종목 목록
    print("\n=== 1단계: 종목 목록 ===")
    candidates = [s.copy() for s in WATCH_LIST]
    print(f"분석 대상: {len(candidates)}개")
    for s in candidates:
        print(f"  {s['name']} ({s['ticker']}  |  {s['market']})")

    # 2. yfinance 상세 재무 수집
    print(f"\n=== 2단계: 재무 상세 수집 ({len(candidates)}개) ===")
    detailed: list[dict] = []
    with ThreadPoolExecutor(max_workers=YF_WORKERS) as executor:
        futures = {executor.submit(fetch_yf_financials, s): s for s in candidates}
        for future in as_completed(futures):
            detailed.append(future.result())

    # 3. 재무 점수 계산
    print("\n=== 3단계: 재무 점수 계산 ===")
    scored: list[dict] = []
    for stock in detailed:
        metrics = {
            "per":            stock.get("per"),
            "pbr":            stock.get("pbr"),
            "roe":            stock.get("roe"),
            "profit_margin":  stock.get("profit_margin"),
            "de_ratio":       stock.get("de_ratio"),
            "revenue_growth": stock.get("revenue_growth"),
        }
        fin_score, fin_breakdown = calc_financial_score(metrics)
        stock["fin_score"]     = fin_score
        stock["fin_breakdown"] = fin_breakdown
        scored.append(stock)

    # 4. 뉴스 감성 분석
    print("\n=== 4단계: 뉴스 감성 분석 (Claude) ===")
    news_results = analyze_news_batch(scored)

    # 5. 종합 점수 계산 및 정렬
    print("\n=== 5단계: 최종 순위 계산 ===")
    final_results: list[dict] = []
    for stock in scored:
        ticker = stock["ticker"]
        fin_score  = stock["fin_score"]
        news_info  = news_results.get(ticker, {"score": 50.0, "positive": "", "negative": ""})
        news_score = news_info["score"]
        final_score = fin_score * FINANCIAL_WEIGHT + news_score * NEWS_WEIGHT
        final_results.append({
            "stock":         stock,
            "fin_breakdown": stock["fin_breakdown"],
            "news_result":   news_info,
            "final_score":   round(final_score, 1),
        })

    final_results.sort(key=lambda x: x["final_score"], reverse=True)

    for r in final_results:
        s = r["stock"]
        print(f"  {s['name']} ({s['ticker']}) | 재무 {r['fin_breakdown']['total']:.1f} | "
              f"뉴스 {r['news_result'].get('score', 50):.0f} | 종합 {r['final_score']:.1f}")

    # 6. 투자 의견 생성 (Claude Sonnet)
    print("\n=== 6단계: 투자 의견 생성 (Claude Sonnet) ===")
    opinion_msg = analyze_investment_opinion(final_results)

    # 7. 텔레그램 전송
    print("\n=== 7단계: 텔레그램 전송 ===")
    send_telegram(format_full_message(final_results, now_kst))
    if opinion_msg:
        time.sleep(0.5)
        send_telegram(opinion_msg)

    print("\n완료")


if __name__ == "__main__":
    main()
