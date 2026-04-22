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
    """PER: 낮을수록 유리 (저평가 판단)."""
    if per is None or per <= 0:
        return 0.0
    if per <= 5:   return 20.0
    if per <= 10:  return 20.0 - (per - 5) * 1.0
    if per <= 15:  return 15.0 - (per - 10) * 1.0
    if per <= 25:  return 10.0 - (per - 15) * 0.5
    if per <= 50:  return max(0.0, 5.0 - (per - 25) * 0.2)
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
    """ROE: 높을수록 자본 효율성 우수. roe는 소수 (0.15 = 15%)."""
    if roe is None:
        return 0.0
    pct = roe * 100
    if pct < 0:    return 0.0
    if pct >= 30:  return 20.0
    if pct >= 20:  return 15.0 + (pct - 20) * 0.5
    if pct >= 10:  return 10.0 + (pct - 10) * 0.5
    if pct >= 5:   return 5.0 + (pct - 5) * 1.0
    return pct * 1.0


def score_debt(de_ratio) -> float:
    """부채비율(D/E): 낮을수록 재무 안정성 우수."""
    if de_ratio is None:
        return 10.0  # 데이터 없으면 중간값
    if de_ratio < 0:   return 5.0
    if de_ratio <= 0.3: return 20.0
    if de_ratio <= 0.7: return 20.0 - (de_ratio - 0.3) / 0.4 * 5.0
    if de_ratio <= 1.5: return 15.0 - (de_ratio - 0.7) / 0.8 * 5.0
    if de_ratio <= 3.0: return 10.0 - (de_ratio - 1.5) / 1.5 * 5.0
    return max(0.0, 5.0 - (de_ratio - 3.0))


def score_revenue_growth(growth) -> float:
    """매출성장률: 높을수록 성장성 우수. growth는 소수 (0.15 = 15%)."""
    if growth is None:
        return 5.0  # 데이터 없으면 낮게
    pct = growth * 100
    if pct >= 30:  return 20.0
    if pct >= 15:  return 15.0 + (pct - 15) / 15 * 5.0
    if pct >= 5:   return 10.0 + (pct - 5) / 10 * 5.0
    if pct >= 0:   return 5.0 + pct / 5 * 5.0
    if pct >= -10: return max(0.0, 5.0 + pct * 0.5)
    return 0.0


def calc_financial_score(metrics: dict) -> tuple[float, dict]:
    """재무 종합 점수 계산. (총점 0~100, 항목별 점수) 반환."""
    s_per    = score_per(metrics.get("per"))
    s_pbr    = score_pbr(metrics.get("pbr"))
    s_roe    = score_roe(metrics.get("roe"))
    s_debt   = score_debt(metrics.get("de_ratio"))
    s_growth = score_revenue_growth(metrics.get("revenue_growth"))
    total = s_per + s_pbr + s_roe + s_debt + s_growth
    breakdown = {
        "per_score":    round(s_per, 1),
        "pbr_score":    round(s_pbr, 1),
        "roe_score":    round(s_roe, 1),
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
    {"ticker": "INMD",      "name": "인모드",           "market": "NASDAQ", "per": None, "pbr": None},
    {"ticker": "033100.KQ", "name": "제룡전기",         "market": "KOSDAQ", "per": None, "pbr": None},
    {"ticker": "020000.KS", "name": "한섬",             "market": "KOSPI",  "per": None, "pbr": None},
]


# ─── yfinance 상세 재무 수집 ──────────────────────────────

def fetch_kr_per_pbr(ticker: str) -> tuple:
    """pykrx로 한국 종목 PER/PBR 조회. ticker는 yfinance 형식 (예: 005380.KS)."""
    try:
        from pykrx import stock as pykrx_stock
        code = ticker.split(".")[0]
        for days_back in range(0, 7):
            date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
            try:
                df = pykrx_stock.get_market_fundamental(date, date, code)
                if df.empty or code not in df.index:
                    continue
                per_val = df.loc[code, "PER"]
                pbr_val = df.loc[code, "PBR"]
                per = float(per_val) if per_val and per_val > 0 else None
                pbr = float(pbr_val) if pbr_val and pbr_val > 0 else None
                return per, pbr
            except Exception:
                continue
    except Exception:
        pass
    return None, None


def fetch_yf_financials(stock_info: dict) -> dict:
    """yfinance로 상세 재무 지표 수집. 한국 종목은 pykrx로 PER/PBR 보완."""
    ticker = stock_info["ticker"]
    is_kr = stock_info.get("market") in ("KOSPI", "KOSDAQ")
    try:
        info = yf.Ticker(ticker).info
        result = stock_info.copy()
        result["per"]            = result.get("per") or info.get("trailingPE")
        result["pbr"]            = result.get("pbr") or info.get("priceToBook")
        result["roe"]            = info.get("returnOnEquity")
        de = info.get("debtToEquity")
        result["de_ratio"]       = de / 100 if de is not None else None  # yfinance는 % 단위
        result["revenue_growth"] = info.get("revenueGrowth")
        result["current_price"]  = info.get("currentPrice") or info.get("regularMarketPrice")
        result["currency"]       = info.get("currency", "KRW")

        # 한국 종목 PER/PBR이 없으면 pykrx로 보완
        if is_kr and (not result["per"] or not result["pbr"]):
            kr_per, kr_pbr = fetch_kr_per_pbr(ticker)
            if not result["per"]:
                result["per"] = kr_per
            if not result["pbr"]:
                result["pbr"] = kr_pbr

        return result
    except Exception:
        # yfinance 실패 시 한국 종목은 pykrx로 PER/PBR만이라도 수집
        result = stock_info.copy()
        if is_kr:
            kr_per, kr_pbr = fetch_kr_per_pbr(ticker)
            result["per"] = kr_per
            result["pbr"] = kr_pbr
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
        return [
            n.get("title", "") + " " + n.get("summary", n.get("description", ""))
            for n in news[:count]
        ]
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


# ─── 텔레그램 메시지 포맷 ─────────────────────────────────

def fmt_val(val, fmt, suffix="", fallback="N/A") -> str:
    return f"{val:{fmt}}{suffix}" if val is not None else fallback


def format_stock_message(rank: int, stock: dict, fin_breakdown: dict, news_result: dict, final_score: float) -> str:
    name    = stock["name"]
    ticker  = stock["ticker"]
    market  = stock["market"]
    price   = stock.get("current_price")
    currency = "원" if stock.get("currency") == "KRW" else "$"
    price_str = f"{price:,.0f}{currency}" if price else "N/A"

    per    = stock.get("per")
    pbr    = stock.get("pbr")
    roe    = stock.get("roe")
    de     = stock.get("de_ratio")
    growth = stock.get("revenue_growth")

    fin_total  = fin_breakdown["total"]
    news_score = news_result.get("score", 50.0)
    fin_contrib  = round(fin_total * FINANCIAL_WEIGHT, 1)
    news_contrib = round(news_score * NEWS_WEIGHT, 1)

    lines = [
        f"{'🥇' if rank==1 else '🥈' if rank==2 else '🥉' if rank==3 else '📌'} "
        f"*{rank}위. {name}* ({ticker}  |  {market})",
        f"현재가: {price_str}  |  *종합점수: {final_score:.1f}점*",
        f"  ├ 재무({int(FINANCIAL_WEIGHT*100)}%): {fin_total:.1f}/100 → {fin_contrib}점",
        f"  └ 뉴스감성({int(NEWS_WEIGHT*100)}%): {news_score:.0f}/100 → {news_contrib}점",
        "",
        f"📈 *재무 상세* ({fin_total:.1f}/100점)",
        f"  • PER  {fmt_val(per, '.1f'):<6}  →  {fin_breakdown['per_score']:.1f}/20점",
        f"  • PBR  {fmt_val(pbr, '.2f'):<6}  →  {fin_breakdown['pbr_score']:.1f}/20점",
        f"  • ROE  {fmt_val(roe*100 if roe else None, '.1f', '%'):<6}  →  {fin_breakdown['roe_score']:.1f}/20점",
        f"  • 부채비율  {fmt_val(de*100 if de else None, '.0f', '%'):<6}  →  {fin_breakdown['debt_score']:.1f}/20점",
        f"  • 매출성장  {fmt_val(growth*100 if growth else None, '+.1f', '%'):<6}  →  {fin_breakdown['growth_score']:.1f}/20점",
        "",
        f"📰 *뉴스 감성* ({news_score:.0f}/100점)",
    ]
    positive = news_result.get("positive", "")
    negative = news_result.get("negative", "")
    if positive and positive != "없음":
        lines.append(f"  ✅ {positive}")
    if negative and negative != "없음":
        lines.append(f"  ⚠️ {negative}")

    return "\n".join(lines)


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

    # 6. 텔레그램 전송
    print("\n=== 6단계: 텔레그램 전송 ===")
    header = (
        f"📊 *[주식 상승여력 분석]* {now_kst.strftime('%Y-%m-%d %H:%M')} KST\n"
        f"분석 종목: {len(final_results)}개\n"
        f"가중치: 재무 {int(FINANCIAL_WEIGHT*100)}%  +  뉴스감성 {int(NEWS_WEIGHT*100)}%\n"
        + "═" * 24
    )
    send_telegram(header)

    for i, r in enumerate(final_results, 1):
        msg = format_stock_message(i, r["stock"], r["fin_breakdown"], r["news_result"], r["final_score"])
        send_telegram(msg)
        time.sleep(0.3)

    print("\n완료")


if __name__ == "__main__":
    main()
