import os
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from pykrx import stock
from telegram import Bot
import anthropic

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

def get_recent_trading_day():
    d = datetime.now()- timedelta(days=1)
    for _ in range(7):
        if d.weekday() < 5:
            return d.strftime("%Y%m%d")
        d -= timedelta(days=1)

def get_ohlcv(ticker, days=130):
    end   = get_recent_trading_day()
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    try:
        df = stock.get_market_ohlcv_by_date(start, end, ticker)
        if len(df) < 65:
            return None
        df.columns = ['open','high','low','close','volume','amount','changes']
        return df
    except:
        return None

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def score_ticker(df):
    df = df.copy()
    df['RSI']      = rsi(df['close'])
    df['MA5']      = df['close'].rolling(5).mean()
    df['MA20']     = df['close'].rolling(20).mean()
    df['MA60']     = df['close'].rolling(60).mean()
    df['vol_ma5']  = df['volume'].rolling(5).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['high_52w'] = df['high'].rolling(252).max()

    cur  = df.iloc[-1]
    prev = df.iloc[-2]

    score   = 0
    signals = []

    if 30 <= cur['RSI'] <= 40:
        score += 1; signals.append("RSI반등")
    if prev['close'] < prev['MA5'] and cur['close'] > cur['MA5']:
        score += 1; signals.append("5일선돌파")
    if cur['MA20'] > cur['MA60']:
        score += 1; signals.append("정배열")
    if cur['high_52w'] > 0 and cur['close'] / cur['high_52w'] >= 0.95:
        score += 1; signals.append(f"52주고점{cur['close']/cur['high_52w']*100:.0f}%")
    if cur['vol_ma5'] > cur['vol_ma20'] * 2:
        score += 1; signals.append("거래량폭발")
    if cur['close'] > cur['open']:
        score += 1; signals.append("양봉")

    return score, signals, df

def is_danta(df, market_cap):
    cur      = df.iloc[-1]
    prev_avg = df['amount'].rolling(20).mean().iloc[-2]
    if prev_avg == 0:
        return False, 0
    ratio = cur['amount'] / prev_avg
    ok = ratio >= 5 and cur['close'] > cur['open'] and market_cap >= 100_000_000_000
    return ok, ratio

def is_bull():
    end   = get_recent_trading_day()
    start = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
    try:
        k = stock.get_index_ohlcv_by_date(start, end, "1001")
        r = k.iloc[-1]
        return r['종가'] > r['시가']
    except:
        return True

def claude_summary(name):
    try:
        c = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        r = c.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=120,
            messages=[{"role":"user","content":
                f"한국 주식 {name}의 최근 투자 포인트를 1문장으로만 요약해줘. 모르면 '-'라고만 해."}]
        )
        return r.content[0].text.strip()
    except:
        return "-"

async def run():
    today = datetime.now().strftime("%Y-%m-%d")
    date  = get_recent_trading_day()
    bull  = is_bull()
    print(f"[{today}] 스크리닝 시작 | {'강세장' if bull else '약세장'}")

    cap_kospi  = stock.get_market_cap_by_ticker(date, market="KOSPI")
    cap_kosdaq = stock.get_market_cap_by_ticker(date, market="KOSDAQ")
    cap_all    = pd.concat([cap_kospi, cap_kosdaq])
    cap_all    = cap_all[cap_all['시가총액'] >= 300_000_000_000]
    tickers    = cap_all.index.tolist()
    print(f"대상 종목: {len(tickers)}개")

    momentum, danta = [], []

    for i, ticker in enumerate(tickers):
        if i % 200 == 0: print(f"{i}/{len(tickers)}")
        df = get_ohlcv(ticker)
        if df is None: continue
        try:
            sc, sigs, df2 = score_ticker(df)
            cap  = cap_all.loc[ticker, '시가총액']
            name = stock.get_market_ticker_name(ticker)
            cur  = df2.iloc[-1]

            if sc >= 3:
                vol_r   = cur['vol_ma5'] / cur['vol_ma20'] if cur['vol_ma20'] > 0 else 0
                pct_52w = cur['close'] / cur['high_52w'] * 100 if cur['high_52w'] > 0 else 0
                momentum.append(dict(ticker=ticker, name=name, score=sc,
                    price=cur['close'], rsi=cur['RSI'],
                    vol_r=vol_r, pct_52w=pct_52w, signals=sigs))

            if bull:
                ok, ratio = is_danta(df2, cap)
                if ok:
                    danta.append(dict(ticker=ticker, name=name,
                        price=cur['close'], ratio=ratio,
                        amount=cur['amount']/100_000_000))
        except:
            continue

    momentum.sort(key=lambda x: x['score'], reverse=True)
    danta.sort(key=lambda x: x['ratio'], reverse=True)
    top5  = momentum[:5]
    top_d = danta[:3]

    msg = f"🔍 *오늘의 스크리닝* ({today})\n"
    msg += f"시장: {'📈 강세장' if bull else '📉 약세장'}\n\n"

    for i, r in enumerate(top5, 1):
        summary = claude_summary(r['name'])
        msg += f"{i}. *{r['name']} ({r['ticker']})* | {r['score']*10}점\n"
        msg += f"   {r['price']:,.0f}원 | RSI {r['rsi']:.1f} | 거래량 {r['vol_r']:.1f}x | 52w {r['pct_52w']:.1f}%\n"
        msg += f"   시그널: {', '.join(r['signals'])}\n"
        msg += f"   💬 {summary}\n\n"

    if top_d and bull:
        msg += "━━━━━━━━━━━━━━━━━\n"
        msg += "⚡ *단타 신호 (거래대금 폭발)*\n\n"
        for r in top_d:
            msg += f"• *{r['name']} ({r['ticker']})* | {r['ratio']:.1f}배 | {r['amount']:.0f}억\n"
            msg += f"  {r['price']:,.0f}원 | 손절 -2% | 익절 +3%\n\n"
        msg += "⚠️ 익일 시초 -3%+ 갭하락 시 진입 보류.\n"

    msg += "\n⚠️ 후보 발굴이지 매수 추천 아님. 본인 판단 필수."

    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
    print("✅ 완료")

if __name__ == "__main__":
    asyncio.run(run())
