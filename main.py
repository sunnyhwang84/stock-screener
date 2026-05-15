import os
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import yfinance as yf
import anthropic
from telegram import Bot

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def get_prev_trading_day():
    d = datetime.now() - timedelta(days=1)
    for _ in range(7):
        if d.weekday() < 5:
            return d.strftime('%Y-%m-%d')
        d -= timedelta(days=1)


def is_bull():
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        df = fdr.DataReader('KS11', start, end)
        if len(df) < 20:
            return True
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        return bool(df['Close'].iloc[-1] > ma20)
    except Exception as e:
        print(f"is_bull 오류: {e}")
        return True


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


async def run():
    today = datetime.now().strftime('%Y-%m-%d')
    bull = is_bull()
    print(f"[{today}] 스크리닝 시작 | {'강세장' if bull else '약세장'}")

    try:
        df_kospi = fdr.StockListing('KOSPI')
        df_kosdaq = fdr.StockListing('KOSDAQ')
    except Exception as e:
        print(f"StockListing 오류: {e}")
        return

    df_kospi = df_kospi[df_kospi['Marcap'] >= 300_000_000_000]
    df_kosdaq = df_kosdaq[df_kosdaq['Marcap'] >= 300_000_000_000]

    kospi_codes = df_kospi['Code'].tolist()
    kosdaq_codes = df_kosdaq['Code'].tolist()
    all_codes = kospi_codes + kosdaq_codes
    yf_tickers = [f"{c}.KS" for c in kospi_codes] + [f"{c}.KQ" for c in kosdaq_codes]
    code_to_yf = dict(zip(all_codes, yf_tickers))

    name_map = {}
    for _, row in pd.concat([df_kospi, df_kosdaq]).iterrows():
        name_map[row['Code']] = row['Name']

    print(f"대상 종목: {len(all_codes)}개")

    start_dt = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')
    print("OHLCV 다운로드 중...")

    try:
        raw = yf.download(
            yf_tickers,
            start=start_dt,
            end=today,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"yfinance 오류: {e}")
        return

    if isinstance(raw.columns, pd.MultiIndex):
        close_all = raw['Close']
        volume_all = raw['Volume']
        open_all = raw['Open']
    else:
        close_all = raw[['Close']].rename(columns={'Close': yf_tickers[0]})
        volume_all = raw[['Volume']].rename(columns={'Volume': yf_tickers[0]})
        open_all = raw[['Open']].rename(columns={'Open': yf_tickers[0]})

    momentum = []
    top_d = []

    for code in all_codes:
        yf_t = code_to_yf[code]
        try:
            if yf_t not in close_all.columns:
                continue
            close = close_all[yf_t].dropna()
            volume = volume_all[yf_t].dropna()
            open_ = open_all[yf_t].dropna()
            if len(close) < 60:
                continue

            rsi = calc_rsi(close)
            ma5 = close.rolling(5).mean()
            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()

            curr = close.iloc[-1]
            curr_rsi = rsi.iloc[-1]
            curr_ma5 = ma5.iloc[-1]
            curr_ma20 = ma20.iloc[-1]
            curr_ma60 = ma60.iloc[-1]
            curr_vol = volume.iloc[-1]
            curr_open = open_.iloc[-1]

            high_52w = close.iloc[-252:].max() if len(close) >= 252 else close.max()
            vol_ma20 = volume.rolling(20).mean().iloc[-1]
            vol_ratio = curr_vol / (vol_ma20 + 1)
            is_bullish = curr > curr_open
            amount = (curr * curr_vol) / 1e8

            score = 0
            if 30 < curr_rsi < 70: score += 1
            if curr > curr_ma5: score += 1
            if curr_ma5 > curr_ma20 > curr_ma60: score += 1
            if curr >= high_52w * 0.95: score += 1
            if vol_ratio >= 2: score += 1
            if is_bullish: score += 1

            r = {
                'name': name_map.get(code, code),
                'ticker': code,
                'price': int(curr),
                'rsi': round(float(curr_rsi), 1),
                'score': score,
                'vol_ratio': round(float(vol_ratio), 1),
                'amount': round(float(amount), 0),
                'ratio': round(float((curr - curr_ma20) / curr_ma20 * 100), 1),
                'is_bullish': is_bullish,
            }

            if score >= 4:
                momentum.append(r)
            if vol_ratio >= 5 and is_bullish:
                top_d.append(r)
        except:
            continue

    momentum.sort(key=lambda x: x['score'], reverse=True)
    top_d.sort(key=lambda x: x['vol_ratio'], reverse=True)
    print(f"모멘텀 {len(momentum)}개 / 단타 {len(top_d)}개")

    summary = ""
    if ANTHROPIC_API_KEY and momentum:
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            ticker_list = ', '.join([f"{r['name']}({r['ticker']})" for r in momentum[:10]])
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{"role": "user", "content": f"오늘 스크리닝 종목: {ticker_list}. 공통 테마나 특징 2줄 요약. 반말로."}]
            )
            summary = resp.content[0].text
        except Exception as e:
            print(f"Claude 요약 오류: {e}")

    msg = f"📊 *{today} 주식 스크리닝* | {'🐂 강세장' if bull else '🐻 약세장'}\n\n"

    if momentum:
        msg += f"🚀 *모멘텀 종목 (4점+)* — {len(momentum)}개\n"
        for r in momentum[:10]:
            msg += f"• *{r['name']}* ({r['ticker']}) | {r['price']:,}원 | RSI {r['rsi']} | {r['score']}점\n"
            msg += f"  거래량 {r['vol_ratio']}배 | {r['amount']:.0f}억 | MA20 대비 {r['ratio']:+.1f}%\n"
    else:
        msg += "모멘텀 종목 없음\n"

    if summary:
        msg += f"\n💬 *AI 요약*\n{summary}\n"

    if top_d and bull:
        msg += f"\n⚡ *단타 신호 (거래대금 폭발)* — {len(top_d)}개\n"
        for r in top_d[:5]:
            msg += f"• *{r['name']}* ({r['ticker']}) | {r['price']:,}원 | {r['vol_ratio']}배 | {r['amount']:.0f}억\n"
            msg += f"  손절 -2% | 익절 +3%\n"
        msg += "⚠️ 익일 시초 -3%+ 갭하락 시 진입 보류.\n"

    msg += "\n⚠️ 후보 발굴이지 매수 추천 아님. 본인 판단 필수."

    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
    print("✅ 완료")


if __name__ == "__main__":
    asyncio.run(run())
