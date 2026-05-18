from jugaad_data.nse import NSELive
from datetime import datetime
import math
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/option-chain",
    "X-Requested-With": "XMLHttpRequest",
    "Connection": "keep-alive",
}

def get_nse_session():
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=HEADERS, timeout=15)
    session.get("https://www.nseindia.com/option-chain", headers=HEADERS, timeout=15)
    return session

def fetch_nse_api(url):
    session = get_nse_session()
    response = session.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.json()

n = NSELive()

def format_expiry_for_comparison(expiry_str):
    dt = datetime.strptime(expiry_str, "%d-%b-%Y")
    return dt.strftime("%d-%m-%Y")

def get_india_vix():
    try:
        data = n.get('live_index', {'index': 'INDIA VIX'})
        vix = float(data['metadata']['last'])
        print("VIX:", vix)
        return vix * 0.01
    except Exception as e:
        print("VIX error:", e)
        try:
            data = fetch_nse_api("https://www.nseindia.com/api/equity-stockIndices?index=INDIA%20VIX")
            vix = float(data['data'][0]['lastPrice'])
            return vix * 0.01
        except:
            return 0.15

def get_tte_days(expiry_str):
    expiry_dt = datetime.strptime(expiry_str, "%d-%b-%Y")
    now = datetime.now()
    diff = expiry_dt - now
    total_days = diff.total_seconds() / 86400

    if total_days <= 0:
        market_close = expiry_dt.replace(hour=15, minute=30, second=0, microsecond=0)
        remaining = (market_close - now).total_seconds() / 86400
        return max(remaining, 1/1440)

    return total_days

def get_all_expiries(symbol="NIFTY"):
    try:
        data = fetch_nse_api(f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}")
        if "records" in data:
            return data["records"]["expiryDates"]
        raise Exception("No records")
    except:
        try:
            data = n.index_option_chain(symbol)
            return data["records"]["expiryDates"]
        except:
            data = n.get('option_chain_v3', {'symbol': symbol})
            return data["records"]["expiryDates"]

def get_spot_and_premiums(symbol="NIFTY", expiry=None, otm_range=20, itm_range=2):
    # try fetch_nse_api first, fallback to jugaad
    chain = None
    try:
        chain = fetch_nse_api(f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}")
        if "records" not in chain:
            raise Exception("No records key")
    except:
        chain = n.index_option_chain(symbol)

    spot_price = chain["records"]["underlyingValue"]
    all_expiries = chain["records"]["expiryDates"]

    if expiry is None:
        expiry = all_expiries[0]

    expiry_formatted = format_expiry_for_comparison(expiry)
    atm_strike = round(spot_price / 50) * 50

    ce_lower = atm_strike - itm_range * 50
    ce_upper = atm_strike + otm_range * 50
    pe_lower = atm_strike - otm_range * 50
    pe_upper = atm_strike + itm_range * 50

    # try fetch_nse_api first, fallback to jugaad
    chain_v3 = None
    try:
        chain_v3 = fetch_nse_api(f"https://www.nseindia.com/api/option-chain-v3?symbol={symbol}&expiry={expiry}")
        if "filtered" not in chain_v3:
            raise Exception("No filtered key")
    except:
        chain_v3 = n.get('option_chain_v3', {'symbol': symbol, 'expiry': expiry})

    all_data = chain_v3.get('filtered', {}).get('data', [])

    filtered = []
    for record in all_data:
        ce_expiry = record.get("CE", {}).get("expiryDate", "")
        pe_expiry = record.get("PE", {}).get("expiryDate", "")

        if ce_expiry != expiry_formatted and pe_expiry != expiry_formatted:
            continue

        strike = record["strikePrice"]

        ce_premium = record.get("CE", {}).get("lastPrice", 0)
        pe_premium = record.get("PE", {}).get("lastPrice", 0)
        ce_iv = record.get("CE", {}).get("impliedVolatility", 0)
        pe_iv = record.get("PE", {}).get("impliedVolatility", 0)

        if ce_lower <= strike <= ce_upper:
            pass
        elif pe_lower <= strike <= pe_upper:
            pass
        else:
            continue

        if not (ce_lower <= strike <= ce_upper):
            ce_premium = 0
            ce_iv = 0
        if not (pe_lower <= strike <= pe_upper):
            pe_premium = 0
            pe_iv = 0

        diff = (strike - atm_strike) / 50

        filtered.append({
            "strike": strike,
            "diff_from_atm": int(diff),
            "ce_premium": ce_premium,
            "pe_premium": pe_premium,
            "ce_iv": ce_iv,
            "pe_iv": pe_iv,
            "expiry": expiry,
        })

    filtered.sort(key=lambda x: x["strike"])

    vix = get_india_vix()
    tte_days = get_tte_days(expiry)

    return {
        "symbol": symbol,
        "spot": spot_price,
        "atm": atm_strike,
        "expiry": expiry,
        "all_expiries": all_expiries,
        "vix": vix,
        "tte_days": tte_days,
        "options": filtered,
    }