from jugaad_data.nse import NSELive
from datetime import datetime
import math

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
        return 0.15
    
def get_tte_days(expiry_str):
    expiry_dt = datetime.strptime(expiry_str, "%d-%b-%Y")
    now = datetime.now()
    diff = expiry_dt - now
    total_days = diff.total_seconds() / 86400

    if total_days <= 0:
        market_close = expiry_dt.replace(hour=15, minute=30, second=0, microsecond=0)
        remaining = (market_close - now).total_seconds() / 86400
        return max(remaining, 1/1440)  # minimum 1 minute

    return total_days

def get_spot_and_premiums(symbol="NIFTY", expiry=None, otm_range=20, itm_range=2):
    # get all expiries and spot from main chain
    chain = n.index_option_chain(symbol)
    spot_price = chain["records"]["underlyingValue"]
    all_expiries = chain["records"]["expiryDates"]

    if expiry is None:
        expiry = all_expiries[0]

    expiry_formatted = format_expiry_for_comparison(expiry)
    atm_strike = round(spot_price / 50) * 50

    # use option_chain_v3 with specific expiry to get all strikes
    chain_v3 = n.get('option_chain_v3', {'symbol': symbol, 'expiry': expiry})
    all_data = chain_v3.get('filtered', {}).get('data', [])

    filtered = []
    for record in all_data:
        ce_expiry = record.get("CE", {}).get("expiryDate", "")
        pe_expiry = record.get("PE", {}).get("expiryDate", "")

        if ce_expiry != expiry_formatted and pe_expiry != expiry_formatted:
            continue

        strike = record["strikePrice"]
        diff = (strike - atm_strike) / 50

        if diff > otm_range or diff < -otm_range:
            continue

        ce_premium = record.get("CE", {}).get("lastPrice", 0)
        pe_premium = record.get("PE", {}).get("lastPrice", 0)
        ce_iv = record.get("CE", {}).get("impliedVolatility", 0)
        pe_iv = record.get("PE", {}).get("impliedVolatility", 0)

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