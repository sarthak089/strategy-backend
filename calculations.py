from scipy.stats import norm
import math

def black_scholes_delta(spot, strike, tte_days, iv, option_type, risk_free_rate=0.0675):
    if tte_days <= 0 or iv <= 0:
        if option_type == "CE":
            return 1.0 if spot > strike else 0.0
        else:
            return -1.0 if spot < strike else 0.0

    T = tte_days / 365
    S = spot
    K = strike
    r = risk_free_rate
    sigma = iv

    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        if option_type == "CE":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    except:
        return 0.0

LOT_SIZE = 65

class LinearSegment:
    def __init__(self, min_spot, max_spot, slope, intercept):
        self.min_spot = min_spot
        self.max_spot = max_spot
        self.slope = slope
        self.intercept = intercept

class PayOffBenchmarks:
    def __init__(self):
        self.max_profit = float('nan')
        self.max_loss = float('nan')
        self.risk_reward = float('nan')
        self.pop = float('nan')
        self.pol = float('nan')
        self.pmp = float('nan')
        self.pml = float('nan')
        self.expected_value = float('nan')

class Option:
    def __init__(self, strike, option_type, premium):
        self.strike = strike
        self.option_type = option_type
        self.premium = premium

class OptionsWithQuantity:
    def __init__(self, option, quantity):
        self.option = option
        self.quantity = quantity

def get_payoff_at_spot(spot, segments):
    for seg in segments:
        if seg.min_spot <= spot <= seg.max_spot:
            return seg.slope * spot + seg.intercept
    last = segments[-1]
    return last.slope * spot + last.intercept

def get_finite_vertices(segments):
    vertices = set()
    for seg in segments:
        if not math.isinf(seg.min_spot):
            vertices.add(seg.min_spot)
        if not math.isinf(seg.max_spot):
            vertices.add(seg.max_spot)
    return sorted(vertices)

def get_flat_intervals_for_value(segments, target_value, use_log_normal=False):
    tolerance = 1e-6
    intervals = []
    lower_clip = 1e-6 if use_log_normal else 0.0
    ordered = sorted(segments, key=lambda s: s.min_spot)

    for seg in ordered:
        if abs(seg.slope) < 1e-9 and abs(seg.intercept - target_value) < tolerance:
            min_val = max(seg.min_spot, lower_clip)
            max_val = seg.max_spot
            if min_val < max_val:
                intervals.append((min_val, max_val))

    if ordered:
        left = ordered[0]
        if abs(left.slope) < 1e-9 and abs(left.intercept - target_value) < tolerance:
            left_min = max(float('-inf'), lower_clip)
            left_max = left.min_spot
            if left_min < left_max:
                intervals.insert(0, (left_min, left_max))

    if ordered:
        right = ordered[-1]
        if abs(right.slope) < 1e-9 and abs(right.intercept - target_value) < tolerance:
            right_min = right.max_spot
            right_max = float('inf')
            if right_min < right_max:
                intervals.append((right_min, right_max))

    return intervals

def get_positive_payoff_intervals(segments):
    intervals = []
    ordered = sorted(segments, key=lambda s: s.min_spot)

    for seg in ordered:
        a = seg.min_spot
        b = seg.max_spot

        if abs(seg.slope) < 1e-9:
            if seg.intercept > 0:
                intervals.append((a, b))
            continue

        zero_cross = -seg.intercept / seg.slope

        if seg.slope > 0:
            pos_start = max(a, zero_cross)
            pos_end = b
        else:
            pos_start = a
            pos_end = min(b, zero_cross)

        pos_start = max(pos_start, a)
        pos_end = min(pos_end, b)

        if pos_start < pos_end:
            intervals.append((pos_start, pos_end))

    return intervals

def compute_normal_prob_over_intervals(intervals, mu, std):
    prob = 0.0
    for (a, b) in intervals:
        cdf_a = norm.cdf(a, loc=mu, scale=std) if not math.isinf(a) else 0.0
        cdf_b = norm.cdf(b, loc=mu, scale=std) if not math.isinf(b) else 1.0
        prob += cdf_b - cdf_a
    return prob * 100

def get_linear_segments(options_with_qty_list, total_pnl_hitherto=0):
    segments = []
    strikes = []
    entry_cost = 0.0
    call_pos_by_strike = {}
    put_pos_by_strike = {}

    for owq in options_with_qty_list:
        opt = owq.option
        qty = owq.quantity
        entry_cost += qty * opt.premium

        strike = opt.strike
        if opt.option_type == "CE":
            call_pos_by_strike[strike] = call_pos_by_strike.get(strike, 0) + qty
        else:
            put_pos_by_strike[strike] = put_pos_by_strike.get(strike, 0) + qty

        if strike not in strikes:
            strikes.append(strike)

    strikes.sort()

    if not strikes:
        segments.append(LinearSegment(
            min_spot=float('-inf'),
            max_spot=float('inf'),
            slope=0,
            intercept=total_pnl_hitherto - entry_cost
        ))
        return segments

    base_constant = total_pnl_hitherto - entry_cost
    total_put_pos = sum(put_pos_by_strike.values())
    current_slope = -1 * total_put_pos
    current_constant = base_constant

    for strike, qty in put_pos_by_strike.items():
        current_constant += qty * strike

    segments.append(LinearSegment(
        min_spot=float('-inf'),
        max_spot=strikes[0],
        slope=current_slope,
        intercept=current_constant
    ))

    for i, K in enumerate(strikes):
        call_pos_at_k = call_pos_by_strike.get(K, 0)
        put_pos_at_k = put_pos_by_strike.get(K, 0)
        delta_slope = call_pos_at_k + put_pos_at_k
        delta_constant = -(call_pos_at_k * K) - (put_pos_at_k * K)
        current_slope += delta_slope
        current_constant += delta_constant
        current_max = strikes[i + 1] if i < len(strikes) - 1 else float('inf')
        segments.append(LinearSegment(
            min_spot=K,
            max_spot=current_max,
            slope=current_slope,
            intercept=current_constant
        ))

    return segments

def get_payoff_benchmarks(segments, spot_price, atm_iv, tte_days):
    bench = PayOffBenchmarks()
    ordered = sorted(segments, key=lambda s: s.min_spot)

    left_slope = ordered[0].slope
    right_slope = ordered[-1].slope

    loss_unbounded = left_slope > 0 or right_slope < 0
    profit_unbounded = left_slope < 0 or right_slope > 0

    max_loss = float('-inf') if loss_unbounded else float('nan')
    max_profit = float('inf') if profit_unbounded else float('nan')

    vertices = get_finite_vertices(ordered)
    if not vertices:
        return bench

    payoffs = [get_payoff_at_spot(v, ordered) for v in vertices]

    if not loss_unbounded:
        max_loss = min(payoffs)
    if not profit_unbounded:
        max_profit = max(payoffs)

    if max_loss >= 0 or max_profit <= 0:
        return bench

    if math.isinf(max_loss):
        risk_reward = 0
    elif math.isinf(max_profit):
        risk_reward = float('inf')
    else:
        risk_reward = abs(max_profit / max_loss)

    bench.max_loss = max_loss
    bench.max_profit = max_profit
    bench.risk_reward = risk_reward

    if tte_days <= 0:
        current_payoff = get_payoff_at_spot(spot_price, ordered)
        bench.pop = 100.0 if current_payoff > 0 else 0.0
        bench.pol = 100 - bench.pop
        return bench

    ltp = max(spot_price, 1e-6)
    daily_vol = atm_iv / math.sqrt(365)
    std_dev = daily_vol * math.sqrt(tte_days)
    abs_std = ltp * std_dev

    if abs_std <= 0:
        return bench

    if not math.isinf(max_profit):
        max_profit_intervals = get_flat_intervals_for_value(ordered, max_profit, False)
        bench.pmp = compute_normal_prob_over_intervals(max_profit_intervals, ltp, abs_std)

    if not math.isinf(max_loss):
        max_loss_intervals = get_flat_intervals_for_value(ordered, max_loss, False)
        bench.pml = compute_normal_prob_over_intervals(max_loss_intervals, ltp, abs_std)

    positive_intervals = get_positive_payoff_intervals(ordered)
    bench.pop = compute_normal_prob_over_intervals(positive_intervals, ltp, abs_std)
    bench.pol = 100 - bench.pop

    return bench

# ── Build strategy legs ──

def build_bull_call_spread(buy_strike, sell_strike, buy_ce_premium, sell_ce_premium):
    # Buy lower CE + Sell higher CE (debit)
    buy_option  = Option(strike=buy_strike,  option_type="CE", premium=buy_ce_premium)
    sell_option = Option(strike=sell_strike, option_type="CE", premium=sell_ce_premium)
    return [
        OptionsWithQuantity(buy_option,   1 * LOT_SIZE),
        OptionsWithQuantity(sell_option, -1 * LOT_SIZE),
    ]

def build_bear_put_spread(sell_strike, buy_strike, sell_pe_premium, buy_pe_premium):
    # Buy higher PE + Sell lower PE (debit)
    sell_option = Option(strike=sell_strike, option_type="PE", premium=sell_pe_premium)
    buy_option  = Option(strike=buy_strike,  option_type="PE", premium=buy_pe_premium)
    return [
        OptionsWithQuantity(sell_option, -1 * LOT_SIZE),
        OptionsWithQuantity(buy_option,   1 * LOT_SIZE),
    ]

def build_bull_put_spread(sell_strike, buy_strike, sell_pe_premium, buy_pe_premium):
    # Sell higher PE + Buy lower PE (credit)
    sell_option = Option(strike=sell_strike, option_type="PE", premium=sell_pe_premium)
    buy_option  = Option(strike=buy_strike,  option_type="PE", premium=buy_pe_premium)
    return [
        OptionsWithQuantity(sell_option, -1 * LOT_SIZE),
        OptionsWithQuantity(buy_option,   1 * LOT_SIZE),
    ]

def build_bear_call_spread(sell_strike, buy_strike, sell_ce_premium, buy_ce_premium):
    # Sell lower CE + Buy higher CE (credit)
    sell_option = Option(strike=sell_strike, option_type="CE", premium=sell_ce_premium)
    buy_option  = Option(strike=buy_strike,  option_type="CE", premium=buy_ce_premium)
    return [
        OptionsWithQuantity(sell_option, -1 * LOT_SIZE),
        OptionsWithQuantity(buy_option,   1 * LOT_SIZE),
    ]

def build_iron_condor(sell_put, buy_put, sell_call, buy_call,
                      sell_put_premium, buy_put_premium,
                      sell_call_premium, buy_call_premium):
    # Sell OTM PE + Buy further OTM PE + Sell OTM CE + Buy further OTM CE
    return [
        OptionsWithQuantity(Option(sell_put,  "PE", sell_put_premium),  -1 * LOT_SIZE),
        OptionsWithQuantity(Option(buy_put,   "PE", buy_put_premium),    1 * LOT_SIZE),
        OptionsWithQuantity(Option(sell_call, "CE", sell_call_premium), -1 * LOT_SIZE),
        OptionsWithQuantity(Option(buy_call,  "CE", buy_call_premium),   1 * LOT_SIZE),
    ]

def build_iron_butterfly(atm_strike, buy_put, buy_call,
                         atm_put_premium, atm_call_premium,
                         buy_put_premium, buy_call_premium):
    # Sell ATM PE + Sell ATM CE + Buy OTM PE + Buy OTM CE
    return [
        OptionsWithQuantity(Option(atm_strike, "PE", atm_put_premium),  -1 * LOT_SIZE),
        OptionsWithQuantity(Option(atm_strike, "CE", atm_call_premium), -1 * LOT_SIZE),
        OptionsWithQuantity(Option(buy_put,    "PE", buy_put_premium),   1 * LOT_SIZE),
        OptionsWithQuantity(Option(buy_call,   "CE", buy_call_premium),  1 * LOT_SIZE),
    ]

def build_call_butterfly(buy_lower, sell_middle, buy_upper,
                         buy_lower_premium, sell_middle_premium, buy_upper_premium):
    # Buy lower CE + Sell 2x middle CE + Buy upper CE
    return [
        OptionsWithQuantity(Option(buy_lower,   "CE", buy_lower_premium),    1 * LOT_SIZE),
        OptionsWithQuantity(Option(sell_middle, "CE", sell_middle_premium), -2 * LOT_SIZE),
        OptionsWithQuantity(Option(buy_upper,   "CE", buy_upper_premium),    1 * LOT_SIZE),
    ]

def build_put_butterfly(buy_lower, sell_middle, buy_upper,
                        buy_lower_premium, sell_middle_premium, buy_upper_premium):
    # Buy lower PE + Sell 2x middle PE + Buy upper PE
    return [
        OptionsWithQuantity(Option(buy_lower,   "PE", buy_lower_premium),    1 * LOT_SIZE),
        OptionsWithQuantity(Option(sell_middle, "PE", sell_middle_premium), -2 * LOT_SIZE),
        OptionsWithQuantity(Option(buy_upper,   "PE", buy_upper_premium),    1 * LOT_SIZE),
    ]

def build_broken_wing_call_butterfly(buy_lower, sell_middle, buy_upper,
                                     buy_lower_premium, sell_middle_premium, buy_upper_premium):
    # Buy lower CE + Sell 2x middle CE + Buy upper CE (upper wing wider)
    return [
        OptionsWithQuantity(Option(buy_lower,   "CE", buy_lower_premium),    1 * LOT_SIZE),
        OptionsWithQuantity(Option(sell_middle, "CE", sell_middle_premium), -2 * LOT_SIZE),
        OptionsWithQuantity(Option(buy_upper,   "CE", buy_upper_premium),    1 * LOT_SIZE),
    ]

def build_broken_wing_put_butterfly(buy_lower, sell_middle, buy_upper,
                                    buy_lower_premium, sell_middle_premium, buy_upper_premium):
    # Buy lower PE + Sell 2x middle PE + Buy upper PE (lower wing wider)
    return [
        OptionsWithQuantity(Option(buy_lower,   "PE", buy_lower_premium),    1 * LOT_SIZE),
        OptionsWithQuantity(Option(sell_middle, "PE", sell_middle_premium), -2 * LOT_SIZE),
        OptionsWithQuantity(Option(buy_upper,   "PE", buy_upper_premium),    1 * LOT_SIZE),
    ]

# ── Main strategy generator ──

def generate_strategies(options, spot, atm_iv, tte_days, min_rr=1.0, min_pop=40.0, max_loss=None, bias=None, leg_count=None):
    if leg_count is None or leg_count == 'Up To 4':
        allowed_legs = [2, 4]
    elif leg_count == 'Up To 3':
        allowed_legs = [2]
    elif leg_count == 'Fix 2':
        allowed_legs = [2]
    elif leg_count == 'Fix 3':
        allowed_legs = []
    elif leg_count == 'Fix 4':
        allowed_legs = [4]
    else:
        allowed_legs = [2, 4]

    results = []
    strike_map = {o["strike"]: o for o in options}
    strikes = sorted(strike_map.keys())

    def passes_filters(bench):
        return (
            not math.isnan(bench.risk_reward) and
            not math.isnan(bench.pop) and
            bench.risk_reward >= min_rr and
            bench.pop >= min_pop and
            (max_loss is None or abs(bench.max_loss) <= max_loss)
        )

    def append_result(strategy_name, legs_str, bench, net_credit, delta, strategy_bias):
        results.append({
            "strategy": strategy_name,
            "legs": legs_str,
            "rr": f"1 : {round(bench.risk_reward, 1)}",
            "pop": round(bench.pop, 2),
            "net_credit": round(net_credit, 2),
            "max_profit": round(bench.max_profit, 2),
            "max_loss": round(bench.max_loss, 2),
            "delta": round(delta, 4),
            "bias": strategy_bias,
        })

    def calc_strategy_delta(legs_data):
        total_delta = 0.0
        for leg in legs_data:
            d = black_scholes_delta(spot, leg["strike"], tte_days, leg["iv"], leg["option_type"])
            total_delta += d * leg["quantity"]
        return round(total_delta , 4)

    # ── 2 LEG STRATEGIES ──
    if 2 in allowed_legs:
        for i, strike_a in enumerate(strikes):
            for strike_b in strikes[i + 1:]:
                data_a = strike_map[strike_a]
                data_b = strike_map[strike_b]

                # Bull Call Spread — Buy lower CE + Sell higher CE (debit, bullish)
                if bias in (None, 'Bullish'):
                    if data_a["ce_premium"] > 0 and data_b["ce_premium"] > 0:
                        legs = build_bull_call_spread(strike_a, strike_b, data_a["ce_premium"], data_b["ce_premium"])
                        bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                        if passes_filters(bench):
                            delta = calc_strategy_delta([
                                {"strike": strike_a, "option_type": "CE", "quantity": 1,  "iv": data_a["ce_iv"] / 100},
                                {"strike": strike_b, "option_type": "CE", "quantity": -1, "iv": data_b["ce_iv"] / 100},
                            ])
                            append_result(
                                "Bull Call Spread (2L)",
                                f"B {strike_a}CE | S {strike_b}CE",
                                bench,
                                (data_b["ce_premium"] - data_a["ce_premium"]) * LOT_SIZE,
                                delta,
                                "Bullish"
                            )

                # Bear Call Spread — Sell lower CE + Buy higher CE (credit, bearish)
                if bias in (None, 'Bearish'):
                    if data_a["ce_premium"] > 0 and data_b["ce_premium"] > 0:
                        legs = build_bear_call_spread(strike_a, strike_b, data_a["ce_premium"], data_b["ce_premium"])
                        bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                        if passes_filters(bench):
                            delta = calc_strategy_delta([
                                {"strike": strike_a, "option_type": "CE", "quantity": -1, "iv": data_a["ce_iv"] / 100},
                                {"strike": strike_b, "option_type": "CE", "quantity": 1,  "iv": data_b["ce_iv"] / 100},
                            ])
                            append_result(
                                "Bear Call Spread (2L)",
                                f"S {strike_a}CE | B {strike_b}CE",
                                bench,
                                (data_a["ce_premium"] - data_b["ce_premium"]) * LOT_SIZE,
                                delta,
                                "Bearish"
                            )

                # Bear Put Spread — Buy higher PE + Sell lower PE (debit, bearish)
                if bias in (None, 'Bearish'):
                    if data_a["pe_premium"] > 0 and data_b["pe_premium"] > 0:
                        legs = build_bear_put_spread(strike_a, strike_b, data_a["pe_premium"], data_b["pe_premium"])
                        bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                        if passes_filters(bench):
                            delta = calc_strategy_delta([
                                {"strike": strike_a, "option_type": "PE", "quantity": -1, "iv": data_a["pe_iv"] / 100},
                                {"strike": strike_b, "option_type": "PE", "quantity": 1,  "iv": data_b["pe_iv"] / 100},
                            ])
                            append_result(
                                "Bear Put Spread (2L)",
                                f"S {strike_a}PE | B {strike_b}PE",
                                bench,
                                (data_a["pe_premium"] - data_b["pe_premium"]) * LOT_SIZE,
                                delta,
                                "Bearish"
                            )

                # Bull Put Spread — Sell higher PE + Buy lower PE (credit, bullish)
                if bias in (None, 'Bullish'):
                    if data_a["pe_premium"] > 0 and data_b["pe_premium"] > 0:
                        legs = build_bull_put_spread(strike_b, strike_a, data_b["pe_premium"], data_a["pe_premium"])
                        bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                        if passes_filters(bench):
                            delta = calc_strategy_delta([
                                {"strike": strike_b, "option_type": "PE", "quantity": -1, "iv": data_b["pe_iv"] / 100},
                                {"strike": strike_a, "option_type": "PE", "quantity": 1,  "iv": data_a["pe_iv"] / 100},
                            ])
                            append_result(
                                "Bull Put Spread (2L)",
                                f"S {strike_b}PE | B {strike_a}PE",
                                bench,
                                (data_b["pe_premium"] - data_a["pe_premium"]) * LOT_SIZE,
                                delta,
                                "Bullish"
                            )

    # ── 4 LEG STRATEGIES ──
    if 4 in allowed_legs:
        MAX_WING = 150
        MAX_MIDDLE_GAP = 50

        for i, strike_a in enumerate(strikes):
            for j, strike_b in enumerate(strikes[i + 1:], i + 1):
                if strike_b - strike_a > MAX_WING:
                    break
                for k, strike_c in enumerate(strikes[j + 1:], j + 1):
                    if strike_c - strike_b > MAX_MIDDLE_GAP:
                        break
                    for strike_d in strikes[k + 1:]:
                        if strike_d - strike_c > MAX_WING:
                            break

                        data_a = strike_map[strike_a]
                        data_b = strike_map[strike_b]
                        data_c = strike_map[strike_c]
                        data_d = strike_map[strike_d]

                        # Iron Condor
                        if bias in (None, 'Neutral'):
                            if (data_a["pe_premium"] > 0 and data_b["pe_premium"] > 0 and
                                data_c["ce_premium"] > 0 and data_d["ce_premium"] > 0):
                                legs = build_iron_condor(
                                    strike_b, strike_a, strike_c, strike_d,
                                    data_b["pe_premium"], data_a["pe_premium"],
                                    data_c["ce_premium"], data_d["ce_premium"]
                                )
                                bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                                if passes_filters(bench):
                                    delta = calc_strategy_delta([
                                        {"strike": strike_a, "option_type": "PE", "quantity": 1,  "iv": data_a["pe_iv"] / 100},
                                        {"strike": strike_b, "option_type": "PE", "quantity": -1, "iv": data_b["pe_iv"] / 100},
                                        {"strike": strike_c, "option_type": "CE", "quantity": -1, "iv": data_c["ce_iv"] / 100},
                                        {"strike": strike_d, "option_type": "CE", "quantity": 1,  "iv": data_d["ce_iv"] / 100},
                                    ])
                                    net_credit = (data_b["pe_premium"] - data_a["pe_premium"] +
                                                  data_c["ce_premium"] - data_d["ce_premium"]) * LOT_SIZE
                                    append_result(
                                        "Iron Condor (4L)",
                                        f"B {strike_a}PE | S {strike_b}PE | S {strike_c}CE | B {strike_d}CE",
                                        bench, net_credit, delta, "Neutral"
                                    )

                        # Iron Butterfly
                        if bias in (None, 'Neutral') and strike_b == strike_c:
                            if (data_b["pe_premium"] > 0 and data_b["ce_premium"] > 0 and
                                data_a["pe_premium"] > 0 and data_d["ce_premium"] > 0):
                                legs = build_iron_butterfly(
                                    strike_b, strike_a, strike_d,
                                    data_b["pe_premium"], data_b["ce_premium"],
                                    data_a["pe_premium"], data_d["ce_premium"]
                                )
                                bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                                if passes_filters(bench):
                                    delta = calc_strategy_delta([
                                        {"strike": strike_a, "option_type": "PE", "quantity": 1,  "iv": data_a["pe_iv"] / 100},
                                        {"strike": strike_b, "option_type": "PE", "quantity": -1, "iv": data_b["pe_iv"] / 100},
                                        {"strike": strike_c, "option_type": "CE", "quantity": -1, "iv": data_c["ce_iv"] / 100},
                                        {"strike": strike_d, "option_type": "CE", "quantity": 1,  "iv": data_d["ce_iv"] / 100},
                                    ])
                                    net_credit = (data_b["pe_premium"] + data_b["ce_premium"] -
                                                  data_a["pe_premium"] - data_d["ce_premium"]) * LOT_SIZE
                                    append_result(
                                        "Iron Butterfly (4L)",
                                        f"B {strike_a}PE | S {strike_b}PE | S {strike_c}CE | B {strike_d}CE",
                                        bench, net_credit, delta, "Neutral"
                                    )

        # Butterfly strategies — 3 strikes
        for i, strike_a in enumerate(strikes):
            for j, strike_b in enumerate(strikes[i + 1:], i + 1):
                if strike_b - strike_a > MAX_WING:
                    break
                for strike_c in strikes[j + 1:]:
                    if strike_c - strike_b > MAX_WING:
                        break

                    data_a = strike_map[strike_a]
                    data_b = strike_map[strike_b]
                    data_c = strike_map[strike_c]

                    lower_wing = strike_b - strike_a
                    upper_wing = strike_c - strike_b

                    # Call Butterfly — equal wings
                    if bias in (None, 'Neutral') and lower_wing == upper_wing:
                        if data_a["ce_premium"] > 0 and data_b["ce_premium"] > 0 and data_c["ce_premium"] > 0:
                            legs = build_call_butterfly(strike_a, strike_b, strike_c, data_a["ce_premium"], data_b["ce_premium"], data_c["ce_premium"])
                            bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                            if passes_filters(bench):
                                delta = calc_strategy_delta([
                                    {"strike": strike_a, "option_type": "CE", "quantity": 1,  "iv": data_a["ce_iv"] / 100},
                                    {"strike": strike_b, "option_type": "CE", "quantity": -2, "iv": data_b["ce_iv"] / 100},
                                    {"strike": strike_c, "option_type": "CE", "quantity": 1,  "iv": data_c["ce_iv"] / 100},
                                ])
                                net_credit = (2 * data_b["ce_premium"] - data_a["ce_premium"] - data_c["ce_premium"]) * LOT_SIZE
                                append_result(
                                    "Call Butterfly (4L)",
                                    f"B {strike_a}CE | S {strike_b}CE | S {strike_b}CE | B {strike_c}CE",
                                    bench, net_credit, delta, "Neutral"
                                )

                    # Broken Wing Call Butterfly — upper wing wider
                    if bias in (None, 'Bearish') and upper_wing > lower_wing:
                        if data_a["ce_premium"] > 0 and data_b["ce_premium"] > 0 and data_c["ce_premium"] > 0:
                            legs = build_broken_wing_call_butterfly(strike_a, strike_b, strike_c, data_a["ce_premium"], data_b["ce_premium"], data_c["ce_premium"])
                            bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                            if passes_filters(bench):
                                delta = calc_strategy_delta([
                                    {"strike": strike_a, "option_type": "CE", "quantity": 1,  "iv": data_a["ce_iv"] / 100},
                                    {"strike": strike_b, "option_type": "CE", "quantity": -2, "iv": data_b["ce_iv"] / 100},
                                    {"strike": strike_c, "option_type": "CE", "quantity": 1,  "iv": data_c["ce_iv"] / 100},
                                ])
                                net_credit = (2 * data_b["ce_premium"] - data_a["ce_premium"] - data_c["ce_premium"]) * LOT_SIZE
                                append_result(
                                    "Broken Wing Call Butterfly (4L)",
                                    f"B {strike_a}CE | S {strike_b}CE | S {strike_b}CE | B {strike_c}CE",
                                    bench, net_credit, delta, "Bearish"
                                )

                    # Put Butterfly — equal wings
                    if bias in (None, 'Neutral') and lower_wing == upper_wing:
                        if data_a["pe_premium"] > 0 and data_b["pe_premium"] > 0 and data_c["pe_premium"] > 0:
                            legs = build_put_butterfly(strike_a, strike_b, strike_c, data_a["pe_premium"], data_b["pe_premium"], data_c["pe_premium"])
                            bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                            if passes_filters(bench):
                                delta = calc_strategy_delta([
                                    {"strike": strike_a, "option_type": "PE", "quantity": 1,  "iv": data_a["pe_iv"] / 100},
                                    {"strike": strike_b, "option_type": "PE", "quantity": -2, "iv": data_b["pe_iv"] / 100},
                                    {"strike": strike_c, "option_type": "PE", "quantity": 1,  "iv": data_c["pe_iv"] / 100},
                                ])
                                net_credit = (2 * data_b["pe_premium"] - data_a["pe_premium"] - data_c["pe_premium"]) * LOT_SIZE
                                append_result(
                                    "Put Butterfly (4L)",
                                    f"B {strike_a}PE | S {strike_b}PE | S {strike_b}PE | B {strike_c}PE",
                                    bench, net_credit, delta, "Neutral"
                                )

                    # Broken Wing Put Butterfly — lower wing wider
                    if bias in (None, 'Bullish') and lower_wing > upper_wing:
                        if data_a["pe_premium"] > 0 and data_b["pe_premium"] > 0 and data_c["pe_premium"] > 0:
                            legs = build_broken_wing_put_butterfly(strike_a, strike_b, strike_c, data_a["pe_premium"], data_b["pe_premium"], data_c["pe_premium"])
                            bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                            if passes_filters(bench):
                                delta = calc_strategy_delta([
                                    {"strike": strike_a, "option_type": "PE", "quantity": 1,  "iv": data_a["pe_iv"] / 100},
                                    {"strike": strike_b, "option_type": "PE", "quantity": -2, "iv": data_b["pe_iv"] / 100},
                                    {"strike": strike_c, "option_type": "PE", "quantity": 1,  "iv": data_c["pe_iv"] / 100},
                                ])
                                net_credit = (2 * data_b["pe_premium"] - data_a["pe_premium"] - data_c["pe_premium"]) * LOT_SIZE
                                append_result(
                                    "Broken Wing Put Butterfly (4L)",
                                    f"B {strike_a}PE | S {strike_b}PE | S {strike_b}PE | B {strike_c}PE",
                                    bench, net_credit, delta, "Bullish"
                                )

    results.sort(key=lambda x: float(x["rr"].split(": ")[1]), reverse=True)
    return results
# def generate_strategies(options, spot, atm_iv, tte_days, min_rr=1.0, min_pop=40.0, max_loss=None, bias=None, leg_count=None):
#     print(f">>> generate_strategies called with leg_count='{leg_count}' type={type(leg_count)}")
#     print(f">>> allowed_legs will be calculated next")
#     if leg_count is None or leg_count == 'Up To 4':
#        allowed_legs = [2, 4]
#     elif leg_count == 'Up To 3':
#         allowed_legs = [2]
#     elif leg_count == 'Fix 2':
#         allowed_legs = [2]
#     elif leg_count == 'Fix 3':
#         allowed_legs = []
#     elif leg_count == 'Fix 4':
#         allowed_legs = [4]
#     else:
#         allowed_legs = [2, 4]

#     results = []
#     strike_map = {o["strike"]: o for o in options}
#     strikes = sorted(strike_map.keys())

#     def passes_filters(bench):
#         return (
#             not math.isnan(bench.risk_reward) and
#             not math.isnan(bench.pop) and
#             bench.risk_reward >= min_rr and
#             bench.pop >= min_pop and
#             (max_loss is None or abs(bench.max_loss) <= max_loss)
#         )

#     def append_result(strategy_name, legs_str, bench, net_credit, delta, strategy_bias):
#         results.append({
#             "strategy": strategy_name,
#             "legs": legs_str,
#             "rr": f"1 : {round(bench.risk_reward, 1)}",
#             "pop": round(bench.pop, 2),
#             "net_credit": round(net_credit, 2),
#             "max_profit": round(bench.max_profit, 2),
#             "max_loss": round(bench.max_loss, 2),
#             "delta": round(delta, 4),
#             "bias": strategy_bias,
#         })
  

#     # ── 2 LEG STRATEGIES ──
#     if 2 in allowed_legs:
#         for i, strike_a in enumerate(strikes):
#             for strike_b in strikes[i + 1:]:
#                 data_a = strike_map[strike_a]
#                 data_b = strike_map[strike_b]

#                 # Bull Call Spread — Buy lower CE + Sell higher CE (debit, bullish)
#                 if bias in (None, 'Bullish'):
#                     if data_a["ce_premium"] > 0 and data_b["ce_premium"] > 0:
#                         legs = build_bull_call_spread(strike_a, strike_b, data_a["ce_premium"], data_b["ce_premium"])
#                         bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
#                         if passes_filters(bench):
#                             append_result(
#                                 "Bull Call Spread (2L)",
#                                 f"B {strike_a}CE | S {strike_b}CE",
#                                 bench,
#                                 (data_b["ce_premium"] - data_a["ce_premium"]) * LOT_SIZE,
#                                 (data_a["ce_premium"] - data_b["ce_premium"]) / spot,
#                                 "Bullish"
#                             )

#                 # Bear Call Spread — Sell lower CE + Buy higher CE (credit, bearish)
#                 if bias in (None, 'Bearish'):
#                     if data_a["ce_premium"] > 0 and data_b["ce_premium"] > 0:
#                         legs = build_bear_call_spread(strike_a, strike_b, data_a["ce_premium"], data_b["ce_premium"])
#                         bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
#                         if passes_filters(bench):
#                             append_result(
#                                 "Bear Call Spread (2L)",
#                                 f"S {strike_a}CE | B {strike_b}CE",
#                                 bench,
#                                 (data_a["ce_premium"] - data_b["ce_premium"]) * LOT_SIZE,
#                                 (data_b["ce_premium"] - data_a["ce_premium"]) / spot,
#                                 "Bearish"
#                             )

#                 # Bear Put Spread — Buy higher PE + Sell lower PE (debit, bearish)
#                 if bias in (None, 'Bearish'):
#                     if data_a["pe_premium"] > 0 and data_b["pe_premium"] > 0:
#                         legs = build_bear_put_spread(strike_a, strike_b, data_a["pe_premium"], data_b["pe_premium"])
#                         bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
#                         if passes_filters(bench):
#                             append_result(
#                                 "Bear Put Spread (2L)",
#                                 f"S {strike_a}PE | B {strike_b}PE",
#                                 bench,
#                                 (data_a["pe_premium"] - data_b["pe_premium"]) * LOT_SIZE,
#                                 (data_b["pe_premium"] - data_a["pe_premium"]) / spot,
#                                 "Bearish"
#                             )

#                 # Bull Put Spread — Sell higher PE + Buy lower PE (credit, bullish)
#                 if bias in (None, 'Bullish'):
#                     if data_a["pe_premium"] > 0 and data_b["pe_premium"] > 0:
#                         legs = build_bull_put_spread(strike_b, strike_a, data_b["pe_premium"], data_a["pe_premium"])
#                         bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
#                         if passes_filters(bench):
#                             append_result(
#                                 "Bull Put Spread (2L)",
#                                 f"S {strike_b}PE | B {strike_a}PE",
#                                 bench,
#                                 (data_b["pe_premium"] - data_a["pe_premium"]) * LOT_SIZE,
#                                 (data_a["pe_premium"] - data_b["pe_premium"]) / spot,
#                                 "Bullish"
#                             )

    # ── 4 LEG STRATEGIES ──
    # if 4 in allowed_legs:
    #     for i, strike_a in enumerate(strikes):
    #         for j, strike_b in enumerate(strikes[i + 1:], i + 1):
    #             for k, strike_c in enumerate(strikes[j + 1:], j + 1):
    #                 for strike_d in strikes[k + 1:]:
    #                     data_a = strike_map[strike_a]
    #                     data_b = strike_map[strike_b]
    #                     data_c = strike_map[strike_c]
    #                     data_d = strike_map[strike_d]

    #                     # Iron Condor — Buy low PE + Sell mid-low PE + Sell mid-high CE + Buy high CE
    #                     if bias in (None, 'Neutral'):
    #                         if (data_a["pe_premium"] > 0 and data_b["pe_premium"] > 0 and
    #                             data_c["ce_premium"] > 0 and data_d["ce_premium"] > 0):
    #                             legs = build_iron_condor(
    #                                 strike_b, strike_a, strike_c, strike_d,
    #                                 data_b["pe_premium"], data_a["pe_premium"],
    #                                 data_c["ce_premium"], data_d["ce_premium"]
    #                             )
    #                             bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
    #                             if passes_filters(bench):
    #                                 net_credit = (data_b["pe_premium"] - data_a["pe_premium"] +
    #                                               data_c["ce_premium"] - data_d["ce_premium"]) * LOT_SIZE
    #                                 append_result(
    #                                     "Iron Condor (4L)",
    #                                     f"B {strike_a}PE | S {strike_b}PE | S {strike_c}CE | B {strike_d}CE",
    #                                     bench, net_credit, 0.0, "Neutral"
    #                                 )

    #                     # Iron Butterfly — only when strike_b == strike_c (ATM)
    #                     if bias in (None, 'Neutral'):
    #                         if (data_b["pe_premium"] > 0 and data_b["ce_premium"] > 0 and
    #                             data_a["pe_premium"] > 0 and data_d["ce_premium"] > 0 and
    #                             strike_b == strike_c):
    #                             legs = build_iron_butterfly(
    #                                 strike_b, strike_a, strike_d,
    #                                 data_b["pe_premium"], data_b["ce_premium"],
    #                                 data_a["pe_premium"], data_d["ce_premium"]
    #                             )
    #                             bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
    #                             if passes_filters(bench):
    #                                 net_credit = (data_b["pe_premium"] + data_b["ce_premium"] -
    #                                               data_a["pe_premium"] - data_d["ce_premium"]) * LOT_SIZE
    #                                 append_result(
    #                                     "Iron Butterfly (4L)",
    #                                     f"B {strike_a}PE | S {strike_b}PE | S {strike_c}CE | B {strike_d}CE",
    #                                     bench, net_credit, 0.0, "Neutral"
    #                                 )

    #     # Call Butterfly and Broken Wing Call Butterfly — 3 CE strikes
    #     for i, strike_a in enumerate(strikes):
    #         for j, strike_b in enumerate(strikes[i + 1:], i + 1):
    #             for strike_c in strikes[j + 1:]:
    #                 data_a = strike_map[strike_a]
    #                 data_b = strike_map[strike_b]
    #                 data_c = strike_map[strike_c]

    #                 lower_wing = strike_b - strike_a
    #                 upper_wing = strike_c - strike_b

    #                 # Call Butterfly — equal wings
    #                 if bias in (None, 'Neutral') and lower_wing == upper_wing:
    #                     if data_a["ce_premium"] > 0 and data_b["ce_premium"] > 0 and data_c["ce_premium"] > 0:
    #                         legs = build_call_butterfly(
    #                             strike_a, strike_b, strike_c,
    #                             data_a["ce_premium"], data_b["ce_premium"], data_c["ce_premium"]
    #                         )
    #                         bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
    #                         if passes_filters(bench):
    #                             net_credit = (2 * data_b["ce_premium"] - data_a["ce_premium"] - data_c["ce_premium"]) * LOT_SIZE
    #                             append_result(
    #                                 "Call Butterfly (4L)",
    #                                 f"B {strike_a}CE | S {strike_b}CE | S {strike_b}CE | B {strike_c}CE",
    #                                 bench, net_credit,
    #                                 (data_a["ce_premium"] - 2 * data_b["ce_premium"] + data_c["ce_premium"]) / spot,
    #                                 "Neutral"
    #                             )

    #                 # Broken Wing Call Butterfly — unequal wings (upper wing wider)
    #                 if bias in (None, 'Bearish') and upper_wing > lower_wing:
    #                     if data_a["ce_premium"] > 0 and data_b["ce_premium"] > 0 and data_c["ce_premium"] > 0:
    #                         legs = build_broken_wing_call_butterfly(
    #                             strike_a, strike_b, strike_c,
    #                             data_a["ce_premium"], data_b["ce_premium"], data_c["ce_premium"]
    #                         )
    #                         bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
    #                         if passes_filters(bench):
    #                             net_credit = (2 * data_b["ce_premium"] - data_a["ce_premium"] - data_c["ce_premium"]) * LOT_SIZE
    #                             append_result(
    #                                 "Broken Wing Call Butterfly (4L)",
    #                                 f"B {strike_a}CE | S {strike_b}CE | S {strike_b}CE | B {strike_c}CE",
    #                                 bench, net_credit,
    #                                 (data_a["ce_premium"] - 2 * data_b["ce_premium"] + data_c["ce_premium"]) / spot,
    #                                 "Bearish"
    #                             )

    #                 # Put Butterfly — equal wings
    #                 if bias in (None, 'Neutral') and lower_wing == upper_wing:
    #                     if data_a["pe_premium"] > 0 and data_b["pe_premium"] > 0 and data_c["pe_premium"] > 0:
    #                         legs = build_put_butterfly(
    #                             strike_a, strike_b, strike_c,
    #                             data_a["pe_premium"], data_b["pe_premium"], data_c["pe_premium"]
    #                         )
    #                         bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
    #                         if passes_filters(bench):
    #                             net_credit = (2 * data_b["pe_premium"] - data_a["pe_premium"] - data_c["pe_premium"]) * LOT_SIZE
    #                             append_result(
    #                                 "Put Butterfly (4L)",
    #                                 f"B {strike_a}PE | S {strike_b}PE | S {strike_b}PE | B {strike_c}PE",
    #                                 bench, net_credit,
    #                                 (data_a["pe_premium"] - 2 * data_b["pe_premium"] + data_c["pe_premium"]) / spot,
    #                                 "Neutral"
    #                             )

    #                 # Broken Wing Put Butterfly — unequal wings (lower wing wider)
    #                 if bias in (None, 'Bullish') and lower_wing > upper_wing:
    #                     if data_a["pe_premium"] > 0 and data_b["pe_premium"] > 0 and data_c["pe_premium"] > 0:
    #                         legs = build_broken_wing_put_butterfly(
    #                             strike_a, strike_b, strike_c,
    #                             data_a["pe_premium"], data_b["pe_premium"], data_c["pe_premium"]
    #                         )
    #                         bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
    #                         if passes_filters(bench):
    #                             net_credit = (2 * data_b["pe_premium"] - data_a["pe_premium"] - data_c["pe_premium"]) * LOT_SIZE
    #                             append_result(
    #                                 "Broken Wing Put Butterfly (4L)",
    #                                 f"B {strike_a}PE | S {strike_b}PE | S {strike_b}PE | B {strike_c}PE",
    #                                 bench, net_credit,
    #                                 (data_a["pe_premium"] - 2 * data_b["pe_premium"] + data_c["pe_premium"]) / spot,
    #                                 "Bullish"
    #                             )


    # ── 4 LEG STRATEGIES ──
    if 4 in allowed_legs:
     MAX_WING_STEPS = 2
     MAX_WING = 150
     MAX_MIDDLE_GAP = 50
 
     # Iron Condor and Iron Butterfly
     for i, strike_a in enumerate(strikes):
         for j, strike_b in enumerate(strikes[i + 1:], i + 1):
             if strike_b - strike_a > MAX_WING:
                 break
             for k, strike_c in enumerate(strikes[j + 1:], j + 1):
                 if strike_c - strike_b > MAX_MIDDLE_GAP:
                     break
                 for strike_d in strikes[k + 1:]:
                     if strike_d - strike_c > MAX_WING:
                         break
 
                     data_a = strike_map[strike_a]
                     data_b = strike_map[strike_b]
                     data_c = strike_map[strike_c]
                     data_d = strike_map[strike_d]
 
                     # Iron Condor
                     if bias in (None, 'Neutral'):
                         if (data_a["pe_premium"] > 0 and data_b["pe_premium"] > 0 and
                             data_c["ce_premium"] > 0 and data_d["ce_premium"] > 0):
                             legs = build_iron_condor(
                                 strike_b, strike_a, strike_c, strike_d,
                                 data_b["pe_premium"], data_a["pe_premium"],
                                 data_c["ce_premium"], data_d["ce_premium"]
                             )
                             bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                             if passes_filters(bench):
                                 net_credit = (data_b["pe_premium"] - data_a["pe_premium"] +
                                               data_c["ce_premium"] - data_d["ce_premium"]) * LOT_SIZE
                                 append_result(
                                     "Iron Condor (4L)",
                                     f"B {strike_a}PE | S {strike_b}PE | S {strike_c}CE | B {strike_d}CE",
                                     bench, net_credit, 0.0, "Neutral"
                                 )
 
                     # Iron Butterfly — strike_b == strike_c
                     if bias in (None, 'Neutral') and strike_b == strike_c:
                         if (data_b["pe_premium"] > 0 and data_b["ce_premium"] > 0 and
                             data_a["pe_premium"] > 0 and data_d["ce_premium"] > 0):
                             legs = build_iron_butterfly(
                                 strike_b, strike_a, strike_d,
                                 data_b["pe_premium"], data_b["ce_premium"],
                                 data_a["pe_premium"], data_d["ce_premium"]
                             )
                             bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                             if passes_filters(bench):
                                 net_credit = (data_b["pe_premium"] + data_b["ce_premium"] -
                                               data_a["pe_premium"] - data_d["ce_premium"]) * LOT_SIZE
                                 append_result(
                                     "Iron Butterfly (4L)",
                                     f"B {strike_a}PE | S {strike_b}PE | S {strike_c}CE | B {strike_d}CE",
                                     bench, net_credit, 0.0, "Neutral"
                                 )
 
     # Butterfly strategies — 3 strikes
     for i, strike_a in enumerate(strikes):
         for j, strike_b in enumerate(strikes[i + 1:], i + 1):
             if strike_b - strike_a > MAX_WING:
                 break
             for strike_c in strikes[j + 1:]:
                 if strike_c - strike_b > MAX_WING:
                     break
 
                 data_a = strike_map[strike_a]
                 data_b = strike_map[strike_b]
                 data_c = strike_map[strike_c]
 
                 lower_wing = strike_b - strike_a
                 upper_wing = strike_c - strike_b
 
                 # Call Butterfly — equal wings
                 if bias in (None, 'Neutral') and lower_wing == upper_wing:
                     if data_a["ce_premium"] > 0 and data_b["ce_premium"] > 0 and data_c["ce_premium"] > 0:
                         legs = build_call_butterfly(
                             strike_a, strike_b, strike_c,
                             data_a["ce_premium"], data_b["ce_premium"], data_c["ce_premium"]
                         )
                         bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                         if passes_filters(bench):
                             net_credit = (2 * data_b["ce_premium"] - data_a["ce_premium"] - data_c["ce_premium"]) * LOT_SIZE
                             append_result(
                                 "Call Butterfly (4L)",
                                 f"B {strike_a}CE | S {strike_b}CE | S {strike_b}CE | B {strike_c}CE",
                                 bench, net_credit,
                                 (data_a["ce_premium"] - 2 * data_b["ce_premium"] + data_c["ce_premium"]) / spot,
                                 "Neutral"
                             )
 
                 # Broken Wing Call Butterfly — upper wing wider
                 if bias in (None, 'Bearish') and upper_wing > lower_wing:
                     if data_a["ce_premium"] > 0 and data_b["ce_premium"] > 0 and data_c["ce_premium"] > 0:
                         legs = build_broken_wing_call_butterfly(
                             strike_a, strike_b, strike_c,
                             data_a["ce_premium"], data_b["ce_premium"], data_c["ce_premium"]
                         )
                         bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                         if passes_filters(bench):
                             net_credit = (2 * data_b["ce_premium"] - data_a["ce_premium"] - data_c["ce_premium"]) * LOT_SIZE
                             append_result(
                                 "Broken Wing Call Butterfly (4L)",
                                 f"B {strike_a}CE | S {strike_b}CE | S {strike_b}CE | B {strike_c}CE",
                                 bench, net_credit,
                                 (data_a["ce_premium"] - 2 * data_b["ce_premium"] + data_c["ce_premium"]) / spot,
                                 "Bearish"
                             )
 
                 # Put Butterfly — equal wings
                 if bias in (None, 'Neutral') and lower_wing == upper_wing:
                     if data_a["pe_premium"] > 0 and data_b["pe_premium"] > 0 and data_c["pe_premium"] > 0:
                         legs = build_put_butterfly(
                             strike_a, strike_b, strike_c,
                             data_a["pe_premium"], data_b["pe_premium"], data_c["pe_premium"]
                         )
                         bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                         if passes_filters(bench):
                             net_credit = (2 * data_b["pe_premium"] - data_a["pe_premium"] - data_c["pe_premium"]) * LOT_SIZE
                             append_result(
                                 "Put Butterfly (4L)",
                                 f"B {strike_a}PE | S {strike_b}PE | S {strike_b}PE | B {strike_c}PE",
                                 bench, net_credit,
                                 (data_a["pe_premium"] - 2 * data_b["pe_premium"] + data_c["pe_premium"]) / spot,
                                 "Neutral"
                             )
 
                 # Broken Wing Put Butterfly — lower wing wider
                 if bias in (None, 'Bullish') and lower_wing > upper_wing:
                     if data_a["pe_premium"] > 0 and data_b["pe_premium"] > 0 and data_c["pe_premium"] > 0:
                         legs = build_broken_wing_put_butterfly(
                             strike_a, strike_b, strike_c,
                             data_a["pe_premium"], data_b["pe_premium"], data_c["pe_premium"]
                         )
                         bench = get_payoff_benchmarks(get_linear_segments(legs), spot, atm_iv, tte_days)
                         if passes_filters(bench):
                             net_credit = (2 * data_b["pe_premium"] - data_a["pe_premium"] - data_c["pe_premium"]) * LOT_SIZE
                             append_result(
                                 "Broken Wing Put Butterfly (4L)",
                                 f"B {strike_a}PE | S {strike_b}PE | S {strike_b}PE | B {strike_c}PE",
                                 bench, net_credit,
                                 (data_a["pe_premium"] - 2 * data_b["pe_premium"] + data_c["pe_premium"]) / spot,
                                 "Bullish"
                             )

    results.sort(key=lambda x: float(x["rr"].split(": ")[1]), reverse=True)
    return results





# from scipy.stats import norm
# import math

# LOT_SIZE = 65

# # ── Linear Segment ──
# class LinearSegment:
#     def __init__(self, min_spot, max_spot, slope, intercept):
#         self.min_spot = min_spot
#         self.max_spot = max_spot
#         self.slope = slope
#         self.intercept = intercept

# # ── PayOff Benchmarks ──
# class PayOffBenchmarks:
#     def __init__(self):
#         self.max_profit = float('nan')
#         self.max_loss = float('nan')
#         self.risk_reward = float('nan')
#         self.pop = float('nan')
#         self.pol = float('nan')
#         self.pmp = float('nan')
#         self.pml = float('nan')
#         self.expected_value = float('nan')

# # ── Option representation ──
# class Option:
#     def __init__(self, strike, option_type, premium):
#         self.strike = strike
#         self.option_type = option_type  # "CE" or "PE"
#         self.premium = premium

# class OptionsWithQuantity:
#     def __init__(self, option, quantity):
#         self.option = option
#         self.quantity = quantity

# # ── Get payoff at a specific spot ──
# def get_payoff_at_spot(spot, segments):
#     for seg in segments:
#         if seg.min_spot <= spot <= seg.max_spot:
#             return seg.slope * spot + seg.intercept
#     last = segments[-1]
#     return last.slope * spot + last.intercept

# # ── Get finite vertices (strike prices) ──
# def get_finite_vertices(segments):
#     vertices = set()
#     for seg in segments:
#         if not math.isinf(seg.min_spot):
#             vertices.add(seg.min_spot)
#         if not math.isinf(seg.max_spot):
#             vertices.add(seg.max_spot)
#     return sorted(vertices)

# # ── Get flat intervals for a target value ──
# def get_flat_intervals_for_value(segments, target_value, use_log_normal=False):
#     tolerance = 1e-6
#     intervals = []
#     lower_clip = 1e-6 if use_log_normal else 0.0
#     ordered = sorted(segments, key=lambda s: s.min_spot)

#     for seg in ordered:
#         if abs(seg.slope) < 1e-9 and abs(seg.intercept - target_value) < tolerance:
#             min_val = max(seg.min_spot, lower_clip)
#             max_val = seg.max_spot
#             if min_val < max_val:
#                 intervals.append((min_val, max_val))

#     # left tail
#     if ordered:
#         left = ordered[0]
#         if abs(left.slope) < 1e-9 and abs(left.intercept - target_value) < tolerance:
#             left_min = max(float('-inf'), lower_clip)
#             left_max = left.min_spot
#             if left_min < left_max:
#                 intervals.insert(0, (left_min, left_max))

#     # right tail
#     if ordered:
#         right = ordered[-1]
#         if abs(right.slope) < 1e-9 and abs(right.intercept - target_value) < tolerance:
#             right_min = right.max_spot
#             right_max = float('inf')
#             if right_min < right_max:
#                 intervals.append((right_min, right_max))

#     return intervals

# # ── Get positive payoff intervals ──
# def get_positive_payoff_intervals(segments):
#     intervals = []
#     ordered = sorted(segments, key=lambda s: s.min_spot)

#     for seg in ordered:
#         a = seg.min_spot
#         b = seg.max_spot

#         if abs(seg.slope) < 1e-9:
#             if seg.intercept > 0:
#                 intervals.append((a, b))
#             continue

#         zero_cross = -seg.intercept / seg.slope

#         if seg.slope > 0:
#             pos_start = max(a, zero_cross)
#             pos_end = b
#         else:
#             pos_start = a
#             pos_end = min(b, zero_cross)

#         pos_start = max(pos_start, a)
#         pos_end = min(pos_end, b)

#         if pos_start < pos_end:
#             intervals.append((pos_start, pos_end))

#     return intervals

# # ── Normal probability over intervals ──
# def compute_normal_prob_over_intervals(intervals, mu, std):
#     prob = 0.0
#     for (a, b) in intervals:
#         cdf_a = norm.cdf(a, loc=mu, scale=std) if not math.isinf(a) else 0.0
#         cdf_b = norm.cdf(b, loc=mu, scale=std) if not math.isinf(b) else 1.0
#         prob += cdf_b - cdf_a
#     return prob * 100

# # ── GetLinearSegments ──
# def get_linear_segments(options_with_qty_list, total_pnl_hitherto=0):
#     segments = []
#     strikes = []
#     entry_cost = 0.0
#     call_pos_by_strike = {}
#     put_pos_by_strike = {}

#     for owq in options_with_qty_list:
#         opt = owq.option
#         qty = owq.quantity
#         entry = opt.premium
#         entry_cost += qty * entry

#         strike = opt.strike
#         if opt.option_type == "CE":
#             call_pos_by_strike[strike] = call_pos_by_strike.get(strike, 0) + qty
#         else:
#             put_pos_by_strike[strike] = put_pos_by_strike.get(strike, 0) + qty

#         if strike not in strikes:
#             strikes.append(strike)

#     strikes.sort()

#     if not strikes:
#         segments.append(LinearSegment(
#             min_spot=float('-inf'),
#             max_spot=float('inf'),
#             slope=0,
#             intercept=total_pnl_hitherto - entry_cost
#         ))
#         return segments

#     base_constant = total_pnl_hitherto - entry_cost
#     total_put_pos = sum(put_pos_by_strike.values())
#     current_slope = -1 * total_put_pos
#     current_constant = base_constant

#     for strike, qty in put_pos_by_strike.items():
#         current_constant += qty * strike

#     # first segment: -inf to first strike
#     segments.append(LinearSegment(
#         min_spot=float('-inf'),
#         max_spot=strikes[0],
#         slope=current_slope,
#         intercept=current_constant
#     ))

#     for i, K in enumerate(strikes):
#         call_pos_at_k = call_pos_by_strike.get(K, 0)
#         put_pos_at_k = put_pos_by_strike.get(K, 0)

#         delta_slope = call_pos_at_k + put_pos_at_k
#         delta_constant = -(call_pos_at_k * K) - (put_pos_at_k * K)

#         current_slope += delta_slope
#         current_constant += delta_constant

#         current_max = strikes[i + 1] if i < len(strikes) - 1 else float('inf')

#         segments.append(LinearSegment(
#             min_spot=K,
#             max_spot=current_max,
#             slope=current_slope,
#             intercept=current_constant
#         ))

#     return segments

# # ── GetPayOffBenchmarks ──
# def get_payoff_benchmarks(segments, spot_price, atm_iv, tte_days):
#     bench = PayOffBenchmarks()

#     ordered = sorted(segments, key=lambda s: s.min_spot)

#     left_slope = ordered[0].slope
#     right_slope = ordered[-1].slope

#     loss_unbounded = left_slope > 0 or right_slope < 0
#     profit_unbounded = left_slope < 0 or right_slope > 0

#     max_loss = float('-inf') if loss_unbounded else float('nan')
#     max_profit = float('inf') if profit_unbounded else float('nan')

#     vertices = get_finite_vertices(ordered)
#     if not vertices:
#         return bench

#     payoffs = [get_payoff_at_spot(v, ordered) for v in vertices]

#     if not loss_unbounded:
#         max_loss = min(payoffs)
#     if not profit_unbounded:
#         max_profit = max(payoffs)

#     if max_loss >= 0 or max_profit <= 0:
#         return bench

#     if math.isinf(max_loss):
#         risk_reward = 0
#     elif math.isinf(max_profit):
#         risk_reward = float('inf')
#     else:
#         risk_reward = abs(max_profit / max_loss)

#     bench.max_loss = max_loss
#     bench.max_profit = max_profit
#     bench.risk_reward = risk_reward

#     # on expiry day use spot to determine POP
#     if tte_days <= 0:
#         current_payoff = get_payoff_at_spot(spot_price, ordered)
#         bench.pop = 100.0 if current_payoff > 0 else 0.0
#         bench.pol = 100 - bench.pop
#         return bench

#     ltp = max(spot_price, 1e-6)
#     daily_vol = atm_iv / math.sqrt(365)
#     std_dev = daily_vol * math.sqrt(tte_days)
#     abs_std = ltp * std_dev

#     if abs_std <= 0:
#         return bench

#     # PMP — probability of max profit
#     if not math.isinf(max_profit):
#         max_profit_intervals = get_flat_intervals_for_value(ordered, max_profit, False)
#         bench.pmp = compute_normal_prob_over_intervals(max_profit_intervals, ltp, abs_std)

#     # PML — probability of max loss
#     if not math.isinf(max_loss):
#         max_loss_intervals = get_flat_intervals_for_value(ordered, max_loss, False)
#         bench.pml = compute_normal_prob_over_intervals(max_loss_intervals, ltp, abs_std)

#     # POP
#     positive_intervals = get_positive_payoff_intervals(ordered)
#     bench.pop = compute_normal_prob_over_intervals(positive_intervals, ltp, abs_std)
#     bench.pol = 100 - bench.pop

#     return bench

# # ── Build strategy legs ──
# def build_bull_call_spread(buy_strike, sell_strike, buy_ce_premium, sell_ce_premium):
#     buy_option  = Option(strike=buy_strike,  option_type="CE", premium=buy_ce_premium)
#     sell_option = Option(strike=sell_strike, option_type="CE", premium=sell_ce_premium)
#     return [
#         OptionsWithQuantity(buy_option,   1 * LOT_SIZE),
#         OptionsWithQuantity(sell_option, -1 * LOT_SIZE),
#     ]

# def build_bear_put_spread(sell_strike, buy_strike, sell_pe_premium, buy_pe_premium):
#     sell_option = Option(strike=sell_strike, option_type="PE", premium=sell_pe_premium)
#     buy_option  = Option(strike=buy_strike,  option_type="PE", premium=buy_pe_premium)
#     return [
#         OptionsWithQuantity(sell_option, -1 * LOT_SIZE),
#         OptionsWithQuantity(buy_option,   1 * LOT_SIZE),
#     ]

# # ── Main strategy generator ──
# def generate_strategies(options, spot, atm_iv, tte_days, min_rr=1.0, min_pop=40.0, max_loss=None, bias=None):
#     results = []
#     strike_map = {o["strike"]: o for o in options}
#     strikes = sorted(strike_map.keys())

#     for i, buy_strike in enumerate(strikes):
#         for sell_strike in strikes[i + 1:]:
#             buy_data  = strike_map[buy_strike]
#             sell_data = strike_map[sell_strike]

#             # ── Bull Call Spread ──
#             if bias in (None, 'Bullish'):
#                 if buy_data["ce_premium"] > 0 and sell_data["ce_premium"] > 0:
#                     legs = build_bull_call_spread(
#                         buy_strike=buy_strike,
#                         sell_strike=sell_strike,
#                         buy_ce_premium=buy_data["ce_premium"],
#                         sell_ce_premium=sell_data["ce_premium"],
#                     )
#                     segments = get_linear_segments(legs)
#                     bench = get_payoff_benchmarks(segments, spot, atm_iv, tte_days)

#                     if (not math.isnan(bench.risk_reward) and
#                         not math.isnan(bench.pop) and
#                         bench.risk_reward >= min_rr and
#                         bench.pop >= min_pop and
#                         (max_loss is None or abs(bench.max_loss) <= max_loss)):

#                         net_credit = (sell_data["ce_premium"] - buy_data["ce_premium"]) * LOT_SIZE
#                         results.append({
#                             "strategy": "Bull Call Spread (2L)",
#                             "legs": f"B {buy_strike}CE | S {sell_strike}CE",
#                             "rr": f"1 : {round(bench.risk_reward, 1)}",
#                             "pop": round(bench.pop, 2),
#                             "net_credit": round(net_credit, 2),
#                             "max_profit": round(bench.max_profit, 2),
#                             "max_loss": round(bench.max_loss, 2),
#                             "delta": round((buy_data["ce_premium"] - sell_data["ce_premium"]) / spot, 4),
#                             "bias": "Bullish",
#                         })

#             # ── Bear Put Spread ──
#             if bias in (None, 'Bearish'):
#                 if buy_data["pe_premium"] > 0 and sell_data["pe_premium"] > 0:
#                     legs = build_bear_put_spread(
#                         sell_strike=buy_strike,
#                         buy_strike=sell_strike,
#                         sell_pe_premium=buy_data["pe_premium"],
#                         buy_pe_premium=sell_data["pe_premium"],
#                     )
#                     segments = get_linear_segments(legs)
#                     bench = get_payoff_benchmarks(segments, spot, atm_iv, tte_days)

#                     if (not math.isnan(bench.risk_reward) and
#                         not math.isnan(bench.pop) and
#                         bench.risk_reward >= min_rr and
#                         bench.pop >= min_pop and
#                         (max_loss is None or abs(bench.max_loss) <= max_loss)):

#                         net_credit = (buy_data["pe_premium"] - sell_data["pe_premium"]) * LOT_SIZE
#                         results.append({
#                             "strategy": "Bear Put Spread (2L)",
#                             "legs": f"S {buy_strike}PE | B {sell_strike}PE",
#                             "rr": f"1 : {round(bench.risk_reward, 1)}",
#                             "pop": round(bench.pop, 2),
#                             "net_credit": round(net_credit, 2),
#                             "max_profit": round(bench.max_profit, 2),
#                             "max_loss": round(bench.max_loss, 2),
#                             "delta": round((sell_data["pe_premium"] - buy_data["pe_premium"]) / spot, 4),
#                             "bias": "Bearish",
#                         })

#     results.sort(key=lambda x: float(x["rr"].split(": ")[1]), reverse=True)
#     return results

