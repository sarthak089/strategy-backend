from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from nse import get_spot_and_premiums
from calculations import generate_strategies

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://strategy-finder.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/expiries")
def expiries(symbol: str = Query(default="NIFTY")):
    try:
        from jugaad_data.nse import NSELive
        n = NSELive()
        chain = n.index_option_chain(symbol)
        all_expiries = chain["records"]["expiryDates"]
        return {"success": True, "expiries": all_expiries}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/options-chain")
def options_chain(
    symbol: str = Query(default="NIFTY"),
    expiry: str = Query(default=None),
    otm_range: int = Query(default=20),
    itm_range: int = Query(default=2),
):
    try:
        data = get_spot_and_premiums(
            symbol=symbol,
            expiry=expiry,
            otm_range=otm_range,
            itm_range=itm_range,
        )
        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/strategies")
def strategies(
    symbol: str = Query(default="NIFTY"),
    expiry: str = Query(default=None),
    otm_range: int = Query(default=20),
    itm_range: int = Query(default=2),
    min_rr: float = Query(default=1.0),
    min_pop: float = Query(default=40.0),
    max_loss: float = Query(default=None),
    bias: str = Query(default=None),
    leg_count: str = Query(default=None),
):
    print(f">>> bias={bias}, max_loss={max_loss}, leg_count={leg_count}")
    try:
        data = get_spot_and_premiums(
            symbol=symbol,
            expiry=expiry,
            otm_range=otm_range,
            itm_range=itm_range,
        )

        results = generate_strategies(
            options=data["options"],
            spot=data["spot"],
            atm_iv=data["vix"],
            tte_days=data["tte_days"],
            min_rr=min_rr,
            min_pop=min_pop,
            max_loss=max_loss,
            bias=bias,
            leg_count=leg_count,
        )

        return {
            "success": True,
            "spot": data["spot"],
            "atm": data["atm"],
            "expiry": data["expiry"],
            "vix": data["vix"],
            "tte_days": round(data["tte_days"], 4),
            "total": len(results),
            "strategies": results,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}



            