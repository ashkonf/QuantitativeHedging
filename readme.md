# Quantitative Hedging

## Contents

- [Overview](#Overview)
- [Dependencies](#Dependencies)
- [Usage](#Usage)
- [Example](#Example)
- [License](#License)
- [Links](#Links)

## Overview

The Quantitative Hedging repository provides an easy way to hedge a stock using a basket of other stocks which collectively behave as a hedge against the desired stock. The repo is intended for two types of users: (1) market makers who need to offset the risk derived from undesired inventory and (2) quantitative researchers who need to identify factors or replicate studies involving the performance of a security, portfolio, or hedge fund. 

## Dependencies

Trading Baskets requires the following libraries:

- [`pandas`](https://pandas.pydata.org/)
- [`cvxopt`](https://cvxopt.org/)
- [`numpy`](https://numpy.org/)

Install these libraries using `pip` with requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

This repo exports one public functions (in hedge.py) `build_basket` which builds the basket of stocks intended to hedge a desired stock. 

### build_basket()

Use `build_basket()` to compose a hedging basket from (1) the stock to be hedged (`hedged_ticker_symbol`) and (2) the ticker symbols to consider including in the hedging basket (`basket_ticker_symbols`):

     basket(build_basket, basket_ticker_symbols)

The `basket()` arguments are as follows:

| Name              | Type                                        | Description  | Optional? | Sample Value |
|-------------------|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|------|
| `hedged_ticker_symbol` | `str` | The stock-ticker name of the stock to be hedged. | No    | `"AAPL"` |
| `basket_ticker_symbols`               | `list`                                       | A list of ticker symbols the library will consider including in the hedging basket. | No   | `["GOOG", "MSFT", "NFLX", "AMZN", "FB"]` |

## Example

The code below shows how to hedge a stock. The code defines APPL (Apple) as the stock to hedge, a list of stocks to consider using as part of the hedge (GOOG, MSFT, NFLX, AMZN, and FB), and composes a corresponding hedge basket for AAPL.

```bash
from hedge import build_basket

hedged_ticker_symbol = "AAPL"
basket_ticker_symbols = ["GOOG", "MSFT", "NFLX", "AMZN", "FB"]
print("Hedge for %s:" % hedged_ticker_symbol)
print(build_basket(hedged_ticker_symbol, basket_ticker_symbols))
```

This will produce the following hedging basket:

```
{'AAPL': 0.2614353523521262, 'FB': 0.1921680128468791, 'AMZN': 0.5463966348009947}
```

i.e. AAPL with weight 27%, FB with weight 19%, and AMZN with weight 54%.

A snippet like this can be incorporated in any Python application.

## License

Trading Baskets is licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Links

- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [CVXOPT](https://cvxopt.org/)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
