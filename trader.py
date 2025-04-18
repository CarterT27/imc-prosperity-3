from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
import json
import math
import statistics
import pandas as pd
import numpy as np
import jsonpickle
import typing


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        self.kelp_prices = []
        self.resin_prices = []
        self.squid_ink_prices = []
        self.croissants_prices = []
        self.jams_prices = []
        self.djembes_prices = []
        self.kelp_vwap = []
        self.resin_vwap = []
        self.squid_ink_vwap = []
        self.croissants_vwap = []
        self.jams_vwap = []
        self.djembes_vwap = []
        self.active_products = {
            "KELP": True,
            "RAINFOREST_RESIN": True,
            "SQUID_INK": True,
            "CROISSANTS": True,
            "JAMS": True,
            "DJEMBES": True,
            "PICNIC_BASKET1": True,
            "PICNIC_BASKET2": True,
            "VOLCANIC_ROCK": True,
        }
        self.position_limits = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "VOLCANIC_ROCK": 400,
        }
        self.timespan = 20
        self.make_width = {
            "KELP": 8.0,
            "RAINFOREST_RESIN": 3.0,
            "SQUID_INK": 5.0,
            "CROISSANTS": 1.0,
            "JAMS": 2.0,
            "DJEMBES": 2.0,
        }
        self.take_width = {
            "KELP": 1.0,
            "RAINFOREST_RESIN": 0.3,
            "SQUID_INK": 0.7,
            "CROISSANTS": 0.5,
            "JAMS": 0.5,
            "DJEMBES": 0.5,
        }
        self.squid_ink_volatility_threshold = 3.0
        self.squid_ink_momentum_period = 10
        self.squid_ink_mean_window = 30
        self.squid_ink_deviation_threshold = 0.05
        self.squid_ink_max_position_time = 5
        self.squid_ink_position_start_time = 0
        self.squid_ink_last_position = 0
        self.voucher_strikes = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500,
        }
        self.days_to_expiry = 7
        self.mean_volatility = 0.18
        self.volatility_window = 30
        self.zscore_threshold = 1.8
        self.past_volatilities = {}
        self.arbitrage_threshold = 0.01
        self.max_arbitrage_size = 50
        self.risk_free_rate = 0.0
        self.stop_loss_multiplier = 1.2
        self.profit_target_multiplier = 2.5
        self.max_stop_loss_hits = 1
        self.stop_loss_hits = 0
        self.positions = {}
        self.daily_pnl = 0
        self.current_day = 0
        self.max_daily_loss = 50000
        self.profit_target = 20000
        self.position_scale = 1.0
        self.max_volatility_history = 30
        self.cache = {}
        self.last_tick_time = 0

    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        try:
            if not order_depth.buy_orders or not order_depth.sell_orders:
                return None
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        except Exception as e:
            logger.print(f"Error calculating fair value: {e}")
            return None

    def clear_position_order(
        self,
        orders: list[Order],
        order_depth: OrderDepth,
        position: int,
        position_limit: int,
        product: str,
        buy_order_volume: int,
        sell_order_volume: int,
        fair_value: float,
        width: int,
    ) -> tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = int(round(fair_value))
        fair_for_bid = int(math.floor(fair_value))
        fair_for_ask = int(math.ceil(fair_value))
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)
        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(
                    order_depth.buy_orders[fair_for_ask], position_after_take
                )
                sent_quantity = min(sell_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                    sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(
                    abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take)
                )
                sent_quantity = min(buy_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                    buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def product_orders(
        self, product: str, order_depth: OrderDepth, position: int
    ) -> list[Order]:
        orders = []
        position_limit = self.position_limits[product]
        buy_order_volume = 0
        sell_order_volume = 0
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return orders
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_asks = [
            p
            for p in order_depth.sell_orders.keys()
            if abs(order_depth.sell_orders[p]) >= 15
        ]
        filtered_bids = [
            p
            for p in order_depth.buy_orders.keys()
            if abs(order_depth.buy_orders[p]) >= 15
        ]
        mm_ask = min(filtered_asks) if filtered_asks else best_ask
        mm_bid = max(filtered_bids) if filtered_bids else best_bid
        mm_mid_price = (mm_ask + mm_bid) / 2
        if product == "KELP":
            self.kelp_prices.append(mm_mid_price)
            if len(self.kelp_prices) > self.timespan:
                self.kelp_prices.pop(0)
            volume = (
                -1 * order_depth.sell_orders[best_ask]
                + order_depth.buy_orders[best_bid]
            )
            vwap = (
                best_bid * (-1) * order_depth.sell_orders[best_ask]
                + best_ask * order_depth.buy_orders[best_bid]
            ) / volume
            self.kelp_vwap.append({"vol": volume, "vwap": vwap})
            if len(self.kelp_vwap) > self.timespan:
                self.kelp_vwap.pop(0)
            if len(self.kelp_vwap) > 0:
                total_vol = sum(x["vol"] for x in self.kelp_vwap)
                fair_value = (
                    (sum(x["vwap"] * x["vol"] for x in self.kelp_vwap) / total_vol)
                    if total_vol > 0
                    else mm_mid_price
                )
            else:
                fair_value = mm_mid_price
        elif product == "RAINFOREST_RESIN":
            self.resin_prices.append(mm_mid_price)
            if len(self.resin_prices) > self.timespan:
                self.resin_prices.pop(0)
            volume = (
                -1 * order_depth.sell_orders[best_ask]
                + order_depth.buy_orders[best_bid]
            )
            vwap = (
                best_bid * (-1) * order_depth.sell_orders[best_ask]
                + best_ask * order_depth.buy_orders[best_bid]
            ) / volume
            self.resin_vwap.append({"vol": volume, "vwap": vwap})
            if len(self.resin_vwap) > self.timespan:
                self.resin_vwap.pop(0)
            if len(self.resin_vwap) > 0:
                total_vol = sum(x["vol"] for x in self.resin_vwap)
                fair_value = (
                    (sum(x["vwap"] * x["vol"] for x in self.resin_vwap) / total_vol)
                    if total_vol > 0
                    else mm_mid_price
                )
            else:
                fair_value = mm_mid_price
        else:
            fair_value = mm_mid_price
        if best_ask <= fair_value - self.take_width.get(product, 0):
            ask_amount = -1 * order_depth.sell_orders[best_ask]
            if ask_amount <= 20:
                quantity = min(ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
        if best_bid >= fair_value + self.take_width.get(product, 0):
            bid_amount = order_depth.buy_orders[best_bid]
            if bid_amount <= 20:
                quantity = min(bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders,
            order_depth,
            position,
            position_limit,
            product,
            buy_order_volume,
            sell_order_volume,
            fair_value,
            2,
        )
        asks_above_fair = [
            p for p in order_depth.sell_orders.keys() if p > fair_value + 1
        ]
        bids_below_fair = [
            p for p in order_depth.buy_orders.keys() if p < fair_value - 1
        ]
        best_ask_above_fair = (
            min(asks_above_fair) if asks_above_fair else int(fair_value) + 2
        )
        best_bid_below_fair = (
            max(bids_below_fair) if bids_below_fair else int(fair_value) - 2
        )
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            buy_price = int(best_bid_below_fair + 1)
            orders.append(Order(product, buy_price, buy_quantity))
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            sell_price = int(best_ask_above_fair - 1)
            orders.append(Order(product, sell_price, -sell_quantity))
        return orders

    def close_position(
        self, product: str, order_depth: OrderDepth, position: int
    ) -> list[Order]:
        orders = []
        if position == 0:
            return orders
        if position > 0:
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                sell_quantity = min(position, order_depth.buy_orders[best_bid])
                if sell_quantity > 0:
                    orders.append(Order(product, best_bid, -sell_quantity))
        else:
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                buy_quantity = min(-position, -order_depth.sell_orders[best_ask])
                if buy_quantity > 0:
                    orders.append(Order(product, best_ask, buy_quantity))
        return orders

    def squid_ink_strategy(
        self, order_depth: OrderDepth, position: int, state_timestamp: int
    ) -> list[Order]:
        orders = []
        position_limit = self.position_limits["SQUID_INK"]
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2
        self.squid_ink_prices.append(mid_price)
        if len(self.squid_ink_prices) > max(self.timespan, self.squid_ink_mean_window):
            self.squid_ink_prices.pop(0)
        if len(self.squid_ink_prices) < 10:
            return orders
        recent_window = min(len(self.squid_ink_prices), self.squid_ink_mean_window)
        mean_price = sum(self.squid_ink_prices[-recent_window:]) / recent_window
        if len(self.squid_ink_prices) >= 2:
            volatility = statistics.stdev(
                self.squid_ink_prices[-min(10, len(self.squid_ink_prices)) :]
            )
        else:
            volatility = 0
        deviation_pct = (
            abs(mid_price - mean_price) / mean_price if mean_price > 0 else 0
        )
        if position != 0 and self.squid_ink_position_start_time > 0:
            time_in_position = state_timestamp - self.squid_ink_position_start_time
            if time_in_position >= self.squid_ink_max_position_time:
                return self.close_position("SQUID_INK", order_depth, position)
        if position == 0:
            if (
                volatility > self.squid_ink_volatility_threshold
                and deviation_pct > self.squid_ink_deviation_threshold
            ):
                if mid_price > mean_price:
                    quantity = min(order_depth.buy_orders[best_bid], position_limit)
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", best_bid, -quantity))
                        self.squid_ink_position_start_time = state_timestamp
                        self.squid_ink_last_position = -quantity
                elif mid_price < mean_price:
                    quantity = min(-order_depth.sell_orders[best_ask], position_limit)
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", best_ask, quantity))
                        self.squid_ink_position_start_time = state_timestamp
                        self.squid_ink_last_position = quantity
        else:
            if position > 0 and mid_price >= mean_price:
                quantity = min(position, order_depth.buy_orders[best_bid])
                if quantity > 0:
                    orders.append(Order("SQUID_INK", best_bid, -quantity))
                    if quantity == position:
                        self.squid_ink_position_start_time = 0
                        self.squid_ink_last_position = 0
            elif position < 0 and mid_price <= mean_price:
                quantity = min(-position, -order_depth.sell_orders[best_ask])
                if quantity > 0:
                    orders.append(Order("SQUID_INK", best_ask, quantity))
                    if quantity == -position:
                        self.squid_ink_position_start_time = 0
                        self.squid_ink_last_position = 0
        return orders

    def calculate_synthetic_value(self, state: TradingState, basket_type: str) -> float:
        if basket_type == "PICNIC_BASKET1":
            croissant_price = self.calculate_fair_value(
                state.order_depths["CROISSANTS"]
            )
            jams_price = self.calculate_fair_value(state.order_depths["JAMS"])
            djembes_price = self.calculate_fair_value(state.order_depths["DJEMBES"])
            if None in [croissant_price, jams_price, djembes_price]:
                return None
            return 6 * croissant_price + 3 * jams_price + 1 * djembes_price
        elif basket_type == "PICNIC_BASKET2":
            croissant_price = self.calculate_fair_value(
                state.order_depths["CROISSANTS"]
            )
            jams_price = self.calculate_fair_value(state.order_depths["JAMS"])
            if None in [croissant_price, jams_price]:
                return None
            return 4 * croissant_price + 2 * jams_price
        return None

    def trade_basket_divergence(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        synthetic_value: float,
    ) -> list[Order]:
        orders = []
        position_limit = self.position_limits[product]
        if synthetic_value is None:
            return orders
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            current_price = (best_ask + best_bid) / 2
            divergence = synthetic_value - current_price
            if product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
                if product == "PICNIC_BASKET1":
                    buy_threshold = 5.0
                    sell_threshold = -5.0
                    make_width = 1.0
                else:
                    buy_threshold = 7.5
                    sell_threshold = -7.5
                    make_width = 1.0
                if divergence > buy_threshold:
                    ask_amount = -1 * order_depth.sell_orders[best_ask]
                    quantity = min(ask_amount, (position_limit - position) // 2)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                elif divergence < sell_threshold:
                    bid_amount = order_depth.buy_orders[best_bid]
                    quantity = min(bid_amount, (position_limit + position) // 2)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                if abs(divergence) > 1.0:
                    buy_price = int(current_price - make_width)
                    buy_quantity = (position_limit - position) // 2
                    if buy_quantity > 0:
                        orders.append(Order(product, buy_price, buy_quantity))
                    sell_price = int(current_price + make_width)
                    sell_quantity = (position_limit + position) // 2
                    if sell_quantity > 0:
                        orders.append(Order(product, sell_price, -sell_quantity))
            else:
                if divergence > 10.0:
                    ask_amount = -1 * order_depth.sell_orders[best_ask]
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                elif divergence < -10.0:
                    bid_amount = order_depth.buy_orders[best_bid]
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                make_width = 2.0
                if abs(divergence) > 2.0:
                    make_width *= 1.5
                buy_price = int(current_price - make_width)
                buy_quantity = position_limit - position
                if buy_quantity > 0:
                    orders.append(Order(product, buy_price, buy_quantity))
                sell_price = int(current_price + make_width)
                sell_quantity = position_limit + position
                if sell_quantity > 0:
                    orders.append(Order(product, sell_price, -sell_quantity))
        return orders

    def execute_basket_arbitrage(
        self, state: TradingState, basket_type: str
    ) -> dict[str, list[Order]]:
        result = {}
        basket_order_depth = state.order_depths[basket_type]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(
            state, basket_type
        )
        if basket_type == "PICNIC_BASKET1":
            components = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
        else:
            components = {"CROISSANTS": 4, "JAMS": 2}
        if basket_order_depth.sell_orders and synthetic_order_depth.buy_orders:
            basket_ask = min(basket_order_depth.sell_orders.keys())
            synthetic_bid = max(synthetic_order_depth.buy_orders.keys())
            if basket_ask < synthetic_bid:
                basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask])
                synthetic_bid_volume = synthetic_order_depth.buy_orders[synthetic_bid]
                arb_volume = min(basket_ask_volume, synthetic_bid_volume)
                basket_position = state.position.get(basket_type, 0)
                arb_volume = min(
                    arb_volume, self.position_limits[basket_type] - basket_position
                )
                if arb_volume > 0:
                    result.setdefault(basket_type, []).append(
                        Order(basket_type, basket_ask, arb_volume)
                    )
                    for p, w in components.items():
                        result.setdefault(p, [])
                        if p in state.order_depths and state.order_depths[p].buy_orders:
                            best_bid = max(state.order_depths[p].buy_orders.keys())
                            result[p].append(Order(p, best_bid, -w * arb_volume))
        if basket_order_depth.buy_orders and synthetic_order_depth.sell_orders:
            basket_bid = max(basket_order_depth.buy_orders.keys())
            synthetic_ask = min(synthetic_order_depth.sell_orders.keys())
            if basket_bid > synthetic_ask:
                basket_bid_volume = basket_order_depth.buy_orders[basket_bid]
                synthetic_ask_volume = abs(
                    synthetic_order_depth.sell_orders[synthetic_ask]
                )
                arb_volume = min(basket_bid_volume, synthetic_ask_volume)
                basket_position = state.position.get(basket_type, 0)
                arb_volume = min(
                    arb_volume, self.position_limits[basket_type] + basket_position
                )
                if arb_volume > 0:
                    result.setdefault(basket_type, []).append(
                        Order(basket_type, basket_bid, -arb_volume)
                    )
                    for p, w in components.items():
                        result.setdefault(p, [])
                        if (
                            p in state.order_depths
                            and state.order_depths[p].sell_orders
                        ):
                            best_ask = min(state.order_depths[p].sell_orders.keys())
                            result[p].append(Order(p, best_ask, w * arb_volume))
        return result

    def norm_cdf(self, x: float) -> float:
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        sign = 1
        if x < 0:
            sign = -1
        x = abs(x) / math.sqrt(2.0)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(
            -x * x
        )
        return 0.5 * (1.0 + sign * y)

    def norm_pdf(self, x: float) -> float:
        return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

    def black_scholes_call(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        cache_key = f"bs_call_{S}_{K}_{T}_{r}_{sigma}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            if S <= 0 or K <= 0 or T <= 0:
                return 0.0
            d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            result = S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
            self.cache[cache_key] = result
            return result
        except Exception as e:
            logger.print(f"Error in black_scholes_call: {e}")
            return 0.0

    def black_scholes_delta(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        if S <= 0 or K <= 0 or T <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
        return self.norm_cdf(d1)

    def black_scholes_vega(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        if S <= 0 or K <= 0 or T <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
        return S * math.sqrt(T) * self.norm_pdf(d1)

    def implied_volatility(
        self, option_price: float, S: float, K: float, T: float, r: float
    ) -> float:
        cache_key = f"iv_{option_price}_{S}_{K}_{T}_{r}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            if option_price <= 0 or S <= 0 or K <= 0 or T <= 0:
                return self.mean_volatility
            sigma = 0.5
            for _ in range(50):
                price = self.black_scholes_call(S, K, T, r, sigma)
                vega = self.black_scholes_vega(S, K, T, r, sigma)
                if vega == 0:
                    return self.mean_volatility
                diff = option_price - price
                if abs(diff) < 1e-5:
                    break
                sigma = sigma + diff / vega
                sigma = max(0.01, min(sigma, 2.0))
            self.cache[cache_key] = sigma
            return sigma
        except Exception as e:
            logger.print(f"Error in implied_volatility: {e}")
            return self.mean_volatility

    def calculate_premium(
        self, voucher_mid: float, rock_mid: float, strike: float
    ) -> float:
        return voucher_mid

    def should_stop_loss(self, voucher_symbol: str, current_price: float) -> bool:
        if voucher_symbol not in self.positions:
            return False
        entry_price = self.positions[voucher_symbol]["price"]
        position = self.positions[voucher_symbol]["position"]
        premium = self.positions[voucher_symbol]["premium"]
        if position > 0:
            loss = entry_price - current_price
            return loss > premium * self.stop_loss_multiplier
        else:
            loss = current_price - entry_price
            return loss > premium * self.stop_loss_multiplier

    def should_take_profit(self, voucher_symbol: str, current_price: float) -> bool:
        if voucher_symbol not in self.positions:
            return False
        entry_price = self.positions[voucher_symbol]["price"]
        position = self.positions[voucher_symbol]["position"]
        premium = self.positions[voucher_symbol]["premium"]
        if position > 0:
            profit = current_price - entry_price
            return profit > premium * self.profit_target_multiplier
        else:
            profit = entry_price - current_price
            return profit > premium * self.profit_target_multiplier

    def find_arbitrage_opportunities(
        self, state: TradingState, rock_order_depth: OrderDepth, rock_mid: float
    ) -> list[Order]:
        orders = []
        tte = self.get_time_to_expiry(state.timestamp)
        voucher_prices = {}
        for voucher_symbol in self.voucher_strikes.keys():
            if voucher_symbol in state.order_depths:
                voucher_mid = self.calculate_fair_value(
                    state.order_depths[voucher_symbol]
                )
                if voucher_mid is not None:
                    voucher_prices[voucher_symbol] = voucher_mid
        for i in range(len(self.voucher_strikes)):
            for j in range(i + 1, len(self.voucher_strikes)):
                strike1 = list(self.voucher_strikes.values())[i]
                strike2 = list(self.voucher_strikes.values())[j]
                symbol1 = list(self.voucher_strikes.keys())[i]
                symbol2 = list(self.voucher_strikes.keys())[j]
                if symbol1 in voucher_prices and symbol2 in voucher_prices:
                    price1 = voucher_prices[symbol1]
                    price2 = voucher_prices[symbol2]
                    spread = abs(price1 - price2)
                    strike_diff = abs(strike1 - strike2)
                    if (
                        abs(spread - strike_diff)
                        > self.arbitrage_threshold * strike_diff
                    ):
                        if spread > strike_diff * (1 + self.arbitrage_threshold):
                            if price1 > price2:
                                orders.append(
                                    Order(
                                        symbol1,
                                        int(
                                            min(
                                                state.order_depths[
                                                    symbol1
                                                ].sell_orders.keys()
                                            )
                                        ),
                                        -self.max_arbitrage_size,
                                    )
                                )
                                orders.append(
                                    Order(
                                        symbol2,
                                        int(
                                            max(
                                                state.order_depths[
                                                    symbol2
                                                ].buy_orders.keys()
                                            )
                                        ),
                                        self.max_arbitrage_size,
                                    )
                                )
                            else:
                                orders.append(
                                    Order(
                                        symbol2,
                                        int(
                                            min(
                                                state.order_depths[
                                                    symbol2
                                                ].sell_orders.keys()
                                            )
                                        ),
                                        -self.max_arbitrage_size,
                                    )
                                )
                                orders.append(
                                    Order(
                                        symbol1,
                                        int(
                                            max(
                                                state.order_depths[
                                                    symbol1
                                                ].buy_orders.keys()
                                            )
                                        ),
                                        self.max_arbitrage_size,
                                    )
                                )
        return orders

    def volcanic_rock_voucher_orders(
        self,
        state: TradingState,
        rock_order_depth: OrderDepth,
        rock_position: int,
        voucher_symbol: str,
        voucher_order_depth: OrderDepth,
        voucher_position: int,
        trader_data: dict,
    ) -> tuple[list[Order], list[Order]]:
        try:
            rock_mid = self.calculate_fair_value(rock_order_depth)
            voucher_mid = self.calculate_fair_value(voucher_order_depth)
            if rock_mid is None or voucher_mid is None:
                return [], []

            tte = self.get_time_to_expiry(state.timestamp)
            strike = self.voucher_strikes[voucher_symbol]

            # Calculate implied volatility for this specific voucher
            current_implied_vol = self.implied_volatility(
                voucher_mid, rock_mid, strike, tte, self.risk_free_rate
            )

            # Initialize or update volatility history for this strike
            if voucher_symbol not in self.past_volatilities:
                self.past_volatilities[voucher_symbol] = []

            self.past_volatilities[voucher_symbol].append(current_implied_vol)

            # Keep only the last 20 volatility readings
            if len(self.past_volatilities[voucher_symbol]) > self.volatility_window:
                self.past_volatilities[voucher_symbol].pop(0)

            # Use the mean of recent volatilities if available, otherwise use current
            if len(self.past_volatilities[voucher_symbol]) > 0:
                volatility = statistics.mean(self.past_volatilities[voucher_symbol])
            else:
                volatility = current_implied_vol

            # Calculate theoretical price using the rolling window volatility
            theoretical_price = self.black_scholes_call(
                rock_mid, strike, tte, self.risk_free_rate, volatility
            )

            make_orders = []
            if voucher_position < self.position_limits[voucher_symbol]:
                buy_price = int(theoretical_price)
                make_orders.append(
                    Order(
                        voucher_symbol,
                        buy_price,
                        self.position_limits[voucher_symbol] - voucher_position,
                    )
                )

            if voucher_position > -self.position_limits[voucher_symbol]:
                sell_price = int(theoretical_price + 1)
                make_orders.append(
                    Order(
                        voucher_symbol,
                        sell_price,
                        -self.position_limits[voucher_symbol] - voucher_position,
                    )
                )

            return [], make_orders
        except Exception as e:
            logger.print(f"Error in volcanic_rock_voucher_orders: {e}")
            return [], []

    def volcanic_rock_hedge_orders(
        self,
        rock_order_depth: OrderDepth,
        rock_position: int,
        voucher_positions: dict[str, int],
        deltas: dict[str, float],
    ) -> list[Order]:
        total_delta = 0.0
        for voucher_symbol, pos in voucher_positions.items():
            if voucher_symbol in deltas:
                total_delta += deltas[voucher_symbol] * pos
        target_rock_position = -int(total_delta)
        current_rock_position = rock_position
        if target_rock_position == current_rock_position:
            return []
        orders = []
        quantity = target_rock_position - current_rock_position
        if quantity > 0:
            best_ask = min(rock_order_depth.sell_orders.keys())
            orders.append(Order("VOLCANIC_ROCK", best_ask, quantity))
        else:
            best_bid = max(rock_order_depth.buy_orders.keys())
            orders.append(Order("VOLCANIC_ROCK", best_bid, quantity))
        return orders

    def calculate_synthetic_position(
        self,
        rock_price: float,
        call_price: float,
        put_price: float,
        strike: float,
        tte: float,
    ) -> tuple[float, float]:
        synthetic_price = call_price - put_price
        fair_price = rock_price - strike * math.exp(-self.risk_free_rate * tte)
        return synthetic_price, fair_price

    def volcanic_rock_orders(
        self, rock_order_depth: OrderDepth, rock_position: int, state: TradingState
    ) -> list[Order]:
        orders = []
        rock_mid = self.calculate_fair_value(rock_order_depth)
        if rock_mid is None:
            return orders

        # Calculate average of rolling window volatilities for each strike
        rolling_vols = []
        tte = self.get_time_to_expiry(state.timestamp)

        for voucher_symbol in self.voucher_strikes.keys():
            if (
                voucher_symbol in self.past_volatilities
                and len(self.past_volatilities[voucher_symbol]) > 0
            ):
                # Use the mean of the rolling window for this strike
                rolling_vol = statistics.mean(self.past_volatilities[voucher_symbol])
                rolling_vols.append(rolling_vol)
            elif voucher_symbol in state.order_depths:
                # If no history, calculate current IV
                voucher_order_depth = state.order_depths[voucher_symbol]
                voucher_mid = self.calculate_fair_value(voucher_order_depth)

                if voucher_mid is not None:
                    strike = self.voucher_strikes[voucher_symbol]
                    current_vol = self.implied_volatility(
                        voucher_mid, rock_mid, strike, tte, self.risk_free_rate
                    )
                    rolling_vols.append(current_vol)

        # Use average of rolling window volatilities, or fallback to mean_volatility if none available
        vol = statistics.mean(rolling_vols) if rolling_vols else self.mean_volatility

        avg_strike = sum(self.voucher_strikes.values()) / len(self.voucher_strikes)
        theoretical_price = self.black_scholes_call(
            rock_mid, avg_strike, tte, self.risk_free_rate, vol
        )

        threshold = 0.5
        position_limit = self.position_limits["VOLCANIC_ROCK"]

        if rock_mid < theoretical_price - threshold:
            if len(rock_order_depth.sell_orders) > 0:
                best_ask = min(rock_order_depth.sell_orders.keys())
                quantity = min(
                    position_limit - rock_position,
                    -rock_order_depth.sell_orders[best_ask],
                )
                if quantity > 0:
                    orders.append(Order("VOLCANIC_ROCK", best_ask, quantity))
        elif rock_mid > theoretical_price + threshold:
            if len(rock_order_depth.buy_orders) > 0:
                best_bid = max(rock_order_depth.buy_orders.keys())
                quantity = min(
                    position_limit + rock_position,
                    rock_order_depth.buy_orders[best_bid],
                )
                if quantity > 0:
                    orders.append(Order("VOLCANIC_ROCK", best_bid, -quantity))

        return orders

    def get_synthetic_basket_order_depth(
        self, state: TradingState, basket_type: str
    ) -> OrderDepth:
        synthetic_order_depth = OrderDepth()
        if basket_type == "PICNIC_BASKET1":
            components = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
        else:
            components = {"CROISSANTS": 4, "JAMS": 2}
        component_bids = {}
        component_asks = {}
        for product, weight in components.items():
            if product in state.order_depths and state.order_depths[product].buy_orders:
                component_bids[product] = max(
                    state.order_depths[product].buy_orders.keys()
                )
            else:
                component_bids[product] = 0
            if (
                product in state.order_depths
                and state.order_depths[product].sell_orders
            ):
                component_asks[product] = min(
                    state.order_depths[product].sell_orders.keys()
                )
            else:
                component_asks[product] = float("inf")
        implied_bid = sum(component_bids[p] * w for p, w in components.items())
        implied_ask = sum(component_asks[p] * w for p, w in components.items())
        if implied_bid > 0:
            bid_volumes = []
            for p, w in components.items():
                if p in state.order_depths and component_bids[p] > 0:
                    volume = state.order_depths[p].buy_orders[component_bids[p]] // w
                    bid_volumes.append(volume)
                else:
                    bid_volumes.append(0)
            implied_bid_volume = min(bid_volumes) if bid_volumes else 0
            synthetic_order_depth.buy_orders[implied_bid] = implied_bid_volume
        if implied_ask < float("inf"):
            ask_volumes = []
            for p, w in components.items():
                if p in state.order_depths and state.order_depths[p].sell_orders:
                    volume = (
                        abs(state.order_depths[p].sell_orders[component_asks[p]]) // w
                    )
                    ask_volumes.append(volume)
                else:
                    ask_volumes.append(0)
            implied_ask_volume = min(ask_volumes) if ask_volumes else 0
            synthetic_order_depth.sell_orders[implied_ask] = -implied_ask_volume
        return synthetic_order_depth

    def get_time_to_expiry(self, timestamp):
        current_day = timestamp // 1000000
        days_remaining = max(0, 6 - current_day)  # Assuming 7-day expiry from day 0
        return days_remaining / 365.0

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        try:
            result = {}
            conversions = 0
            trader_data = {}
            
            current_day = state.timestamp // 1000000
            if current_day != self.current_day:
                self.daily_pnl = 0
                self.current_day = current_day
            
            # Update days_to_expiry based on current timestamp
            days_remaining = max(0, 7 - current_day)
            self.days_to_expiry = days_remaining
            
            if state.traderData and state.traderData != "SAMPLE":
                try:
                    trader_data = jsonpickle.decode(state.traderData)
                    if "past_volatilities" in trader_data:
                        self.past_volatilities = trader_data["past_volatilities"]
                    for prod in ["kelp", "resin", "squid_ink", "croissants", "jams", "djembes"]:
                        if f"{prod}_prices" in trader_data:
                            setattr(self, f"{prod}_prices", trader_data[f"{prod}_prices"])
                        if f"{prod}_vwap" in trader_data:
                            setattr(self, f"{prod}_vwap", trader_data[f"{prod}_vwap"])
                except Exception as e:
                    logger.print(f"Could not parse trader data: {e}")

            handled = set()
            # Handle vouchers and VOLCANIC_ROCK
            if "VOLCANIC_ROCK" in state.order_depths:
                rock_position = state.position.get("VOLCANIC_ROCK", 0)
                rock_order_depth = state.order_depths["VOLCANIC_ROCK"]
                rock_mid = self.calculate_fair_value(rock_order_depth)

                # Always process vouchers
                if rock_mid is not None:
                    voucher_positions = {}
                    voucher_deltas = {}
                    for voucher_symbol in self.voucher_strikes.keys():
                        if voucher_symbol in state.order_depths:
                            try:
                                voucher_position = state.position.get(voucher_symbol, 0)
                                voucher_positions[voucher_symbol] = voucher_position
                                take_orders, make_orders = self.volcanic_rock_voucher_orders(
                                    state,
                                    rock_order_depth,
                                    rock_position,
                                    voucher_symbol,
                                    state.order_depths[voucher_symbol],
                                    voucher_position,
                                    trader_data
                                )
                                if take_orders or make_orders:
                                    result.setdefault(voucher_symbol, []).extend(take_orders + make_orders)
                            except Exception as e:
                                logger.print(f"Error processing voucher {voucher_symbol}: {e}")

                # Only trade VOLCANIC_ROCK if it's active
                if self.active_products.get("VOLCANIC_ROCK", False):
                    if rock_mid is not None:
                        arbitrage_orders = self.find_arbitrage_opportunities(state, rock_order_depth, rock_mid)
                        if arbitrage_orders:
                            for order in arbitrage_orders:
                                result.setdefault(order.symbol, []).append(order)
                    
                    if voucher_deltas:
                        try:
                            hedge_orders = self.volcanic_rock_hedge_orders(
                                rock_order_depth,
                                rock_position,
                                voucher_positions,
                                voucher_deltas
                            )
                            if hedge_orders:
                                result["VOLCANIC_ROCK"] = hedge_orders
                        except Exception as e:
                            logger.print(f"Error generating hedge orders: {e}")
                    
                    vol_orders = self.volcanic_rock_orders(rock_order_depth, rock_position, state)
                    if vol_orders:
                        result.setdefault("VOLCANIC_ROCK", []).extend(vol_orders)
                # If VOLCANIC_ROCK is not active but we have a position, close it
                elif rock_position != 0:
                    orders = self.close_position("VOLCANIC_ROCK", rock_order_depth, rock_position)
                    if orders:
                        result["VOLCANIC_ROCK"] = orders

                handled.add("VOLCANIC_ROCK")
                handled.update(self.voucher_strikes.keys())

            # Handle other products
            for product in state.order_depths.keys():
                if product in handled:
                    continue
                if product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
                    arbitrage_orders = self.execute_basket_arbitrage(state, product)
                    if not arbitrage_orders or product not in arbitrage_orders:
                        synthetic_value = self.calculate_synthetic_value(state, product)
                        position = state.position.get(product, 0)
                        orders = self.trade_basket_divergence(product, state.order_depths[product], position, synthetic_value)
                        if orders:
                            result[product] = orders
                    else:
                        for p, orders in arbitrage_orders.items():
                            result.setdefault(p, []).extend(orders)
                elif product in self.active_products and self.active_products[product]:
                    position = state.position.get(product, 0)
                    orders = self.product_orders(product, state.order_depths[product], position)
                    if orders:
                        result[product] = orders
                else:
                    position = state.position.get(product, 0)
                    if position != 0:
                        orders = self.close_position(product, state.order_depths[product], position)
                        if orders:
                            result[product] = orders

            trader_data["past_volatilities"] = self.past_volatilities
            serialized_trader_data = jsonpickle.encode(trader_data)
            if len(self.cache) > 1000:
                self.cache.clear()
            logger.flush(state, result, conversions, serialized_trader_data)
            return result, conversions, serialized_trader_data
        except Exception as e:
            logger.print(f"Error in run method: {e}")
            return {}, 0, "{}"
