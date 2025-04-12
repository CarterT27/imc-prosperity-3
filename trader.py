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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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
        # Maintain price history for calculating fair values
        self.kelp_prices = []
        self.resin_prices = []
        self.squid_ink_prices = []
        self.kelp_vwap = []
        self.resin_vwap = []
        self.squid_ink_vwap = []
        
        # Trading toggles for each product
        self.active_products = {
            "KELP": False,
            "RAINFOREST_RESIN": True,
            "SQUID_INK": False
        }
        
        # Position limits for each product - more conservative for volatile products
        self.position_limits = {
            "KELP": 50,  # Reduced due to high volatility
            "RAINFOREST_RESIN": 50,  # Stable product, can handle larger positions
            "SQUID_INK": 50  # Medium position limit due to predictable patterns
        }
        
        # Parameters for trading strategies
        self.timespan = 20  # Increased history for better pattern detection
        self.make_width = {
            "KELP": 8.0,  # Wider spread due to volatility
            "RAINFOREST_RESIN": 3.0,  # Tighter spread due to stability
            "SQUID_INK": 5.0  # Medium spread
        }
        self.take_width = {
            "KELP": 1.0,  # More aggressive for volatile market
            "RAINFOREST_RESIN": 0.3,  # Conservative taking strategy
            "SQUID_INK": 0.7  # Balanced approach
        }
        
        # SQUID_INK specific parameters
        self.squid_ink_volatility_threshold = 2.0
        self.squid_ink_momentum_period = 10  # Increased for better pattern detection
        self.squid_ink_pattern_length = 20  # New parameter for pattern detection

    def calculate_fair_value(
        self, order_depth: OrderDepth, method="mid_price", min_vol=0
    ) -> float:
        """Calculate fair value of a product based on order book."""
        if method == "mid_price":
            if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
                return None
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        elif method == "mid_price_with_vol_filter":
            filtered_asks = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= min_vol
            ]
            filtered_bids = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= min_vol
            ]

            if len(filtered_asks) == 0 or len(filtered_bids) == 0:
                return self.calculate_fair_value(order_depth, "mid_price")

            best_filtered_ask = min(filtered_asks)
            best_filtered_bid = max(filtered_bids)
            return (best_filtered_ask + best_filtered_bid) / 2

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
        """Add orders to clear position when close to fair value."""
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = int(round(fair_value))
        fair_for_bid = int(math.floor(fair_value))
        fair_for_ask = int(math.ceil(fair_value))

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        # If we're long, try to sell at fair price
        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(
                    order_depth.buy_orders[fair_for_ask], position_after_take
                )
                sent_quantity = min(sell_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                    sell_order_volume += abs(sent_quantity)

        # If we're short, try to buy at fair price
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
        """Generate orders for a specific product."""
        orders = []
        position_limit = self.position_limits[product]

        buy_order_volume = 0
        sell_order_volume = 0

        # Skip if there are no orders in the market
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return orders

        # Calculate fair value based on current market
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        # Find market maker orders (orders with volume >= 15)
        filtered_asks = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
        filtered_bids = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]

        mm_ask = min(filtered_asks) if filtered_asks else best_ask
        mm_bid = max(filtered_bids) if filtered_bids else best_bid
        mm_mid_price = (mm_ask + mm_bid) / 2

        if product == "KELP":
            # More dynamic strategy for volatile KELP
            self.kelp_prices.append(mm_mid_price)
            if len(self.kelp_prices) > self.timespan:
                self.kelp_prices.pop(0)

            # Calculate short-term and long-term VWAP for trend detection
            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume

            self.kelp_vwap.append({"vol": volume, "vwap": vwap})
            if len(self.kelp_vwap) > self.timespan:
                self.kelp_vwap.pop(0)

            # Use recent VWAP for fair value calculation
            if len(self.kelp_vwap) > 0:
                recent_vwaps = self.kelp_vwap[-5:]  # Focus on recent prices
                total_vol = sum([x["vol"] for x in recent_vwaps])
                if total_vol > 0:
                    fair_value = sum([x["vwap"] * x["vol"] for x in recent_vwaps]) / total_vol
                else:
                    fair_value = mm_mid_price
            else:
                fair_value = mm_mid_price

            take_width = self.take_width["KELP"]
            make_width = self.make_width["KELP"]

        elif product == "RAINFOREST_RESIN":
            # Stable product - use fixed fair value with tight spreads
            fair_value = 10000
            take_width = self.take_width["RAINFOREST_RESIN"]
            make_width = self.make_width["RAINFOREST_RESIN"]

            # Only take orders if significantly off fair value
            if abs(mm_mid_price - fair_value) > 5:
                take_width = take_width * 2  # More aggressive taking when price deviates significantly

        elif product == "SQUID_INK":
            self.squid_ink_prices.append(mm_mid_price)
            if len(self.squid_ink_prices) > self.timespan:
                self.squid_ink_prices.pop(0)

            # Enhanced pattern detection
            pattern_detected = False
            if len(self.squid_ink_prices) >= self.squid_ink_pattern_length:
                # Look for repeating patterns in price movements
                recent_prices = self.squid_ink_prices[-self.squid_ink_pattern_length:]
                price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
                
                # Simple pattern detection: look for alternating positive/negative changes
                alternating = True
                for i in range(1, len(price_changes)):
                    if (price_changes[i] > 0) == (price_changes[i-1] > 0):
                        alternating = False
                        break
                pattern_detected = alternating

            # Calculate momentum and volatility
            momentum = 0
            if len(self.squid_ink_prices) >= self.squid_ink_momentum_period:
                momentum = (self.squid_ink_prices[-1] - self.squid_ink_prices[-self.squid_ink_momentum_period]) / self.squid_ink_momentum_period

            volatility = 0
            if len(self.squid_ink_prices) >= 2:
                volatility = statistics.stdev(self.squid_ink_prices[-min(10, len(self.squid_ink_prices)):])

            # Calculate fair value using pattern information
            fair_value = mm_mid_price
            if pattern_detected:
                # If pattern detected, predict next move
                last_change = self.squid_ink_prices[-1] - self.squid_ink_prices[-2]
                fair_value = mm_mid_price - last_change  # Predict reversal

            # Adjust fair value based on momentum
            fair_value += momentum * (0.5 if not pattern_detected else 1.0)

            take_width = self.take_width["SQUID_INK"]
            make_width = self.make_width["SQUID_INK"]
            
            if volatility > self.squid_ink_volatility_threshold:
                take_width *= 1.5
                make_width *= 1.5

        # Taking strategy: take favorable orders
        if best_ask <= fair_value - take_width:
            ask_amount = -1 * order_depth.sell_orders[best_ask]
            if ask_amount <= 20:  # Only take small orders to avoid manipulation
                quantity = min(ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity

        if best_bid >= fair_value + take_width:
            bid_amount = order_depth.buy_orders[best_bid]
            if bid_amount <= 20:  # Only take small orders to avoid manipulation
                quantity = min(bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity

        # Try to clear position near fair value
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

        # Market making strategy with product-specific spreads
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else int(fair_value + make_width)
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else int(fair_value - make_width)

        # Place buy order
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            buy_price = int(best_bid_below_fair + 1)
            orders.append(Order(product, buy_price, buy_quantity))

        # Place sell order
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            sell_price = int(best_ask_above_fair - 1)
            orders.append(Order(product, sell_price, -sell_quantity))

        return orders

    def close_position(self, product: str, order_depth: OrderDepth, position: int) -> list[Order]:
        """
        Attempt to close an existing position for an inactive product.
        Uses market orders to close the position quickly.
        """
        orders = []
        
        if position == 0:
            return orders
        
        if position > 0:  # We need to sell
            if len(order_depth.buy_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                sell_quantity = min(position, order_depth.buy_orders[best_bid])
                if sell_quantity > 0:
                    orders.append(Order(product, best_bid, -sell_quantity))
        else:  # We need to buy
            if len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                buy_quantity = min(-position, -order_depth.sell_orders[best_ask])
                if buy_quantity > 0:
                    orders.append(Order(product, best_ask, buy_quantity))
        
        return orders

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        """
        Main method required by the competition.
        It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        result = {}

        # Load saved state if available
        if state.traderData and state.traderData != "SAMPLE":
            try:
                trader_data = jsonpickle.decode(state.traderData)
                if "kelp_prices" in trader_data:
                    self.kelp_prices = trader_data["kelp_prices"]
                if "resin_prices" in trader_data:
                    self.resin_prices = trader_data["resin_prices"]
                if "kelp_vwap" in trader_data:
                    self.kelp_vwap = trader_data["kelp_vwap"]
                if "resin_vwap" in trader_data:
                    self.resin_vwap = trader_data["resin_vwap"]
                if "squid_ink_prices" in trader_data:
                    self.squid_ink_prices = trader_data["squid_ink_prices"]
                if "squid_ink_vwap" in trader_data:
                    self.squid_ink_vwap = trader_data["squid_ink_vwap"]
            except:
                logger.print("Could not parse trader data")

        # Process each product only if it's active
        for product in state.order_depths.keys():
            if product in self.active_products and self.active_products[product]:
                position = state.position.get(product, 0)
                orders = self.product_orders(product, state.order_depths[product], position)
                if orders:  # Only add to result if we have orders
                    result[product] = orders
            else:
                # For inactive products, if we have a position, try to close it
                position = state.position.get(product, 0)
                if position != 0:
                    orders = self.close_position(product, state.order_depths[product], position)
                    if orders:
                        result[product] = orders

        # Save state for next iteration
        trader_data = {
            "kelp_prices": self.kelp_prices,
            "resin_prices": self.resin_prices,
            "kelp_vwap": self.kelp_vwap,
            "resin_vwap": self.resin_vwap,
            "squid_ink_prices": self.squid_ink_prices,
            "squid_ink_vwap": self.squid_ink_vwap,
        }

        serialized_trader_data = jsonpickle.encode(trader_data)
        conversions = 0  # No conversions in this strategy

        logger.flush(state, result, conversions, serialized_trader_data)

        return result, conversions, serialized_trader_data
